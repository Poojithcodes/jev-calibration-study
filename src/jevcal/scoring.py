"""Turn prediction files into per-run metrics, seed summaries, paired comparisons and routing results.

A *run* is one (dataset, model, label_config, budget, seed). Every run's file holds predictions
for its dataset's calibration IDs and its evaluation IDs. Calibration IDs come from
manifests/<dataset>_calibration_ids.txt (Poojith's split manifest); every other row in the run
is evaluation data. AG-Fresh-2026 has no labels of its own for calibration, so it borrows the
temperature fitted on the same model's AG News calibration rows.
"""

from __future__ import annotations

import glob
from collections import defaultdict
from pathlib import Path

import numpy as np

from jevcal import metrics as M
from jevcal.contract import read_jsonl

CALIB_SOURCE = {"ag_fresh": "ag_news"}
RUN_KEYS = ("dataset", "model", "label_config", "budget", "seed")


def run_key(r: dict) -> tuple:
    return tuple(r.get(k) for k in RUN_KEYS)


def load_runs(patterns: list[str]) -> dict[tuple, list[dict]]:
    runs = defaultdict(list)
    for pat in patterns:
        for f in sorted(glob.glob(pat, recursive=True)):
            if f.endswith(".failures.jsonl"):
                continue
            for r in read_jsonl(f):
                runs[run_key(r)].append(r)
    for k, rows in runs.items():
        ids = [r["example_id"] for r in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"run {k} has duplicate example_ids across files")
    return dict(runs)


def load_manifest(dataset: str, manifest_dir: str | Path = "manifests") -> set[str]:
    p = Path(manifest_dir) / f"{dataset}_calibration_ids.txt"
    if not p.exists():
        return set()
    return {x.strip() for x in p.read_text().splitlines() if x.strip()}


def arrays(rows: list[dict], use_logits: bool = False):
    order = rows[0]["class_order"]
    P = np.array([r["class_probabilities"] for r in rows], dtype=float)
    L = np.array([r["logits"] for r in rows], dtype=float) if use_logits else M.probs_to_logits(P)
    y = np.array([order.index(r["true_label"]) if not r.get("is_oos") else -1 for r in rows])
    oos = np.array([bool(r.get("is_oos")) for r in rows])
    return P, L, y, oos


def score_run(rows_eval: list[dict], rows_cal: list[dict], round_probs: bool = False) -> dict:
    """Raw and temperature-scaled metrics for one run."""
    has_logits = all("logits" in r for r in rows_eval + rows_cal)
    P, L, y, oos = arrays(rows_eval, has_logits)
    if round_probs:  # fairness check against Jev's 0.01 rounding
        P = np.round(P, 2)
        P = P / P.sum(1, keepdims=True)
        L = M.probs_to_logits(P)
    out = {"n_eval": int((~oos).sum()), "n_oos": int(oos.sum()), "n_calib": len(rows_cal),
           "temperature_input": "logits" if has_logits else "log_probs"}
    variants = {"raw": P}
    if rows_cal:
        _, Lc, yc, oc = arrays(rows_cal, has_logits)
        t = M.fit_temperature(Lc[~oc], yc[~oc])
        out["temperature"] = t
        variants["ts"] = M.apply_temperature(L, t)
    for name, Pv in variants.items():
        for k, v in M.summary(Pv[~oos], y[~oos]).items():
            if k != "n":
                out[f"{name}_{k}"] = v
        if oos.any():
            out[f"{name}_oos_auroc"] = M.oos_auroc(Pv, oos)
            out[f"{name}_fpr95"] = M.fpr_at_95_tpr(Pv, oos)
    return out


def split_rows(rows: list[dict], calib_ids: set[str]):
    cal = [r for r in rows if r["example_id"] in calib_ids]
    ev = [r for r in rows if r["example_id"] not in calib_ids]
    return ev, cal


def routing(jev_eval: list[dict], enc_eval: list[dict], jev_cal: list[dict], enc_cal: list[dict]) -> dict:
    """Jev answers when its max probability >= tau, otherwise the encoder answers.
    tau is chosen on calibration rows to maximise accuracy; ties go to the higher tau
    (fewer API calls). Evaluated on the shared evaluation IDs only."""
    def paired(a, b):
        bi = {r["example_id"]: r for r in b}
        a = [r for r in a if r["example_id"] in bi and not r.get("is_oos")]
        b = [bi[r["example_id"]] for r in a]
        check_same_truth(a, b)
        return a, b

    jc, ec = paired(jev_cal, enc_cal)
    je, ee = paired(jev_eval, enc_eval)
    if not jc or not je:
        return {}
    Pj, _, yc, _ = arrays(jc)
    Pe, _, _, _ = arrays(ec)
    best_tau, best_acc = 1.01, -1.0
    for tau in np.round(np.arange(0.0, 1.0001, 0.01), 2)[::-1]:
        use_j = Pj.max(1) >= tau
        acc = np.where(use_j, Pj.argmax(1) == yc, Pe.argmax(1) == yc).mean()
        if acc > best_acc + 1e-12:
            best_tau, best_acc = float(tau), float(acc)
    Qj, _, ye, _ = arrays(je)
    Qe, _, _, _ = arrays(ee)
    use_j = Qj.max(1) >= best_tau
    routed_correct = np.where(use_j, Qj.argmax(1) == ye, Qe.argmax(1) == ye)
    return {"tau": best_tau, "jev_share": float(use_j.mean()), "routed_accuracy": float(routed_correct.mean()),
            "jev_accuracy": float((Qj.argmax(1) == ye).mean()), "encoder_accuracy": float((Qe.argmax(1) == ye).mean()),
            "n": len(je)}


def check_same_truth(a: list[dict], b: list[dict]) -> None:
    """Two systems compared on the same IDs must agree on gold labels and class order."""
    bad = [x["example_id"] for x, y in zip(a, b) if x["true_label"] != y["true_label"]]
    if bad:
        raise ValueError(f"{len(bad)} example_ids have different true labels across systems, e.g. {bad[:3]}")
    if a and a[0]["class_order"] != b[0]["class_order"]:
        raise ValueError("class_order differs between the systems being compared")


def paired_diffs(jev_eval: list[dict], enc_eval: list[dict], n_boot: int = 2000, seed: int = 0) -> dict:
    """Encoder minus Jev on the same example IDs (raw probabilities)."""
    ji = {r["example_id"]: r for r in jev_eval if not r.get("is_oos")}
    enc = [r for r in enc_eval if r["example_id"] in ji and not r.get("is_oos")]
    if not enc:
        return {}
    check_same_truth(enc, [ji[r["example_id"]] for r in enc])
    Pe, _, y, _ = arrays(enc)
    Pj, _, _, _ = arrays([ji[r["example_id"]] for r in enc])
    out = {"n": len(enc)}
    for name, fn in [("accuracy", M.accuracy), ("ece", M.ece), ("brier", M.brier), ("aurc", M.aurc)]:
        d, lo, hi = M.paired_bootstrap(fn, Pe, Pj, y, n_boot=n_boot, seed=seed)
        out[f"{name}_diff"], out[f"{name}_lo"], out[f"{name}_hi"] = d, lo, hi
    return out
