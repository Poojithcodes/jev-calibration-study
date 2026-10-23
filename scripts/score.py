"""Score every prediction file into the paper's tables.

  python scripts/score.py --preds "preds/**/*.jsonl"

Writes:
  results/per_run.csv   one row per run: raw + temperature-scaled metrics, T, counts
  results/summary.csv   mean and SD over seeds per (dataset, model, label_config, budget)
  results/paired.csv    encoder minus Jev, paired bootstrap 95% CI on the same test IDs
  results/routing.csv   Jev-first routing: tau from calibration rows, accuracy and Jev share on test

Add --rounded to also score encoders with probabilities rounded to 0.01 (Jev's precision).
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402

from jevcal import scoring as S  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--preds", nargs="+", default=["preds/**/*.jsonl"])
ap.add_argument("--manifests", default="manifests")
ap.add_argument("--out", default="results")
ap.add_argument("--rounded", action="store_true")
ap.add_argument("--n-boot", type=int, default=2000)
args = ap.parse_args()

runs = S.load_runs(args.preds)
if not runs:
    sys.exit("No prediction files matched.")
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
calib = {}


def cal_ids(ds):
    src = S.CALIB_SOURCE.get(ds, ds)
    if src not in calib:
        calib[src] = S.load_manifest(src, args.manifests)
    return calib[src]


def cal_rows(key):
    """Calibration rows for a run; AG-Fresh borrows the same model's AG News run."""
    ds = key[0]
    src_key = (S.CALIB_SOURCE.get(ds, ds),) + key[1:]
    return [r for r in runs.get(src_key, []) if r["example_id"] in cal_ids(ds)]


per_run, evals = [], {}
for key, rows in sorted(runs.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
    ev, _ = S.split_rows(rows, cal_ids(key[0]))
    cal = cal_rows(key)
    evals[key] = (ev, cal)
    variants = [False] + ([True] if args.rounded and key[1] != "jev" else [])
    for rnd in variants:
        res = S.score_run(ev, cal, round_probs=rnd)
        per_run.append({**dict(zip(S.RUN_KEYS, key)), "rounded_0.01": rnd, **res})
        if not cal:
            print(f"note: no calibration rows for {key}; temperature-scaled metrics skipped")


def write(name, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out / name} ({len(rows)} rows)")


write("per_run.csv", per_run)

# ---- seed summary ----------------------------------------------------------------------
groups = defaultdict(list)
for r in per_run:
    groups[(r["dataset"], r["model"], r["label_config"], r["budget"], r["rounded_0.01"])].append(r)
summary = []
for (ds, model, lc, budget, rnd), rs in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
    row = {"dataset": ds, "model": model, "label_config": lc, "budget": budget, "rounded_0.01": rnd, "n_seeds": len(rs)}
    for k in rs[0]:
        vals = [r.get(k) for r in rs]
        if k.startswith(("raw_", "ts_")) or k == "temperature":
            v = np.array([x for x in vals if x is not None], dtype=float)
            if len(v):
                row[f"{k}_mean"] = float(v.mean())
                row[f"{k}_sd"] = float(v.std(ddof=1)) if len(v) > 1 else None
    summary.append(row)
write("summary.csv", summary)

# ---- paired comparisons and routing vs Jev ---------------------------------------------
jev_runs = {k: v for k, v in evals.items() if k[1] == "jev"}
paired, routed = [], []
for key, (ev, cal) in evals.items():
    if key[1] == "jev" or key[3] is None:
        continue
    for jkey, (jev_ev, jev_cal) in jev_runs.items():
        if jkey[0] != key[0]:
            continue
        base = {"dataset": key[0], "model": key[1], "budget": key[3], "seed": key[4], "jev_label_config": jkey[2]}
        d = S.paired_diffs(jev_ev, ev, n_boot=args.n_boot, seed=int(key[4] or 0))
        if d:
            paired.append({**base, **d})
        rt = S.routing(jev_ev, ev, jev_cal, cal)
        if rt:
            routed.append({**base, **rt})
write("paired.csv", paired)
write("routing.csv", routed)
