"""Paper figures from prediction files and score.py's tables.

  python scripts/score.py --preds "preds/**/*.jsonl"      # writes results/*.csv first
  python scripts/make_figures.py --preds "preds/**/*.jsonl"

Writes figures/<dataset>_budget_curves.(png|pdf), <dataset>_reliability.(png|pdf) and
<dataset>_risk_coverage.(png|pdf). Every figure is drawn from evaluation IDs only (calibration
IDs removed via manifests), and every number in it also exists in results/*.csv.

Encoding: Jev = blue (dashed when temperature-scaled), SetFit = orange, ModernBERT = aqua;
each system also has its own marker, and every panel has a legend, so colour is never the
only cue.
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from jevcal import metrics as M  # noqa: E402
from jevcal import scoring as S  # noqa: E402

STYLE = {  # validated categorical slots 1-3 (light surface)
    "jev": dict(color="#2a78d6", marker="o", label="Jev (zero-shot)"),
    "setfit": dict(color="#eb6834", marker="s", label="SetFit"),
    "modernbert": dict(color="#1baf7a", marker="^", label="ModernBERT"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
TITLES = {"ag_news": "AG News", "banking77": "Banking77", "clinc150": "CLINC150", "ag_fresh": "AG-Fresh-2026"}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight",
    "lines.linewidth": 1.6, "lines.markersize": 5,
})


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    print(f"wrote {out / name}.png/.pdf")


def read_csv(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def budget_curves(summary, ds, out):
    """Metric vs labels per class; encoders mean +/- SD over seeds, Jev as flat reference lines."""
    rows = [r for r in summary if r["dataset"] == ds and r.get("rounded_0.01") in ("False", "", None)]
    panels = [("raw_accuracy", "Accuracy (higher is better)", False), ("raw_ece_15_width", "ECE (lower is better)", True),
              ("raw_aurc", "AURC (lower is better)", True)]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.5))
    fig.subplots_adjust(wspace=0.35)
    budgets_seen = set()
    for ax, (key, label, lower_better) in zip(axes, panels):
        for model in ("setfit", "modernbert"):
            pts = sorted((int(float(r["budget"])), fnum(r.get(f"{key}_mean")), fnum(r.get(f"{key}_sd")) or 0.0)
                         for r in rows if r["model"] == model and r["budget"] not in ("", "None"))
            pts = [p for p in pts if p[1] is not None]
            if not pts:
                continue
            b, m, sd = map(np.array, zip(*pts))
            budgets_seen |= set(b.tolist())
            st = STYLE[model]
            ax.errorbar(b, m, yerr=sd, color=st["color"], marker=st["marker"], label=st["label"], capsize=2)
        for r in rows:
            if r["model"] != "jev":
                continue
            st = STYLE["jev"]
            v = fnum(r.get(f"{key}_mean"))
            if v is not None:
                ax.axhline(v, color=st["color"], linewidth=1.6, label="Jev (raw)")
            ts = fnum(r.get(f"ts_{key[4:]}_mean"))
            if ts is not None:
                ax.axhline(ts, color=st["color"], linewidth=1.4, linestyle="--", label="Jev (temp.-scaled)")
            break
        ax.set_xscale("log", base=2)
        if budgets_seen:
            ax.set_xticks(sorted(budgets_seen), [str(int(x)) for x in sorted(budgets_seen)])
        ax.set_xlabel("Labels per class")
        ax.set_title(label, color=INK)
    h, l = axes[0].get_legend_handles_labels()
    if h:
        fig.legend(h, l, loc="upper center", ncol=len(h), frameon=False, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle(TITLES.get(ds, ds), x=0.01, ha="left", color=INK, fontweight="bold")
    save(fig, out, f"{ds}_budget_curves")


def pooled(runs, ds, model, budget=None):
    """Concatenate evaluation rows (in-scope only) across seeds for one system."""
    P, y = [], []
    for (d, m, _lc, b, _s), (ev, _cal) in runs.items():
        if d != ds or m != model or (budget is not None and b != budget):
            continue
        rows = [r for r in ev if not r.get("is_oos")]
        if rows:
            Pi, _, yi, _ = S.arrays(rows)
            P.append(Pi)
            y.append(yi)
    return (np.vstack(P), np.concatenate(y)) if P else (None, None)


def reliability(runs, ds, budgets, out, n_bins=10):
    systems = [("jev", None)] + [(m, b) for b in budgets for m in ("setfit", "modernbert")]
    systems = [(m, b) for m, b in systems if pooled(runs, ds, m, b)[0] is not None]
    if not systems:
        return
    fig, axes = plt.subplots(1, len(systems), figsize=(1.9 * len(systems) + 0.4, 2.2), sharey=True, squeeze=False)
    edges = np.linspace(0, 1, n_bins + 1)
    for ax, (m, b) in zip(axes[0], systems):
        P, y = pooled(runs, ds, m, b)
        conf, ok = P.max(1), P.argmax(1) == y
        idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
        st = STYLE[m]
        ax.plot([0, 1], [0, 1], color=MUTED, linewidth=0.8, linestyle=":")
        for k in range(n_bins):
            sel = idx == k
            if sel.sum() >= 5:  # skip near-empty bins rather than plot noise
                acc = ok[sel].mean()
                ax.bar((edges[k] + edges[k + 1]) / 2, acc, width=1 / n_bins - 0.01, color=st["color"], alpha=0.85)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("Confidence")
        name = st["label"].split(" (")[0] + (f", {b}/class" if b else "")
        ax.set_title(f"{name}\nECE {M.ece(P, y):.3f}", color=INK)
    axes[0][0].set_ylabel("Accuracy")
    fig.suptitle(f"{TITLES.get(ds, ds)}: reliability (seeds pooled)", x=0.01, ha="left", color=INK, fontweight="bold", y=1.08)
    save(fig, out, f"{ds}_reliability")


def risk_coverage(runs, ds, budgets, out):
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    drawn = False
    for m, b, ls in [("jev", None, "-")] + [(mm, bb, ls) for bb, ls in zip(budgets, ["-", "--"]) for mm in ("setfit", "modernbert")]:
        P, y = pooled(runs, ds, m, b)
        if P is None:
            continue
        cov, risk = M.risk_coverage(P, y)
        st = STYLE[m]
        name = st["label"].split(" (")[0] + (f", {b}/class" if b else "")
        ax.plot(cov, risk, color=st["color"], linestyle=ls, label=f"{name} (AURC {M.aurc(P, y):.3f})")
        drawn = True
    if not drawn:
        plt.close(fig)
        return
    ax.axhline(0.05, color=MUTED, linewidth=0.7, linestyle=":")
    ax.text(0.99, 0.05, "5% risk", color=MUTED, fontsize=7, va="bottom", ha="right")
    ax.set_xlabel("Coverage (share answered)")
    ax.set_ylabel("Selective risk (error rate)")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title(f"{TITLES.get(ds, ds)}: risk-coverage", color=INK, loc="left", fontweight="bold")
    save(fig, out, f"{ds}_risk_coverage")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", default=["preds/**/*.jsonl"])
    ap.add_argument("--manifests", default="manifests")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--budgets", default="8,64", help="budgets shown in reliability/risk-coverage panels")
    args = ap.parse_args()

    raw = S.load_runs(args.preds)
    runs = {}
    for key, rows in raw.items():
        cal_ids = S.load_manifest(S.CALIB_SOURCE.get(key[0], key[0]), args.manifests)
        runs[key] = S.split_rows(rows, cal_ids)
    datasets = sorted({k[0] for k in runs})
    budgets = [int(b) for b in args.budgets.split(",")]
    summary_path = Path(args.results) / "summary.csv"
    summary = read_csv(summary_path) if summary_path.exists() else []
    out = Path(args.out)
    for ds in datasets:
        avail = sorted({k[3] for k in runs if k[0] == ds and k[3] is not None})
        show = [b for b in budgets if b in avail] or avail[:1] + avail[-1:]
        if summary:
            budget_curves(summary, ds, out)
        reliability(runs, ds, sorted(set(show)), out)
        risk_coverage(runs, ds, sorted(set(show))[:2], out)


if __name__ == "__main__":
    main()
