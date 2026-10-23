"""Jev run-to-run variability on a fixed dev subset (the protocol's repeatability check).

  python scripts/variability.py --config configs/labels/banking77.json --splits train \
      --ids-file manifests/banking77_dev_ids.txt --n 100 --repeats 3

Repeat 0 is the ordinary cached call (free if the dev pilot already ran); repeats 1..K resend the
identical request under separate cache entries. Dev IDs only - never test. Writes
results/variability_<config>.json and prints label flip rate, probability spread and per-repeat
accuracy/ECE, so the paper can separate API noise from encoder seed variation.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from jevcal import data, metrics  # noqa: E402
from jevcal.contract import normalise  # noqa: E402
from jevcal.jev_client import JevClient, JevError, JevStopError  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--splits", default="train")
ap.add_argument("--ids-file", required=True)
ap.add_argument("--n", type=int, default=100, help="first n IDs of the dev manifest")
ap.add_argument("--repeats", type=int, default=3)
ap.add_argument("--concurrency", type=int, default=8)
args = ap.parse_args()
if "test" in args.splits or "test" in Path(args.ids_file).name:
    sys.exit("Variability checks use dev IDs only.")

load_dotenv()
cfg_path = Path(args.config)
cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
order = [l["name"] for l in cfg["labels"]]
criteria = {l["name"]: l["description"] for l in cfg["labels"]}
rows = {r["example_id"]: r for s in args.splits.split(",") for r in data.load_split(cfg["dataset"], s)}
ids = [x.strip() for x in Path(args.ids_file).read_text().splitlines() if x.strip()][: args.n]
items = [rows[i] for i in ids]
y = np.array([order.index(r["label"]) for r in items])

client = JevClient()
runs, snapshots = [], set()
for k in range(args.repeats + 1):
    tag = None if k == 0 else f"var{k}"
    ans = client.classify_many([r["text"] for r in items], criteria, cfg["instructions"],
                               concurrency=args.concurrency, repeat_tag=tag)
    if any(isinstance(a, JevStopError) for a in ans):
        sys.exit(f"STOPPED at repeat {k}: OpenRouter refused the account. Completed repeats are cached; "
                 "add credit and rerun the same command.")
    bad = [a for a in ans if isinstance(a, JevError)]
    if bad:
        sys.exit(f"{len(bad)} calls failed in repeat {k}, e.g. {bad[0]}; rerun to retry them.")
    runs.append(np.array([normalise(a.probabilities)[0] for a in ans]))
    snapshots |= {a.model_snapshot for a in ans}

P = np.stack(runs)                       # (R, N, K)
labels = P.argmax(2)                     # (R, N)
msp = P.max(2)
all_agree = (labels == labels[0]).all(0)
spread = (P.max(0) - P.min(0)).max(1)    # per item: largest change in any class probability
report = {
    "config": cfg_path.stem, "n_items": len(ids), "repeats": args.repeats + 1,
    "model_snapshots": sorted(snapshots),
    "label_agreement_all_repeats": float(all_agree.mean()),
    "items_with_any_label_flip": int((~all_agree).sum()),
    "prob_spread_mean": float(spread.mean()), "prob_spread_max": float(spread.max()),
    "items_with_spread_over_0.05": int((spread > 0.05).sum()),
    "msp_sd_mean": float(msp.std(0).mean()),
    "per_repeat": [{"accuracy": metrics.accuracy(p, y), "ece_15_width": metrics.ece(p, y)} for p in P],
}
out = Path("results") / f"variability_{cfg_path.stem}.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
