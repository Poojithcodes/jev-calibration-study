"""Run Jev over dataset split(s) and write predictions in the shared contract.

Pilot (random sample from train - safe to inspect while tuning wording):
  python scripts/run_jev.py --config configs/labels/ag_news.json --splits train --limit 50

Final runs use Poojith's manifest (one example_id per line) and need --allow-test:
  python scripts/run_jev.py --config configs/labels/clinc150.json --splits test,oos_test \
      --ids-file manifests/clinc150_test_ids.txt --allow-test

Any split whose name contains "test" is refused without --allow-test.
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from jevcal import data, metrics  # noqa: E402
from jevcal.contract import normalise, validate, write_jsonl  # noqa: E402
from jevcal.jev_client import DEFAULT_MODEL, JevClient, JevError  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--splits", default="train", help="comma-separated, e.g. test,oos_test")
ap.add_argument("--limit", type=int, default=50, help="random sample size when no --ids-file")
ap.add_argument("--seed", type=int, default=13)
ap.add_argument("--ids-file", default=None, help="file with one example_id per line")
ap.add_argument("--model", default=DEFAULT_MODEL)
ap.add_argument("--concurrency", type=int, default=8)
ap.add_argument("--out", default=None)
ap.add_argument("--allow-test", action="store_true")
ap.add_argument("--dataset", default=None, help="override the config's dataset, e.g. ag_fresh with ag_news.json")
args = ap.parse_args()

splits = [s.strip() for s in args.splits.split(",") if s.strip()]
if any("test" in s for s in splits) and not args.allow_test:
    sys.exit("Refusing to touch a test split without --allow-test (freeze label wording first).")

load_dotenv()
cfg_path = Path(args.config)
cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
class_order = [l["name"] for l in cfg["labels"]]
criteria = {l["name"]: l["description"] for l in cfg["labels"]}
oos_label = cfg.get("oos_label")
dataset = args.dataset or cfg["dataset"]

rows = [r for s in splits for r in data.load_split(dataset, s)]
if args.ids_file:
    wanted = [x.strip() for x in Path(args.ids_file).read_text().splitlines() if x.strip()]
    by_id = {r["example_id"]: r for r in rows}
    unknown = [w for w in wanted if w not in by_id]
    if unknown:
        sys.exit(f"{len(unknown)} ids not in splits {splits}, e.g. {unknown[:3]}")
    rows = [by_id[w] for w in wanted]
else:
    rows = random.Random(args.seed).sample(rows, min(args.limit, len(rows)))

bad = {r["label"] for r in rows} - set(class_order) - {oos_label}
if bad:
    sys.exit(f"Gold labels missing from config: {sorted(bad)[:5]}")

client = JevClient(model=args.model)
answers = client.classify_many([r["text"] for r in rows], criteria, cfg["instructions"],
                               concurrency=args.concurrency)

records, failures = [], []
for r, a in zip(rows, answers):
    if isinstance(a, JevError):
        failures.append({"example_id": r["example_id"], "error": str(a)})
        continue
    probs, raw_sum = normalise(a.probabilities)
    records.append({
        "example_id": r["example_id"], "dataset": dataset, "split": r["example_id"].split(":")[1],
        "model": "jev", "label_config": cfg_path.stem,
        "true_label": r["label"], "is_oos": r["label"] == oos_label and oos_label not in class_order,
        "predicted_label": class_order[int(np.argmax(probs))], "jev_choice": a.predicted_label,
        "class_order": class_order, "class_probabilities": probs,
        "raw_probabilities": a.probabilities, "raw_prob_sum": raw_sum, "jev_confidence": a.confidence,
        "model_snapshot": a.model_snapshot, "request_hash": a.request_hash, "request_id": a.request_id,
        "latency_ms": a.latency_ms, "from_cache": a.from_cache, "usage": a.usage,
        "label_config_status": cfg.get("status"),
    })

tag = "+".join(splits)
out = Path(args.out or f"preds/jev/{dataset}__{cfg_path.stem}__{tag}__n{len(rows)}.jsonl")
write_jsonl(records, out)
if failures:
    write_jsonl(failures, out.with_suffix(".failures.jsonl"))

problems = validate(records, class_order, {r["example_id"] for r in rows} - {f["example_id"] for f in failures})
cost = sum((x["usage"] or {}).get("cost") or 0 for x in records if not x["from_cache"])
print(f"wrote {len(records)} predictions to {out}  ({len(failures)} failed)")
print(f"model snapshots: {sorted({x['model_snapshot'] for x in records})}")
print(f"new API cost this run: ${cost:.6f}")
print("contract check:", "OK" if not problems else problems[:5])

ins = [x for x in records if not x["is_oos"]]
if ins:
    P = np.array([x["class_probabilities"] for x in ins])
    y = np.array([class_order.index(x["true_label"]) for x in ins])
    print(f"in-scope metrics (n={len(ins)}):")
    for k, v in metrics.summary(P, y).items():
        print(f"  {k:24s} {v:.4f}" if isinstance(v, float) else f"  {k:24s} {v}")
if any(x["is_oos"] for x in records) and ins:
    P_all = np.array([x["class_probabilities"] for x in records])
    is_oos = [x["is_oos"] for x in records]
    print(f"OOS AUROC (1 - max prob): {metrics.oos_auroc(P_all, is_oos):.4f}   "
          f"FPR@95TPR: {metrics.fpr_at_95_tpr(P_all, is_oos):.4f}")
