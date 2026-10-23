"""Sample candidates and write one blind annotation sheet per annotator.

  python scripts/make_annotation_sheets.py --n 600

Sampling is balanced across feed sections (to get roughly balanced classes), but the sheets
show only the text - no outlet, section or URL - and both annotators see the same items in
different random orders. Open the CSVs in Excel/Sheets and fill the `label` column with one of:
World, Sports, Business, Sci/Tech, none
"""
import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jevcal import fresh  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=600)
ap.add_argument("--seed", type=int, default=2026)
ap.add_argument("--annotators", default="poojith,mohak")
args = ap.parse_args()

rows = fresh.read_csv(fresh.CANDIDATES)
if not rows:
    sys.exit("No candidates yet - run scripts/collect_headlines.py first.")
rng = random.Random(args.seed)
by_sec = defaultdict(list)
for r in rows:
    by_sec[r["feed_section"]].append(r)
per = args.n // len(by_sec)
picked = []
for sec, items in sorted(by_sec.items()):
    rng.shuffle(items)
    picked += items[:per]
    if len(items) < per:
        print(f"warning: only {len(items)} '{sec}' candidates (wanted {per}); collect more days")

fresh.write_csv(fresh.FRESH / "annotation_items.csv", picked, fresh.FIELDS)  # key file, do not share
for name in [a.strip() for a in args.annotators.split(",")]:
    order = picked[:]
    random.Random(f"{args.seed}-{name}").shuffle(order)
    sheet = [{"item_id": r["item_id"], "text": r["text"], "label": "", "unsure": "", "note": ""} for r in order]
    fresh.write_csv(fresh.FRESH / f"sheet_{name}.csv", sheet, ["item_id", "text", "label", "unsure", "note"])
    print(f"wrote {fresh.FRESH / f'sheet_{name}.csv'} ({len(sheet)} items)")
print("Label independently. Do not discuss items until both sheets are complete.")
