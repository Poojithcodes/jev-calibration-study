"""Inter-annotator agreement, adjudication and the final AG-Fresh-2026 test file.

Step 1, after both sheets are filled:
  python scripts/agreement.py
  -> prints Cohen's kappa and writes data/fresh/adjudication.csv (disagreements only)

Step 2, sit together, fill `final_label` in adjudication.csv for every row, then:
  python scripts/agreement.py --finalise
  -> writes data/fresh/ag_fresh_2026.csv (the test set) and data/fresh/agreement_report.json
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sklearn.metrics import cohen_kappa_score  # noqa: E402

from jevcal import fresh  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--annotators", default="poojith,mohak")
ap.add_argument("--finalise", action="store_true")
args = ap.parse_args()

a, b = [x.strip() for x in args.annotators.split(",")]
valid = set(fresh.CLASSES) | set(fresh.EXTRA_CODES)
A = {r["item_id"]: r for r in fresh.read_csv(fresh.FRESH / f"sheet_{a}.csv")}
B = {r["item_id"]: r for r in fresh.read_csv(fresh.FRESH / f"sheet_{b}.csv")}
items = {r["item_id"]: r for r in fresh.read_csv(fresh.FRESH / "annotation_items.csv")}

problems = [f"{who}:{i} label '{r['label']}'" for who, S in ((a, A), (b, B))
            for i, r in S.items() if r["label"].strip() not in valid]
if problems:
    sys.exit(f"{len(problems)} missing/invalid labels, e.g. {problems[:5]}. Allowed: {sorted(valid)}")
if set(A) != set(B):
    sys.exit("The two sheets contain different items.")

ids = sorted(A)
la = [A[i]["label"].strip() for i in ids]
lb = [B[i]["label"].strip() for i in ids]
kappa = cohen_kappa_score(la, lb)
agree = sum(x == y for x, y in zip(la, lb))
print(f"items: {len(ids)}   raw agreement: {agree / len(ids):.3f}   Cohen's kappa: {kappa:.3f}")
print("confusions:", Counter((x, y) for x, y in zip(la, lb) if x != y).most_common(6))

adj_path = fresh.FRESH / "adjudication.csv"
if not args.finalise:
    rows = [{"item_id": i, "text": A[i]["text"], f"label_{a}": A[i]["label"], f"label_{b}": B[i]["label"],
             f"note_{a}": A[i]["note"], f"note_{b}": B[i]["note"], "final_label": ""}
            for i in ids if A[i]["label"].strip() != B[i]["label"].strip()]
    fresh.write_csv(adj_path, rows, list(rows[0]) if rows else ["item_id", "final_label"])
    print(f"{len(rows)} disagreements -> {adj_path}. Fill final_label together, then run with --finalise.")
    sys.exit(0)

adj = {r["item_id"]: r["final_label"].strip() for r in fresh.read_csv(adj_path)}
missing = [i for i, v in adj.items() if v not in valid]
if missing:
    sys.exit(f"{len(missing)} adjudication rows have no valid final_label, e.g. {missing[:5]}")
final, dropped = [], Counter()
for i in ids:
    lab = adj.get(i, la[ids.index(i)])
    if lab not in fresh.CLASSES:
        dropped[lab] += 1
        continue
    it = items[i]
    final.append({"example_id": f"ag_fresh:test:{len(final)}", "item_id": i, "text": it["text"], "label": lab,
                  "url": it["url"], "published_utc": it["published_utc"], "outlet": it["outlet"]})
fresh.write_csv(fresh.FINAL, final, list(final[0]))
report = {"annotators": [a, b], "items_labelled": len(ids), "raw_agreement": agree / len(ids),
          "cohen_kappa": kappa, "disagreements_adjudicated": len(adj), "dropped": dict(dropped),
          "final_size": len(final), "class_counts": dict(Counter(r["label"] for r in final)),
          "published_range": [min(r["published_utc"] for r in final), max(r["published_utc"] for r in final)]}
(fresh.FRESH / "agreement_report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
