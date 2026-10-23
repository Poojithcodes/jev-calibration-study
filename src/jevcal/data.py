"""One loader for all three datasets, used by both the Jev and encoder pipelines.

Sources are the official GitHub releases (not Hugging Face), downloaded once into
data/raw/ and checksummed in data/raw/SOURCES.json for the reproducibility section.

Example IDs are "<dataset>:<split>:<row index in the official file>", so they are
stable across machines. Labels are strings, never integers, so class order can never
silently disagree between systems.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path

import requests

RAW = Path("data/raw")

SOURCES = {
    "ag_news/train.csv": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv",
    "ag_news/test.csv": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv",
    "banking77/train.csv": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv",
    "banking77/test.csv": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv",
    "clinc150/data_full.json": "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json",
}

AG_CLASSES = ["World", "Sports", "Business", "Sci/Tech"]  # classes.txt order; CSV uses 1..4

SPLITS = {
    "ag_news": ["train", "test"],
    "banking77": ["train", "test"],
    # in-scope splits + the official out-of-scope splits
    "clinc150": ["train", "val", "test", "oos_train", "oos_val", "oos_test"],
    # AG-Fresh-2026: test only; scored with the ag_news label config and AG News-trained encoders
    "ag_fresh": ["test"],
}


def _fetch(rel: str, root: Path = RAW) -> Path:
    path = root / rel
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(SOURCES[rel], timeout=120)
        r.raise_for_status()
        path.write_bytes(r.content)
        log = root / "SOURCES.json"
        meta = json.loads(log.read_text()) if log.exists() else {}
        meta[rel] = {"url": SOURCES[rel], "sha256": hashlib.sha256(r.content).hexdigest(),
                     "bytes": len(r.content), "downloaded": time.strftime("%Y-%m-%d %H:%M:%S")}
        log.write_text(json.dumps(meta, indent=2))
    return path


def load_split(dataset: str, split: str, root: Path = RAW) -> list[dict]:
    """Returns [{"example_id", "text", "label"}] in official file order."""
    if split not in SPLITS.get(dataset, []):
        raise ValueError(f"{dataset} has splits {SPLITS.get(dataset)}")
    rows: list[tuple[str, str]] = []
    if dataset == "ag_news":
        with open(_fetch(f"ag_news/{split}.csv", root), encoding="utf-8", newline="") as f:
            for cls, title, desc in csv.reader(f):
                text = f"{title} {desc}".replace("\\", " ")
                rows.append((" ".join(text.split()), AG_CLASSES[int(cls) - 1]))
    elif dataset == "banking77":
        with open(_fetch(f"banking77/{split}.csv", root), encoding="utf-8", newline="") as f:
            rows = [(r["text"], r["category"]) for r in csv.DictReader(f)]
    elif dataset == "clinc150":
        data = json.loads(_fetch("clinc150/data_full.json", root).read_text(encoding="utf-8"))
        rows = [(t, y) for t, y in data[split]]
    elif dataset == "ag_fresh":
        from jevcal.fresh import FINAL, read_csv
        if not FINAL.exists():
            raise FileNotFoundError(f"{FINAL} not built yet - run scripts/agreement.py --finalise")
        return [{"example_id": r["example_id"], "text": r["text"], "label": r["label"]} for r in read_csv(FINAL)]
    return [{"example_id": f"{dataset}:{split}:{i}", "text": t, "label": y} for i, (t, y) in enumerate(rows)]


def label_set(dataset: str, root: Path = RAW) -> list[str]:
    """Sorted in-scope label names from the training split."""
    return sorted({r["label"] for r in load_split(dataset, "train", root)})
