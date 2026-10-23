"""Shared prediction contract. Every system (Jev, SetFit, ModernBERT) writes JSONL
records in this shape, so one scoring script handles all of them."""

from __future__ import annotations

import json
import math
from pathlib import Path

REQUIRED = ["example_id", "dataset", "split", "model", "true_label", "predicted_label",
            "class_order", "class_probabilities"]
# Encoders also fill: budget, seed. Jev also fills: model_snapshot, request_hash,
# request_id, latency_ms, usage, raw_prob_sum.


def normalise(probs: list[float]) -> tuple[list[float], float]:
    s = float(sum(probs))
    if s <= 0:
        raise ValueError("probabilities sum to zero")
    return [p / s for p in probs], s


def write_jsonl(records: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def validate(records: list[dict], class_order: list[str], expected_ids: set[str] | None = None,
             sum_tol: float = 1e-6) -> list[str]:
    """Returns a list of problems; empty list means the file is safe to score."""
    problems = []
    seen = set()
    for i, r in enumerate(records):
        missing = [k for k in REQUIRED if k not in r]
        if missing:
            problems.append(f"row {i}: missing {missing}")
            continue
        if r["class_order"] != class_order:
            problems.append(f"{r['example_id']}: class_order differs from the label map")
        p = r["class_probabilities"]
        if len(p) != len(class_order):
            problems.append(f"{r['example_id']}: {len(p)} probabilities for {len(class_order)} classes")
        elif not all(isinstance(x, (int, float)) and math.isfinite(x) and x >= 0 for x in p):
            problems.append(f"{r['example_id']}: non-finite or negative probability")
        elif abs(sum(p) - 1) > sum_tol:
            problems.append(f"{r['example_id']}: probabilities sum to {sum(p):.6f} (normalise first)")
        if r["predicted_label"] not in class_order:
            problems.append(f"{r['example_id']}: predicted label outside class_order")
        if r["true_label"] not in class_order and not r.get("is_oos"):
            problems.append(f"{r['example_id']}: true label outside class_order (set is_oos for OOS rows)")
        if r["example_id"] in seen:
            problems.append(f"{r['example_id']}: duplicate example_id")
        seen.add(r["example_id"])
    if expected_ids is not None:
        lost = expected_ids - seen
        extra = seen - expected_ids
        if lost:
            problems.append(f"{len(lost)} expected IDs missing, e.g. {sorted(lost)[:3]}")
        if extra:
            problems.append(f"{len(extra)} unexpected IDs, e.g. {sorted(extra)[:3]}")
    return problems
