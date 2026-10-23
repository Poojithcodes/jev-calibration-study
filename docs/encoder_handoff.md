# Handoff for Poojith: what the encoder side must produce

Everything below exists so that one script (`scripts/score.py`) can score Jev, SetFit and
ModernBERT the same way, and so a reviewer can trace every number back to saved predictions.

## 1. Load data with the shared loader

```python
from jevcal.data import load_split
rows = load_split("banking77", "train")   # [{"example_id", "text", "label"}], label is a string
```

Datasets and splits: `ag_news` (train, test), `banking77` (train, test),
`clinc150` (train, val, test, oos_train, oos_val, oos_test), and later `ag_fresh` (test).
Example IDs look like `banking77:train:123` and are stable on every machine.

## 2. Manifests (one example_id per line, in `manifests/`)

| File | Contents |
| --- | --- |
| `<dataset>_calibration_ids.txt` | Labelled examples used only for temperature fitting and routing thresholds. Same IDs for every system. AG News and Banking77: from train, disjoint from all training pools. CLINC150: from `val`, in-scope only |
| `<dataset>_dev_ids.txt` | Debugging and wording checks. Never used for final numbers |
| `<dataset>_test_ids.txt` | Final evaluation IDs. CLINC150: `test` plus `oos_test` |
| `<dataset>_train_b<budget>_s<seed>.txt` | Training examples per class budget (8, 16, 32, 64) and seed (13, 42, 87). Nested: b8 ⊂ b16 ⊂ b32 ⊂ b64 within a seed |

Record the calibration set size per dataset in the paper; it counts as labels.

## 3. One prediction file per run

`preds/<model>/<dataset>__b<budget>__s<seed>.jsonl`, one JSON object per line, covering the
calibration IDs and the test IDs. AG News models also predict `ag_fresh` test items once that set exists
(written with `"dataset": "ag_fresh"`).

| Field | Example | Notes |
| --- | --- | --- |
| `example_id` | `"banking77:test:17"` | From the loader |
| `dataset`, `split` | `"banking77"`, `"test"` | |
| `model` | `"setfit"` or `"modernbert"` | |
| `label_config` | `null` | Jev-only field |
| `budget`, `seed` | `16`, `42` | Integers |
| `true_label` | `"card_arrival"` | String, or `"oos"` for CLINC out-of-scope rows |
| `is_oos` | `false` | `true` only for CLINC `oos_*` rows |
| `class_order` | list of strings | Exactly the label order in `configs/labels/<dataset>.json` (CLINC: `clinc150.json`, 150 in-scope labels) |
| `class_probabilities` | list of floats | Softmax of logits, same order, sums to 1 |
| `predicted_label` | `"card_arrival"` | Argmax |
| `logits` | list of floats | Raw logits before softmax. Temperature scaling uses these |
| `checkpoint`, `train_seconds`, `config` | | For the run ledger and reproducibility |

Write with `jevcal.contract.write_jsonl` and check with `jevcal.contract.validate(records, class_order, expected_ids)`;
an empty list means the file is ready to share.

## 4. Rules from the protocol

- No test IDs in training, early stopping, hyperparameter choice or calibration.
- Fixed hyperparameters across budgets; record them once in the ledger.
- CLINC encoders train on the 150 in-scope labels only; no OOS examples in training or calibration.
- After scoring, run `python scripts/score.py --preds "preds/**/*.jsonl" --rounded` to produce all tables.

## 5. Run ledger

Add one row to `docs/run_ledger.csv` per finished run (Jev runs included).
