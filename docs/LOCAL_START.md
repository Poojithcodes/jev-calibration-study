# Local setup and shared splits — 23 October 2026

Project: jev-calibration, extracted from Mohak's supplied ZIP. No paid API calls made.
Use the existing Python 3.12 environment at ../jev-study/.venv/Scripts/python.exe.
Base data/scoring dependencies are installed; all 16 supplied tests pass.
GPU PyTorch 2.11.0+cu128 is installed. CUDA forward/backward computation and the
SetFit pilot passed on the RTX 3050 6 GB (driver 581.86, CUDA runtime 12.8).
All 69 installed packages passed the dependency consistency check.
The environment is managed with uv; pip is not installed in this environment.

## Reproduce

From the jev-calibration folder:

    ../jev-study/.venv/Scripts/python.exe scripts/prepare_splits.py
    ../jev-study/.venv/Scripts/python.exe -m pytest -q --basetemp ../work/pytest-splits-check

On another machine, create a Python 3.12 environment and install requirements.txt.
The ZIP includes the raw dataset snapshots used here; keep them with SOURCES.json.
Upstream URLs use mutable branches, so the included byte snapshots and recorded
SHA256 hashes, not a future fresh download, define these manifests.

## Exact split choices

| Dataset | Calibration | Development | Official test | Training examples per class |
|---|---:|---:|---:|---|
| AG News | 400 (100/class) | 100 (25/class) | 7,600 | 8,16,32,64 |
| Banking77 | 770 (10/class) | 385 (5/class) | 3,080 | 8,16 only |
| CLINC150 | 1,500 (10/class) | 750 (5/class) | 5,500 including 1,000 OOS | 8,16,32,64 |

These held-out sizes are explicit initial implementation choices. Split seed 20261023;
training seeds 13,42,87. Calibration/dev/train are disjoint by ID and normalized text.
Normalize with casefold and whitespace collapse. Remove training text overlapping
test, conflicting-label training duplicates, and repeated training text. Test rows
are preserved; Banking77 contains one repeated test text, reported in the audit.
Test labels/scores were not used to choose splits; test text hashes only detect overlap.
CLINC calibration/development come from in-scope validation; OOS is never training data.
Budgets are nested within each seed, and identical class order comes from label configs.

Banking77 has a smallest cleaned category of 35 examples; after holding out 15,
only 20 remain. Budgets 32/64 cannot be produced under this protocol. They are
explicitly omitted, not filled with duplicates, test rows or unequal class counts.
The current feasible grid is 60 encoder runs across two models, rather than 72.
Resolve this protocol difference together before launching the full grid.

## Files to share with Mohak

All manifests/ files plus reports/split_audit.json and data/raw/SOURCES.json.
Each dataset's *_jev_ids.txt is the union of calibration and official test IDs.
Training filenames and prediction class order follow docs/encoder_handoff.md.
The audit records every manifest count and hash. A rerun refuses to replace changed
manifest content. The development-only SetFit pilot is complete; no official test
performance has been evaluated.

## Completed SetFit pilot

AG News, budget 8 per class, seed 13: 32 training examples and 100 development examples.
Backbone: sentence-transformers/all-MiniLM-L6-v2, revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
One contrastive epoch, 20 pair-sampling iterations, batch size 16, learning rate 2e-5,
maximum sequence length 256, and a logistic-regression head.

Development accuracy: 0.80; macro-F1: 0.7964793640. Training time: 12.3144 seconds.
Prediction contract validation passed. All shared manifest, split-audit, and source
metadata hashes were unchanged. Calibration and official test examples were not evaluated.
These are pilot results, not final study numbers or frozen full-grid settings.

Artifacts: `reports/pilots/setfit/ag_news__b8__s13/`; ledger: `docs/run_ledger.csv`.
`reports/environment.json` is the historical pre-install environment snapshot;
the pilot's `environment.json` records the successful GPU environment.

To prepare an equivalent Python 3.12 environment with uv:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements-encoder.txt
uv pip install --python .venv/Scripts/python.exe torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

Restore the original ZIP's `data/raw/` snapshots and verify them against the tracked
`data/raw/SOURCES.json` before running. Raw datasets, virtual environments, caches,
and model weights are intentionally excluded from Git. The original trained checkpoint
remains in the local extracted project at the path recorded in the pilot artifacts.

The runner command is:

```powershell
.venv/Scripts/python.exe scripts/run_setfit_pilot.py --root .
```

The runner refuses to overwrite an existing pilot output directory. Preserve the tracked
pilot artifacts before a new run; the runner records the resolved upstream model revision.
Next milestone: agree on settings and the Banking77 budget limitation before the full grid.
Continue using development examples for debugging; keep final test scoring until settings freeze.
