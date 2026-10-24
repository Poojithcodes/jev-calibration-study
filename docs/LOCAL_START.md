# Local setup and shared splits — 23 October 2026

Project: jev-calibration, extracted from Mohak's supplied ZIP. No paid API calls made.
The original extraction uses the Python 3.12 environment at
`C:/Users/pooji/Documents/Codex/2026-10-17/new-chat/jev-study/.venv/Scripts/python.exe`.
The active Git checkout is now `C:/Users/pooji/OneDrive/Documents/ChatGPT/Jev/jev-calibration-study`.
The relative `../jev-study/` commands below apply to the original extraction only;
in the Git checkout use that absolute interpreter or create `.venv` as described below.
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

## Completed ModernBERT pilot

AG News budget 8, seed 13, using exactly the same 32 training and 100 development IDs.
`answerdotai/ModernBERT-base` revision `8949b909ec900327062f0ebf497f51aef5e6f0c8`.
Settings are recorded in `configs/modernbert_pilot.json`: 10 epochs, 2-example microbatches,
8-step gradient accumulation (effective batch 16), learning rate 2e-5, sequence length 256,
FP16 mixed precision, gradient checkpointing, and SDPA attention without compilation.

Development accuracy: 0.72; macro-F1: 0.7150977664. Training took 87.0934 seconds;
peak allocated CUDA memory was 3,111,879,168 bytes (2.90 GiB). Of 20 attempted optimizer
updates, 2 were skipped by the gradient scaler; the scheduler advanced on 18 successful
updates. Review mixed-precision scaling before freezing full-grid settings. No development
metrics were used for early stopping or checkpoint selection.

The checkpoint was reloaded and its first two development logits matched the saved run.
All 100 predictions passed the shared contract; all 44 shared hashes remained unchanged.
Calibration and official test examples were not evaluated. These single-seed development
results do not establish that one model is better than the other.

Artifacts: `reports/pilots/modernbert/ag_news__b8__s13/`. Runner:
`python scripts/run_modernbert_pilot.py`. Download the revision pinned in the config with
Hugging Face `snapshot_download` first; the runner deliberately loads local files only.
The runner refuses to overwrite an existing pilot output directory.

`python scripts/plan_encoder_grid.py` generates `reports/encoder_grid_plan.json` from
the existing manifests. The 60-run plan remains pending protocol agreement and settings
freeze; generating it launches neither training nor scoring. The test suite now has 21
passing tests, including raw-snapshot and manifest-isolation safeguards.

### Numerical validation repeat

The original run is preserved. A repeat changed only the initial FP16 loss scale from
the PyTorch default 65536 to 1024; the acceptance criterion was set before the repeat:
zero skipped optimizer updates, not improved development performance.
All 20 updates completed and the learning-rate schedule reached zero. Development accuracy
was 0.72 and macro-F1 0.7120972420; training took 89.3591 seconds with the same 2.90 GiB peak
allocation. Checkpoint reload and all prediction/hash checks passed again.

Artifacts: `reports/pilots/modernbert/ag_news__b8__s13__scale1024/`.
Recommended numerical settings for the protocol freeze: `configs/modernbert_numerical_check.json`.
Reproduction command (also refuses an existing output directory):

```powershell
python scripts/run_modernbert_pilot.py --config configs/modernbert_numerical_check.json --run-tag scale1024
```

The precision review is complete. Full-grid launch is still pending agreement on the
Banking77 8/16 budgets and freezing the recorded model settings. No final predictions
or final calibration/test metrics have been generated.

## 24 October: MPNet and bf16 pilots

Completed the first two steps from the new handoff, both on AG News budget 8, seed 13,
using the original 32 training and 100 development IDs. Older artifacts are preserved.

| Run | Dev accuracy | Dev macro-F1 | Training seconds | Peak allocated CUDA |
| --- | --- | --- | --- | --- |
| SetFit paraphrase-mpnet-base-v2 | 0.83 | 0.8277972028 | 167.5114 | 2,629,020,672 bytes |
| ModernBERT-base bf16 | 0.73 | 0.7217431297 | 78.1630 | 3,111,878,144 bytes |

MPNet revision: `6cc9279c672dc57f94445ef259b28a1b736fec8f`. One contrastive epoch,
20 pair iterations, 1,280 pairs and 80 optimizer steps. Learning rate 2e-5, sequence
length 256, FP32, logistic-regression head. Microbatch 2 with accumulation 8 keeps
effective batch size 16 within the 6 GB GPU. Gradient checkpointing is off because
the installed MPNet implementation does not support it.

Two attempts failed before any training or development evaluation: SetFit 1.2 failed
to forward the pinned revision to SentenceTransformer, then MPNet rejected gradient
checkpointing. Both failure records and ledger entries are retained. The successful
runner loads the exact local snapshot directory; no development scores were used
to choose these implementation fixes.

ModernBERT uses `configs/modernbert_bf16_pilot.json`: the same 10 epochs and effective
batch size 16 as before, now with native bf16 autocast and no GradScaler. Native bf16
support and a bf16 matrix forward/backward pass were verified on the RTX 3050. All
20 optimizer updates completed; zero were skipped. Training-length selection is
still pending and this pilot does not freeze that decision.

Successful artifact directories:

- `reports/pilots/setfit/ag_news__b8__s13__mpnet-retry2/`
- `reports/pilots/modernbert/ag_news__b8__s13__bf16/`

Both prediction files contain 100 development rows, finite logits, softmax probabilities,
and the exact shared class order. Saved checkpoints reload with matching sample logits.
All 44 shared artifact hashes are unchanged. Calibration and official test data were
not evaluated. All 21 repository tests pass. These remain single-seed development results.

After downloading the pinned snapshots, the reproduction commands are:

```powershell
python scripts/run_setfit_pilot.py --root . --model-id sentence-transformers/paraphrase-mpnet-base-v2 --revision 6cc9279c672dc57f94445ef259b28a1b736fec8f --run-tag mpnet-retry2 --micro-batch-size 2 --gradient-accumulation-steps 8
python scripts/run_modernbert_pilot.py --config configs/modernbert_bf16_pilot.json --run-tag bf16
```

These commands refuse to overwrite the existing run directories. Use new run tags
for an intentional reproduction. The next work is CLINC150 b64 timing and the
development-based training-length decision, followed by the general runner and freeze.
Banking77 budgets 8/16 are already agreed in `docs/protocol_decisions.md`.

## 24 October: CLINC150 throughput benchmark

The CLINC150 b64/s13 training-only benchmark completed 320 updates, with 300
measured after 20 timing warmup updates. Mean throughput was 3.4807 seconds/update;
peak allocated GPU memory was 2.90 GiB, and no updates were skipped. All 44 shared
hashes remain unchanged. No development, calibration or test evaluation occurred.

The current ModernBERT grid projects to 36.97 training hours. The combined 60-run
estimate is 125.65 hours before a planning buffer, but uses unmeasured SetFit
cross-dataset proxies. See the [full timing report and limitations](../reports/benchmarks/modernbert/clinc150__b64__s13__timing300/SUMMARY.md).
Training-length selection, SetFit timing refinement and the general runner remain.
All 22 tests pass. On this sandbox, pytest's default temporary directory could not
be created; rerunning with a fresh workspace `--basetemp` passed the entire suite.

Integrated teammate commit `985a59c` before publishing the benchmark, preserving
the Jev label freeze, refusal handling, variability check and figure script.
The combined suite passes all 24 tests outside the sandbox; a sandbox rerun had
failed during collection with a Windows subprocess handle error.
