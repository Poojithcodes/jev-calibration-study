# Jev calibration study

How many labels does a fine-tuned encoder need to match Jev on calibration and selective prediction?
Proposal and team split: the "Jev Calibration Study — Revised Research Proposal" doc.

## Current status — 26 October 2026

The [general encoder runner](docs/ENCODER_RUNNER.md) now supports all 60 planned
combinations, defaulting to development-only execution. Final predictions require
explicit test access and a matching frozen protocol record. The full grid has not
been launched; SetFit timing refinement and the encoder freeze remain pending.

ModernBERT training length is selected: ten epochs or at least 100 updates, rounded
to whole epochs. Across three AG News b8 seeds, mean development accuracy improved
from 73% to 77%. These are development selection results, not final test scores.
See the [nine-run comparison](reports/training_length/modernbert_ag_b8_20261025/SUMMARY.md)
and `configs/modernbert_selected.json`.

CLINC150 b64 ModernBERT timing is complete: 300 measured updates averaged
3.48 seconds each, with zero skipped updates. The current ModernBERT grid projects
to about 37 training hours; the full-grid estimate remains provisional.
See the [benchmark and forecast](reports/benchmarks/modernbert/clinc150__b64__s13__timing300/SUMMARY.md).
The updated ModernBERT forecast is 37.51 training hours; final evaluation is still deferred.

Update, 24 October: the MPNet SetFit and native-bf16 ModernBERT AG News b8/s13
development pilots are complete. MPNet: 83% accuracy, macro-F1 0.8278; ModernBERT
bf16: 73% accuracy, macro-F1 0.7217. Both checkpoints reload correctly, all shared
hashes remain unchanged, and official-test evaluation remains deferred. See
[24 October pilot details](docs/LOCAL_START.md#24-october-mpnet-and-bf16-pilots).

The following records the earlier baseline pilots:

Shared manifests and split audits are prepared. GPU setup and the SetFit AG News
budget-8, seed-13 development pilot are complete: 32 training examples, 100 development
examples, 80% development accuracy, macro-F1 0.7965. Calibration and official test
evaluation remain pending. The matching ModernBERT pilot is also complete: 72% development
accuracy, macro-F1 0.7151, with checkpoint reload verified. A numerical-validation repeat
at a lower FP16 loss scale completed all optimizer steps (72% accuracy, macro-F1 0.7121).
Both runs are preserved. These are single-seed development
checks, not final model comparisons. All 21 tests pass; a manifest-backed 60-run plan is
prepared but has not been launched. See [local setup and pilot details](docs/LOCAL_START.md)
and [saved results](reports/pilots/setfit/ag_news__b8__s13/result.json).

Decisions and open items since the pilots (Banking77 budgets, encoder settings to freeze by 27 Oct,
Jev run commands): [docs/protocol_decisions.md](docs/protocol_decisions.md).

## Setup (once)

```bash
pip install -r requirements.txt
copy .env.example .env      # Windows (macOS/Linux: cp); then paste your OpenRouter key into .env
python -m pytest -q         # 33 tests should pass
```

The loader downloads missing datasets automatically into `data/raw/`. For the shared
manifests in this repository, restore the original ZIP's raw snapshots and verify against
the tracked `data/raw/SOURCES.json` first: upstream URLs are mutable, and fresh downloads
may not reproduce the recorded bytes. Raw snapshots and trained weights remain local.

## Jev (Mohak)

```bash
python scripts/smoke_test.py                                                   # one call: key, credits, snapshot
python scripts/run_jev.py --config configs/labels/ag_news.json --splits train --limit 50   # pilot
```

Label configs in `configs/labels/`: `ag_news`, `banking77`, `banking77_bare` (wording ablation),
`clinc150` (150 in-scope labels; OOS scored as 1 − max probability), `clinc150_oos_option`
(adds an explicit "none of these" option for RQ3). All are drafts until Day 5: adjust wording on
dev examples only, then freeze.

Final runs take Poojith's manifests, e.g.

```bash
python scripts/run_jev.py --config configs/labels/clinc150.json --splits val,test,oos_test \
    --ids-file manifests/clinc150_jev_ids.txt --allow-test
```

where the IDs file is the calibration IDs plus the test IDs. Every response is cached in
`cache/jev_cache.sqlite`, so reruns are free and the raw responses are archived.

## AG-Fresh-2026 (both)

```bash
python scripts/collect_headlines.py --since 2026-10-22     # once a day until ~700 candidates
python scripts/make_annotation_sheets.py --n 600           # blind sheet per annotator
#   label data/fresh/sheet_poojith.csv and sheet_mohak.csv independently (docs/annotation_guide.md)
python scripts/agreement.py                                # kappa + adjudication.csv
python scripts/agreement.py --finalise                     # data/fresh/ag_fresh_2026.csv
python scripts/run_jev.py --config configs/labels/ag_news.json --dataset ag_fresh --splits test --allow-test
```

## Encoders (Poojith)

See `docs/encoder_handoff.md` for manifests, the prediction file format and the rules.

## Scoring

```bash
python scripts/score.py --preds "preds/**/*.jsonl" --rounded
```

Writes `results/per_run.csv`, `summary.csv` (mean ± SD over seeds), `paired.csv`
(encoder − Jev with paired bootstrap CIs) and `routing.csv` (Jev answers above a threshold
chosen on calibration data, encoder answers the rest).

## Conventions the code enforces

- Test splits are refused without `--allow-test`.
- Confidence = maximum class probability, never Jev's `confidence` field.
- NLL floors probabilities at 1e-3 (Jev returns many exact zeros); ECE uses 15 equal-width bins,
  plus equal-mass bins that keep tied confidences together.
- Temperature scaling: one T per run, fitted on calibration IDs (encoder logits; Jev log-probabilities).
- Comparisons fail loudly if two systems disagree on a gold label or class order for the same ID.

## Layout

```
src/jevcal/   jev_client · data · contract · metrics · scoring · fresh
scripts/      smoke_test · run_jev · collect_headlines · make_annotation_sheets · agreement · score
configs/labels/   label wording per dataset
docs/         annotation_guide · encoder_handoff · run_ledger.csv
tests/        pytest suite
```
