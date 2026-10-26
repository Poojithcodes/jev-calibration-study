# General encoder runner

`scripts/run_encoder.py` supports the agreed 60 combinations of model, dataset,
budget and seed. It uses the selected MPNet SetFit settings and ModernBERT's
ten-epoch / 100-update minimum rule. The pinned snapshots and original raw files
must already be available locally. No Jev key or network API is used.

## Development checks

```powershell
python scripts/run_encoder.py --model modernbert --dataset ag_news --budget 8 --seed 13 --run-tag check1 --dry-run
python scripts/run_encoder.py --model setfit --dataset ag_news --budget 8 --seed 42 --run-tag check1
```

Development is the default phase. A dry run validates grid membership, configuration,
raw training snapshots and shared manifests without importing model libraries or
training. Normal runs delegate to the verified pilot backends and save development
predictions, a local checkpoint, metrics, environment and hashes under
`reports/pilots/<model>/<dataset>__b<budget>__s<seed>__runner-development-<tag>/`.
Tags prevent accidental overwrites; an existing output directory is rejected.

## Final runs after protocol freeze

Final mode requires both `--allow-test` and a JSON `--freeze-record` with:

- `status`: `frozen`
- `frozen_at`: the actual agreed freeze date/time
- `fingerprint`: the exact result of `jevcal.encoder_runner.freeze_fingerprint(root)`
- Optional `ag_fresh_sha256`: SHA-256 of the final adjudicated
  `data/fresh/ag_fresh_2026.csv` when including fresh news.

The fingerprint binds shared manifests, source metadata, selected settings, label
orders and runner/training code. A draft or mismatched record is rejected before
test data are loaded. Creating this record records a real protocol decision; the
runner does not automatically freeze settings. No frozen encoder record has been
created as part of runner implementation.

After that decision, an example final command is:

```powershell
python scripts/run_encoder.py --model modernbert --dataset clinc150 --budget 8 --seed 13 --phase final --run-tag frozen1 --freeze-record configs/encoder_freeze.json --allow-test
```

Training uses only the manifest's training rows. The backend also writes a dev
diagnostic and verifies checkpoint reload; that diagnostic never changes the
training schedule or selects a checkpoint. The final path loads calibration and
test rows only after training and a repeated freeze check. CLINC150 includes
`oos_test`, never OOS training or OOS calibration rows. It generates logits and
softmax probabilities in the exact shared class order, with calibration/test roles
recorded in `split`; scoring uses manifest IDs to identify calibration rows.

One validated file is written atomically to
`preds/<model>/<dataset>__b<budget>__s<seed>.jsonl`. Existing files are refused.
It includes every calibration and test ID. `final_result.json` records prediction
and freeze-record hashes; the ledger gets a final-prediction entry. Weights remain
ignored locally; final predictions follow the existing ignored `preds/` convention.
Share those files separately when assembling the study results.

If training completed but final inference failed, repeat the identical command
with `--predict-only`. It requires a completed runner provenance record and matching
checkpoint file hashes. This resumes inference from the beginning; it does not
resume partially completed training. A failed training run needs a new run tag.

For AG News, add `--include-fresh` when the adjudicated file and its recorded hash
are available. Its rows join the same prediction file with dataset `ag_fresh`.
Plan AG News final generation after adjudication if this combined file is required;
existing final files are not silently appended to or replaced.

## Remaining execution work

The runner is not the 60-run experiment itself. SetFit throughput was measured on
26 October; see the [compute plan](../reports/benchmarks/setfit/SUMMARY.md). At 8 hours/day,
some runs require training resume support or another GPU arrangement. Resolve this,
agree the final compute schedule, record the settings freeze, then launch the grid.
No official-test scores should guide any settings changes.
