# ModernBERT training-length selection — 25 October 2026

Selected rule: **10 epochs or at least 100 optimizer updates, whichever is longer,
rounded up to whole epochs**. Use `configs/modernbert_selected.json`.
This selects training length; it does not launch final evaluation or mark the entire
encoder protocol frozen.

Before running, plan.json declared candidates 20, 50 and 100 minimum updates on
AG News budget 8, seeds 13/42/87. The selection rule chooses the shortest candidate
within 0.01 macro-F1 of the best three-seed mean. All nine runs start fresh from the
same pinned pretrained model, use the same bf16 settings, 32 training examples per
seed, and the same 100 held-out development examples. Each candidate uses warmup
and decay scaled to its full schedule; this compares complete training schedules.

| Minimum updates | Epochs at b8 | Seed 13 accuracy | Seed 42 | Seed 87 | Mean accuracy | Mean macro-F1 |
| --- | --- | --- | --- | --- | --- | --- |
| 20 | 10 | 73% | 73% | 73% | 73.0% | 0.7242 |
| 50 | 25 | 74% | 73% | 79% | 75.3% | 0.7429 |
| 100 | 50 | 75% | 76% | 80% | 77.0% | 0.7651 |

The 50-update candidate trails the best mean macro-F1 by 0.0222, exceeding the
predeclared 0.01 tolerance. The rule therefore selects 100. No additional candidates
were added after seeing these results. This is the best of the tested schedules,
not evidence that 100 is globally optimal. One small development set was reused
for selection; the four-point accuracy gain is not a final test result or a claim
of statistical significance. Seed variability remains visible in the table.

## Effect on the grid

The same selected rule applies across datasets and budgets. AG News b8/b16/b32/b64
now receive 100/100/104/160 updates respectively. Banking77 and CLINC150 already
exceed 100 updates under ten epochs, so their schedules stay unchanged.
The ModernBERT grid grows from 38,130 to 38,622 updates. Keeping the earlier benchmark
rates for comparability gives 37.51 ModernBERT training hours, about 32 minutes more
than the previous estimate. SetFit remains provisionally 88.68 hours using unmeasured
cross-dataset timing proxies. See forecast.json for assumptions and all 60 runs.
Training times in these pilot logs vary with laptop conditions and are not a new
controlled throughput benchmark.

## Verification and reproduction

All nine checkpoints reload with matching sample logits. An independent audit
validated all 900 development prediction rows and recomputed accuracy and macro-F1.
All 44 shared artifact hashes are unchanged, and all 510 optimizer updates completed
without skips. No calibration or official-test evaluation or Jev API calls occurred.
The repository suite passes 27 tests.

```powershell
python scripts/select_modernbert_length.py
```

The comparison driver retains completed results and refuses to overwrite unfinished
pilot directories. Its plan, per-candidate configurations, logs and aggregate results
are stored here; predictions and run details are under reports/pilots/modernbert.
Checkpoints remain local and ignored by Git. Use the selected configuration for
subsequent encoder runs; preserve the original pilot configurations for reproduction.

Next: finish the general encoder runner and refine SetFit throughput before scheduling
the full grid. Final calibration/test predictions remain deferred until protocol freeze.
