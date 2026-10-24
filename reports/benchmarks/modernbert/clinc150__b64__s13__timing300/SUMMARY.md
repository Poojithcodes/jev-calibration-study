# CLINC150 ModernBERT timing — 24 October 2026

Completed a training-only benchmark on the RTX 3050 6 GB laptop GPU, using the
frozen CLINC150 budget-64, seed-13 training manifest (9,600 examples).
Native bf16, microbatch 2, accumulation 8, effective batch 16, gradient checkpointing,
and maximum sequence length 256 match the existing bf16 configuration.

The run stopped after 320 updates: 20 excluded timing warmup updates and 300 measured
updates. CUDA synchronization brackets each update, including batch preparation,
forward/backward computation and optimizer/scheduler work. The original 6,000-update
learning-rate schedule was retained; timing warmup is distinct from LR warmup.
Only 5,120 training examples were processed, so this is not a completed trained model.

| Measurement | Result |
| --- | ---: |
| Mean seconds/update | 3.4807 |
| Median seconds/update | 3.7156 |
| 90th percentile seconds/update | 4.5994 |
| First / second measured half mean | 3.6636 / 3.2979 seconds |
| Total training elapsed | 1,120.55 seconds (18.68 minutes) |
| Peak allocated CUDA memory | 3,114,120,704 bytes (2.90 GiB) |
| Skipped updates | 0 |

Training token lengths had median 11, 99th percentile 23 and maximum 39. These short
inputs and the observed laptop operating conditions limit extrapolation to other
datasets or devices. Point-in-time GPU readings are saved in runtime_observations.json;
they do not establish thermal throttling or average GPU utilization.

## Provisional forecast

All 60 final runs remain pending. The calculation assumes ModernBERT 10 epochs and
SetFit one contrastive epoch with 20 pair iterations, both effective batch 16.
The minimum-update rule has not been chosen and could change these totals.

| Training hours across all seeds/budgets | ModernBERT | SetFit |
| --- | ---: | ---: |
| AG News | 0.98 | 2.09 |
| Banking77 | 3.37 | 8.06 |
| CLINC150 | 32.63 | 78.52 |
| Total | 36.97 | 88.68 |

ModernBERT uses measured CLINC throughput, the existing AG News bf16 pilot, and a
CLINC proxy for Banking77. SetFit uses the AG News MPNet pilot for every dataset:
its Banking77 and CLINC estimates are unmeasured proxies, not reliable deadlines.
Combined training is provisionally 125.65 hours, or 150.78 hours with a discretionary
20% planning buffer (not a confidence interval). Loading, prediction, calibration,
checkpoint I/O and reruns are excluded.

A single CLINC b64 ModernBERT run projects to 5.80 hours. CLINC b32/b64 across all
three seeds accounts for 26.11 hours and remains the first cloud-offload candidate.
ModernBERT alone plus 20% buffer takes about 44.37 hours, leaving little room in a
two-day window unless the GPU runs nearly continuously. No cloud job was launched.
Measure SetFit on CLINC before treating the combined forecast as a schedule.

## Integrity and reproduction

All 44 shared hashes were independently verified unchanged. No development,
calibration or official-test evaluation occurred; no checkpoint or predictions were
saved. Raw timings and per-run estimates are in result.json and forecast.json.

```powershell
python scripts/run_modernbert_pilot.py --dataset clinc150 --budget 64 --seed 13 --config configs/modernbert_bf16_pilot.json --run-tag timing300 --benchmark-steps 300 --benchmark-warmup 20
```

Use a new run tag for reproduction: existing output directories cannot be overwritten.
Next: select the ModernBERT minimum-update rule using development data, then finish
the general runner and freeze settings before final calibration/test evaluation.
