# SetFit throughput and compute plan — 26 October 2026

Measured the selected FP32 MPNet configuration on the RTX 3050 6 GB, microbatch 2,
accumulation 8, effective batch 16, one embedding epoch and 20 pair iterations.
Only training data were used. Each run excluded 20 initial updates from the timing
statistics and stopped early without fitting a classifier, saving a checkpoint,
or evaluating development, calibration or official-test data.

| Dataset / budget / seed | Train rows | Measured updates | Mean seconds/update | Median | p90 | Peak allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CLINC150 / 8 / 13 | 1,200 | 200 | 2.0176 | 2.1138 | 2.5640 | 2.11 |
| Banking77 / 8 / 13 | 616 | 100 | 2.1402 | 2.1066 | 2.4798 | 2.17 |

The full 3,000/1,540-update learning-rate schedules were retained. Timing warmup
is separate from learning-rate warmup. CUDA synchronization brackets each optimizer
update; fetching batches before the step-begin callback, pair preparation and the
classifier fit are outside these timings. Total embedding call elapsed time was
459.17 / 261.35 seconds, with 7.94 / 3.85 seconds outside update timers respectively.
These overheads at budget 8 cannot be extrapolated as constants to larger budgets.

CLINC first/second measured halves averaged 2.3073 / 1.7279 seconds; Banking77
averaged 2.2270 / 2.0534 seconds. Laptop speed varied, so these are planning estimates,
not deadlines or confidence intervals. Larger budgets have not been timed end to end.

## Pair preparation fix

The installed SetFit sampler originally stores every candidate pair before emitting
the fixed-iteration training sequence. CLINC b64 has 9,600 training rows and
46,084,800 candidate pairs. The new bounded trainer keeps at most 192,000 positive
and 192,000 negative candidates, preserving the prefix consumed by 20 iterations.
When a pool has fewer candidates, cycling is unchanged. Four installed-sampler
fixtures confirm exact emitted-pair equality, including capped pools and cycling.
The sampler's shuffled index arrays remain quadratic; this fix does not make the
entire preparation algorithm linear in memory or time.
The actual CLINC b64 preparation check passed: 384,000 emitted pairs, 116.92 seconds
of preparation and a 2.09 GiB peak process working set (including imported libraries).
All shared hashes remained unchanged. This checks preparation, not a full b64 training run.

## Full-grid forecast at the user's 8 hours/day

| Training hours across all seeds and budgets | SetFit | ModernBERT |
| --- | ---: | ---: |
| AG News | 2.09 | 1.51 |
| Banking77 | 8.24 | 3.36 |
| CLINC150 | 75.66 | 32.63 |
| Total | 85.99 | 37.51 |

Total: **123.50 training hours**, or **148.20 hours with a discretionary 20% buffer**.
At 8 hours/day this is 18.53 compute days (round to 19 for planning). At 16/24 hours
it is 9.26/6.18 compute days. The model loading, classifier fitting, pair preparation,
prediction and checkpoint overheads are not fully captured. The buffer is not a
statistical guarantee. ModernBERT Banking77 still uses the CLINC timing proxy;
SetFit AG News retains the earlier pilot rate for comparability. All 60 final runs
remain pending; no pilots were deducted from the totals.

The recorded Oct 28–31 training window provides only 32 local hours at the user's
availability. Local-only completion in that window is not supported by this forecast.
Even total compute days are not yet an executable local schedule: each SetFit CLINC
b64 run projects to 13.45 training hours, exceeding the 8-hour window, and the runner
does not currently resume interrupted training. The three b32 runs are about 6.73
hours each before overhead, so even those are tight with a 20% buffer.

## Concrete allocation scenarios

- **Local only:** add and verify mid-training checkpoint/resume, then agree a later
  deadline. Starting Oct 28, 19 uninterrupted daily compute slots would extend to
  approximately Nov 15, before additional analysis time. This is an estimate, not
  a promise of completion.
- **Separate all CLINC work:** keep AG News and Banking77's 36 runs local: 15.21
  training hours, 18.25 hours with buffer (roughly three 8-hour slots). The other
  24 runs account for 108.29 hours at this laptop's speed. Another GPU must be
  measured and allocated; moving work alone does not prove it will finish by Oct 31.
- **Offload CLINC b32/b64 for both models:** 12 remote runs leave about 36.87 local
  training hours, or 44.24 with buffer. That still exceeds the 32-hour local window.

No remote machine was booked, no paid job launched, no scope cut made, and no
encoder freeze record created. Decide the compute arrangement or add resume support
before freezing the executable training pipeline and launching the grid.

## Reproduction and verification

```powershell
python scripts/benchmark_setfit.py --dataset clinc150 --steps 200 --run-tag timing200
python scripts/benchmark_setfit.py --dataset banking77 --steps 100 --run-tag timing100
```

Existing directories are protected; use fresh tags for intentional repeats.
The raw update timings, plans and environments are saved beside this report.
Both benchmark means were independently recomputed and all 44 shared hashes match
before/after/current files. The repository suite passes 35 tests. No Jev key was used.
