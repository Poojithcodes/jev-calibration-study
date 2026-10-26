# General runner validation — completed 26 October 2026

All 60 planned configurations passed development preflight checks. The unified
entry point completed real GPU development runs for both backends:

| Backend | AG News budget / seed | Updates | Dev accuracy | Dev macro-F1 |
| --- | --- | --- | --- | --- |
| SetFit MPNet | 8 / 42 | 80 | 84% | 0.8352 |
| ModernBERT bf16 | 8 / 13 | 100 | 75% | 0.7443 |

Both saved checkpoints reload; ModernBERT reproduces the selected schedule's earlier
seed-13 scores. The unified inference path also reproduced all 200 saved development
rows: maximum absolute logit difference was 1.20e-6 for SetFit and zero for ModernBERT.
All 44 shared hashes were independently verified unchanged.
These are runner validation results, not new hyperparameter choices
or final test results. The ModernBERT smoke run spanned a long interruption and its
28,687.85-second elapsed timer is not a usable throughput estimate. Keep the earlier
controlled benchmark for scheduling.

The test suite passes 33 tests, including synthetic final-output generation,
freeze rejection, snapshot corruption, OOS handling, AG-Fresh hash binding and
overwrite protection. Actual CLI calls also reject missing final-mode gates.
No actual calibration or official-test data were evaluated, no frozen encoder
record was created, and no Jev API call was made.

Grid coverage is preflight coverage: the full 60 training runs have not started.
Remaining work is SetFit timing refinement, compute scheduling and the protocol
freeze, followed by full training and prediction generation.
