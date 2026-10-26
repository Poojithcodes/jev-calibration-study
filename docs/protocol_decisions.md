# Protocol decisions and open items

Agreed changes since the 23 October pilots. Everything here is decided on development data only;
nothing below may be chosen by looking at calibration or test results.

## Decided

| Date | Decision | Reason |
| --- | --- | --- |
| 24 Oct | Banking77 uses budgets 8 and 16 only; full grid is 60 runs | Smallest class has 35 training examples; after 15 held out for calibration and dev, 32 and 64 per class are impossible. AG News and CLINC150 keep 8–64 |
| 24 Oct | Split manifests in `manifests/` are frozen | Independently re-checked on 24 Oct: no ID or normalised-text overlap between train, calibration, dev and test; budgets nested per seed; exact per-class counts; CLINC OOS only in test |
| 24 Oct | Schedule shifted: Day 1 = 23 Oct, Day 14 = 5 Nov; settings freeze on Day 5 = 27 Oct | Lost days before the start |

## To settle before the Day 5 freeze (Poojith)

1. **SetFit backbone → `sentence-transformers/paraphrase-mpnet-base-v2`.** all-MiniLM-L6-v2 (~22M parameters) against ModernBERT-base (~150M) invites "SetFit was handicapped". mpnet is the SetFit paper's default and similar in size. Rerun the AG News b8 s13 dev pilot.
2. **ModernBERT training length.** 10 fixed epochs gives 20 optimizer updates at AG News b8 (loss still falling, 0.36 → 0.30) and 6,000 at CLINC b64. Choose a rule such as "10 epochs or at least N updates, whichever is more" on dev data, then fix it for every run.
3. **ModernBERT precision → bf16.** The RTX 3050 supports it; no GradScaler, so no skipped updates and no loss-scale variants.
4. **GPU forecast.** Time a few hundred updates of ModernBERT on CLINC150 b64 before launching. The ModernBERT grid is ~38,000 updates in total. If the forecast misses Day 9, move CLINC b32/b64 to Kaggle or Colab first; see the scope-cut table in the proposal.
5. **General runner.** One script per (dataset, model, budget, seed) writing a single prediction file with calibration IDs + test IDs (+ `ag_fresh` test for AG News models), per `docs/encoder_handoff.md`. Test predictions only after the freeze.

## Jev side (Mohak)

Waiting on OpenRouter credits (card debit dated 24 Oct). Then:

```
python scripts/smoke_test.py
python scripts/run_jev.py --config configs/labels/ag_news.json   --splits train --ids-file manifests/ag_news_dev_ids.txt
python scripts/run_jev.py --config configs/labels/banking77.json --splits train --ids-file manifests/banking77_dev_ids.txt
python scripts/run_jev.py --config configs/labels/banking77_bare.json --splits train --ids-file manifests/banking77_dev_ids.txt
python scripts/run_jev.py --config configs/labels/clinc150.json  --splits val   --ids-file manifests/clinc150_dev_ids.txt
```

Label descriptions are adjusted on these dev results only and frozen by 27 Oct. Final runs after the freeze:

```
python scripts/run_jev.py --config configs/labels/ag_news.json   --splits train,test --ids-file manifests/ag_news_jev_ids.txt --allow-test
python scripts/run_jev.py --config configs/labels/banking77.json --splits train,test --ids-file manifests/banking77_jev_ids.txt --allow-test
python scripts/run_jev.py --config configs/labels/clinc150.json  --splits val,test,oos_test --ids-file manifests/clinc150_jev_ids.txt --allow-test
python scripts/run_jev.py --config configs/labels/clinc150_oos_option.json --splits val,test,oos_test --ids-file manifests/clinc150_jev_ids.txt --allow-test
```

Headline collection for AG-Fresh-2026 runs daily from 24 Oct (`scripts/collect_headlines.py --since 2026-10-22`);
annotation sheets on 26 Oct; both annotators label independently on 26–27 Oct.

## Label wording log (dev data only)

| Date | Config | Change | Evidence (dev pilot, Jev 1.13-20260917) |
| --- | --- | --- | --- |
| 24 Oct | ag_news | Business and Sci/Tech descriptions: tech-company news (lawsuits, earnings, patents) belongs to Sci/Tech, matching AG News's labelling convention | Round 0: 83.0% acc, ECE 0.132; 6 of 17 errors were Sci/Tech -> Business |
| 24 Oct | banking77 | get_physical_card now covers "finding the PIN for a new card" | Round 0: 78.2% acc, ECE 0.119; top confusion get_physical_card -> change_pin (5 of 5 were PIN questions) |

Bare-label ablation, round 0 on Banking77 dev: 77.4% acc, ECE 0.102 (descriptions +0.8 pts accuracy).
CLINC150 dev round 0: 94.7% acc, ECE 0.017 — no wording change; strong in-domain calibration is itself a contamination question for RQ2.
Remaining errors on both datasets are mostly gold-label ambiguity (e.g. AG News "World" labels on Ryder Cup and F1 stories); no further wording rounds, to avoid tuning to dev noise.

## Encoder implementation progress — 24 Oct

- Item 1 pilot complete: SetFit with paraphrase-mpnet-base-v2, AG News b8/s13,
  83% development accuracy and macro-F1 0.8278. Effective batch 16 uses microbatch 2
  and accumulation 8; two pre-training compatibility failures are preserved in the ledger.
- Item 3 pilot complete: native bf16 ModernBERT, same training IDs and 10-epoch
  schedule, 73% development accuracy and macro-F1 0.7217. No GradScaler; all 20
  optimizer updates completed. The earlier fp16 pilots are retained.
- Both checkpoints reload successfully; predictions and shared hashes validate.
  Items 2 and 5 were implemented on 25 Oct; see the updates below.
  No settings freeze or final test evaluation was performed.
- Item 4 benchmark complete: CLINC150 b64/s13, 20 timing warmup updates plus
  300 measured updates, mean 3.4807 seconds/update. ModernBERT's current grid
  projects to 36.97 training hours. The combined 125.65-hour estimate uses
  unmeasured SetFit cross-dataset proxies and needs refinement before scheduling.
  See [benchmark report](../reports/benchmarks/modernbert/clinc150__b64__s13__timing300/SUMMARY.md).
  Cloud execution has not been selected or launched.

## Encoder training length selected — 25 Oct

Nine development-only AG News b8 runs compared minimums of 20, 50 and 100 updates
across seeds 13/42/87. The predeclared rule selected **10 epochs or at least 100
updates, whichever is longer**, rounded to whole epochs. Mean dev accuracy was
73.0%, 75.3% and 77.0%; mean macro-F1 was 0.7242, 0.7429 and 0.7651 respectively.
Use `configs/modernbert_selected.json` for subsequent runs. This does not freeze all
encoder settings or authorize test-driven adjustments. SetFit timing refinement
remains pending. See the [selection report](../reports/training_length/modernbert_ag_b8_20261025/SUMMARY.md)
for the fixed selection criterion, per-seed results, limitations and revised forecast.

## General encoder runner — 25 Oct

`scripts/run_encoder.py` handles both encoders across the agreed grid, with a
development default and guarded final mode. SetFit now accepts dataset, budget and
seed instead of hard-coding AG News b8/s13. Final mode verifies a protocol fingerprint
before loading held-out evaluation data, validates prediction contracts, and saves
one calibration/test file per run (including AG-Fresh when requested and hash-bound).
All 60 development preflight checks pass. No frozen encoder record or final test
predictions were created. See [runner instructions](ENCODER_RUNNER.md).

## Jev side frozen (24 Oct)

All five label configs are frozen. Jev's final runs depend only on these configs and the frozen
manifests, not on encoder settings, so they run now, in this order (most important first, in case
the account is stopped partway):

1. CLINC150 (calibration + test + out-of-scope test)
2. AG News (calibration + test)
3. Banking77 (calibration + test)
4. CLINC150 with the explicit "none of these" option (RQ3)
5. Banking77 with bare label names (wording ablation)

AG-Fresh-2026 runs once its labels are adjudicated. Then a repeatability check on 100 dev items
per dataset, 4 calls each (`scripts/variability.py`).

If OpenRouter refuses the account (HTTP 401/402/403), `run_jev.py` stops at once, keeps every
finished answer in the cache, and exits with code 3; rerunning the same command resumes for free.

Figures: after `scripts/score.py`, run `python scripts/make_figures.py --preds "preds/**/*.jsonl"`
for budget curves (Jev as reference lines), reliability diagrams and risk–coverage curves per dataset.
