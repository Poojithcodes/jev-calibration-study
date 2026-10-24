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
  Items 2 and 5 (training length and general runner) remain pending.
  No settings freeze or final test evaluation was performed.
- Item 4 benchmark complete: CLINC150 b64/s13, 20 timing warmup updates plus
  300 measured updates, mean 3.4807 seconds/update. ModernBERT's current grid
  projects to 36.97 training hours. The combined 125.65-hour estimate uses
  unmeasured SetFit cross-dataset proxies and needs refinement before scheduling.
  See [benchmark report](../reports/benchmarks/modernbert/clinc150__b64__s13__timing300/SUMMARY.md).
  Cloud execution has not been selected or launched.
