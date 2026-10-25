"""Run a predeclared, development-only training-length comparison."""
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/training_length/modernbert_ag_b8_20261025'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plan = dict(dataset='ag_news', budget=8, seeds=[13, 42, 87],
                min_updates=[20, 50, 100], base_epochs=10,
                selection='Shortest candidate within 0.01 of best mean development macro-F1 across three seeds',
                tolerance=0.01, calibration_evaluated=False, official_test_evaluated=False,
                note='Fresh initialization per candidate/seed; identical other settings; whole-epoch rounding')
    plan_path = OUT / 'plan.json'
    if plan_path.exists():
        assert json.loads(plan_path.read_text()) == plan
    else:
        plan_path.write_text(json.dumps(plan, indent=2), encoding='utf-8')
    results = []
    for minimum in plan['min_updates']:
        cfg = json.loads((ROOT/'configs/modernbert_bf16_pilot.json').read_text())
        cfg.update(min_updates=minimum, status='Development-only training-length comparison')
        cfg_path = OUT / f'config_min{minimum}.json'
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding='utf-8')
        for seed in plan['seeds']:
            tag = f'length-min{minimum}-20261025'
            run = ROOT / f'reports/pilots/modernbert/ag_news__b8__s{seed}__{tag}'
            result_path = run / 'result.json'
            if not result_path.exists():
                with (OUT/f'min{minimum}_s{seed}.log').open('w', encoding='utf-8') as log:
                    print(f'Start minimum={minimum}, seed={seed}', flush=True)
                    subprocess.run([sys.executable, '-u', str(ROOT/'scripts/run_modernbert_pilot.py'),
                                    '--dataset','ag_news','--budget','8','--seed',str(seed),
                                    '--config',str(cfg_path),'--run-tag',tag], cwd=ROOT,
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
            result = json.loads(result_path.read_text())
            assert result['status']=='completed_development_only'
            assert result['shared_hashes_unchanged'] and result['checkpoint_reload_passed']
            assert not result['official_test_evaluated'] and not result['calibration_evaluated']
            results.append(dict(min_updates=minimum, seed=seed, result_path=result_path.relative_to(ROOT).as_posix(),
                                accuracy=result['development_accuracy'], macro_f1=result['development_macro_f1'],
                                train_seconds=result['train_seconds']))
            print(json.dumps(results[-1]), flush=True)
    aggregates = [dict(min_updates=m,
                       mean_macro_f1=statistics.mean(r['macro_f1'] for r in results if r['min_updates']==m),
                       mean_accuracy=statistics.mean(r['accuracy'] for r in results if r['min_updates']==m))
                  for m in plan['min_updates']]
    best = max(r['mean_macro_f1'] for r in aggregates)
    selected = min(r['min_updates'] for r in aggregates if r['mean_macro_f1'] >= best-plan['tolerance'])
    summary = dict(plan=plan, runs=results, aggregates=aggregates, selected_min_updates=selected,
                   limitation='Selection on one small development set; not a final generalization estimate')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    main()
