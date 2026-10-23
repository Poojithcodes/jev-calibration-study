"""Write a manifest-backed training plan without training or reading test labels."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from jevcal.encoder_data import read_ids, shared_hashes


def build_plan(root):
    runs = []
    for dataset, budgets in [('ag_news', [8,16,32,64]), ('banking77', [8,16]), ('clinc150', [8,16,32,64])]:
        for model in ['setfit', 'modernbert']:
            for budget in budgets:
                for seed in [13,42,87]:
                    manifest = root / f'manifests/{dataset}_train_b{budget}_s{seed}.txt'
                    train_ids = read_ids(manifest)
                    for split in ['calibration', 'dev', 'test']:
                        if set(train_ids) & set(read_ids(root / f'manifests/{dataset}_{split}_ids.txt')):
                            raise ValueError(f'Train/{split} overlap in {manifest}')
                    classes = len(json.loads((root / f'configs/labels/{dataset}.json').read_text())['labels'])
                    if len(train_ids) != classes * budget:
                        raise ValueError(f'Incorrect training count in {manifest}')
                    runs.append(dict(dataset=dataset, model=model, budget=budget, seed=seed,
                                     training_examples=len(train_ids), manifest=manifest.relative_to(root).as_posix(),
                                     status='pending_protocol_freeze'))
    return dict(status='plan_only', run_count=len(runs),
                gates=['Agree Banking77 budgets 8/16 with collaborator',
                       'Freeze model-specific training settings before full-grid training',
                       'Keep official test evaluation deferred until settings freeze'],
                shared_hashes=shared_hashes(root), runs=runs)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    a = p.parse_args()
    plan = build_plan(a.root.resolve())
    out = a.root / 'reports/encoder_grid_plan.json'
    out.write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print(f'{plan["run_count"]} runs planned; no training or test scoring launched. {out}')
