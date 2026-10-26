"""Data and provenance gates for encoder experiments; no model imports."""
import csv
import hashlib
import json
from pathlib import Path

from .encoder_data import load_development_run, read_ids, shared_hashes
from .data import load_split


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def checkpoint_hashes(path):
    path = Path(path)
    files = sorted(p for p in path.rglob('*') if p.is_file())
    if not files:
        raise ValueError('Checkpoint is empty or missing')
    return {p.relative_to(path).as_posix(): digest(p) for p in files}


def require_grid_run(root, model, dataset, budget, seed):
    plan = json.loads((Path(root)/'reports/encoder_grid_plan.json').read_text())
    if not any((r['model'], r['dataset'], r['budget'], r['seed']) ==
               (model, dataset, budget, seed) for r in plan['runs']):
        raise ValueError('Run is not in the agreed 60-run grid')


def freeze_fingerprint(root):
    root = Path(root)
    paths = ['configs/modernbert_selected.json', 'configs/setfit_selected.json']
    paths += [f'configs/labels/{d}.json' for d in ['ag_news', 'banking77', 'clinc150']]
    paths += ['scripts/run_modernbert_pilot.py', 'scripts/run_setfit_pilot.py',
              'scripts/run_encoder.py', 'src/jevcal/encoder_runner.py',
              'src/jevcal/encoder_data.py', 'src/jevcal/training_schedule.py',
              'src/jevcal/data.py', 'src/jevcal/contract.py', 'requirements.txt']
    return dict(shared_hashes=shared_hashes(root), file_hashes={p:digest(root/p) for p in paths})


def require_freeze(root, freeze_path, allow_test):
    if not allow_test:
        raise ValueError('Final mode requires --allow-test and a matching frozen protocol record')
    if freeze_path is None:
        raise ValueError('Final mode requires --freeze-record')
    freeze = json.loads(Path(freeze_path).read_text())
    if freeze.get('status') != 'frozen' or not freeze.get('frozen_at'):
        raise ValueError('Protocol record is not frozen')
    if freeze.get('fingerprint') != freeze_fingerprint(root):
        raise ValueError('Frozen protocol fingerprint differs from current files')
    return freeze


def load_final_rows(root, dataset, budget, seed, freeze_path, allow_test, include_fresh=False):
    root = Path(root)
    freeze = require_freeze(root, freeze_path, allow_test)
    order, train, _ = load_development_run(root, dataset, budget, seed)
    raw = root/'data/raw'
    sources = json.loads((raw/'SOURCES.json').read_text())
    source_files = ['clinc150/data_full.json'] if dataset=='clinc150' else [f'{dataset}/{s}.csv' for s in ['train','test']]
    for name in source_files:
        if digest(raw/name) != sources[name]['sha256']:
            raise ValueError(f'Raw snapshot hash mismatch: {name}')
    splits = ['train','val','test','oos_test'] if dataset=='clinc150' else ['train','test']
    index = {r['example_id']:r for s in splits for r in load_split(dataset,s,root=raw)}
    rows = []
    for group in ['calibration','test']:
        ids = read_ids(root/f'manifests/{dataset}_{group}_ids.txt')
        for eid in ids:
            row = index[eid]
            oos = ':oos_test:' in eid
            if oos and group != 'test':
                raise ValueError('OOS rows are forbidden in calibration')
            rows.append(dict(row, dataset=dataset, split=group, is_oos=oos))
    if include_fresh:
        if dataset != 'ag_news':
            raise ValueError('AG-Fresh applies only to AG News encoders')
        path = root/'data/fresh/ag_fresh_2026.csv'
        if not freeze.get('ag_fresh_sha256') or digest(path)!=freeze['ag_fresh_sha256']:
            raise ValueError('AG-Fresh needs its adjudicated file hash in the freeze record')
        with path.open(encoding='utf-8', newline='') as f:
            for row in csv.DictReader(f):
                if not row['example_id'].startswith('ag_fresh:') or row['label'] not in order:
                    raise ValueError('Invalid AG-Fresh ID or label')
                rows.append(dict(example_id=row['example_id'], text=row['text'], label=row['label'],
                                 dataset='ag_fresh', split='test', is_oos=False))
    if len({r['example_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate final prediction IDs')
    normalize = lambda r: ' '.join(r['text'].casefold().split())
    if {normalize(r) for r in train} & {normalize(r) for r in rows}:
        raise ValueError('Training text overlaps final prediction rows')
    return order, rows
