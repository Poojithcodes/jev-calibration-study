"""Fail-closed loading of shared encoder manifests and original raw snapshots."""
import hashlib
import json
from collections import Counter
from pathlib import Path

from .data import load_split


def shared_hashes(root):
    root = Path(root)
    paths = sorted((root / 'manifests').glob('*.txt'))
    paths += [root / 'reports/split_audit.json', root / 'data/raw/SOURCES.json']
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def read_ids(path):
    ids = Path(path).read_text().splitlines()
    if not ids or any(not i.strip() for i in ids) or len(ids) != len(set(ids)):
        raise ValueError(f'Empty or duplicate IDs in {path}')
    return ids


def load_development_run(root, dataset, budget, seed):
    root = Path(root)
    manifest = root / 'manifests'
    order = [r['name'] for r in json.loads((root / f'configs/labels/{dataset}.json').read_text())['labels']]
    groups = {
        'train': read_ids(manifest / f'{dataset}_train_b{budget}_s{seed}.txt'),
        **{s: read_ids(manifest / f'{dataset}_{s}_ids.txt') for s in ['dev', 'calibration', 'test']},
    }
    names = list(groups)
    for i, name in enumerate(names):
        for other in names[i + 1:]:
            if set(groups[name]) & set(groups[other]):
                raise ValueError(f'{name}/{other} manifest overlap')
    raw = root / 'data/raw'
    sources = json.loads((raw / 'SOURCES.json').read_text())
    files = [f'{dataset}/train.csv'] if dataset != 'clinc150' else ['clinc150/data_full.json']
    for name in files:
        if hashlib.sha256((raw / name).read_bytes()).hexdigest() != sources[name]['sha256']:
            raise ValueError(f'Raw snapshot hash mismatch: {name}')
    # Never request a test or OOS split. CLINC's source JSON also contains test
    # arrays, but the shared loader returns only the requested train/val rows.
    rows = load_split(dataset, 'train', root=raw)
    if dataset == 'clinc150':
        rows += load_split(dataset, 'val', root=raw)
    index = {r['example_id']: r for r in rows}
    train, dev = [[index[i] for i in groups[g]] for g in ['train', 'dev']]
    if Counter(r['label'] for r in train) != {label: budget for label in order}:
        raise ValueError('Training class counts differ from requested budget')
    norm = lambda r: ' '.join(r['text'].casefold().split())
    train_text = [norm(r) for r in train]
    if len(set(train_text)) != len(train_text):
        raise ValueError('Repeated training text')
    heldout = [index[i] for g in ['dev', 'calibration'] for i in groups[g]]
    if set(train_text) & {norm(r) for r in heldout}:
        raise ValueError('Training text overlaps held-out text')
    return order, train, dev
