"""Create deterministic, text-disjoint manifests without touching the API."""
import hashlib
import json
from pathlib import Path
import random
import sys
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from jevcal.data import load_split

BUDGETS = [8, 16, 32, 64]
SEEDS = [13, 42, 87]
def fingerprint(row):
    return hashlib.sha256(' '.join(row['text'].casefold().split()).encode()).hexdigest()
def unique(rows, blocked):
    labels = defaultdict(set)
    for r in rows: labels[fingerprint(r)].add(r['label'])
    seen = set(blocked)
    result = []
    for r in rows:
        key = fingerprint(r)
        if key not in seen and len(labels[key]) == 1:
            result.append(r)
            seen.add(key)
    return result
def shuffled(rows, seed):
    rows = sorted(rows, key=lambda r: r['example_id'])
    random.Random(seed).shuffle(rows)
    return rows
def build(dataset):
    train = load_split(dataset, 'train')
    test = load_split(dataset, 'test')
    if dataset == 'clinc150': test += load_split(dataset, 'oos_test')
    order = [r['name'] for r in json.loads((ROOT/'configs/labels'/f'{dataset}.json').read_text())['labels']]
    assert set(order) == {r['label'] for r in train}
    blocked = {fingerprint(r) for r in test}
    clean = unique(train, blocked)
    cal, dev, pool = [], [], []
    if dataset == 'clinc150':
        val = unique(load_split(dataset, 'val'), blocked | {fingerprint(r) for r in clean})
        for label in order:
            group = shuffled([r for r in val if r['label'] == label], 20261023)
            assert len(group) >= 15, (label, len(group))
            cal += group[:10]; dev += group[10:15]
        pool = clean
    else:
        nc, nd = (100, 25) if dataset == 'ag_news' else (10, 5)
        for label in order:
            group = shuffled([r for r in clean if r['label'] == label], 20261023)
            assert len(group) >= nc+nd+8, (label, len(group))
            cal += group[:nc]; dev += group[nc:nc+nd]; pool += group[nc+nd:]
    capacity = min(Counter(r['label'] for r in pool).values())
    budgets = [b for b in BUDGETS if b <= capacity]
    parts = {'calibration': cal, 'dev': dev, 'test': test}
    for seed in SEEDS:
        groups = {label: shuffled([r for r in pool if r['label'] == label], seed) for label in order}
        for b in budgets:
            rows = [r for label in order for r in groups[label][:b]]
            assert Counter(r['label'] for r in rows) == {label:b for label in order}
            parts[f'train_b{b}_s{seed}'] = rows
    # Validate actual text and ID isolation, including the held-out official tests.
    for seed in SEEDS:
        sets = [parts['calibration'], parts['dev'], parts['test'], parts[f'train_b{max(budgets)}_s{seed}']]
        for i, a in enumerate(sets):
            for b in sets[i+1:]:
                assert not ({r['example_id'] for r in a} & {r['example_id'] for r in b})
                assert not ({fingerprint(r) for r in a} & {fingerprint(r) for r in b})
        for small, big in zip(budgets, budgets[1:]):
            assert {r['example_id'] for r in parts[f'train_b{small}_s{seed}']} <= {r['example_id'] for r in parts[f'train_b{big}_s{seed}']}
    parts['jev'] = cal + test
    dest = ROOT/'manifests'; dest.mkdir(exist_ok=True)
    files = {}
    for name, rows in parts.items():
        suffix = name if name.startswith('train_') else name+'_ids'
        path = dest/f'{dataset}_{suffix}.txt'
        content = ''.join(r['example_id']+'\n' for r in rows)
        if path.exists() and path.read_text() != content:
            raise RuntimeError(f'Refusing to overwrite changed manifest: {path}')
        path.write_text(content, encoding='utf-8', newline='\n')
        files[path.name] = {'count':len(rows), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    return {'classes':len(order), 'available_budgets':budgets, 'minimum_training_pool_per_class':capacity, 'omitted_budgets':[b for b in BUDGETS if b not in budgets], 'official_train':len(train), 'clean_train':len(clean),
            'removed_train_duplicates_conflicts_or_test_overlap':len(train)-len(clean),
            'calibration':len(cal), 'dev':len(dev), 'test':len(test),
            'test_duplicate_text_rows':len(test)-len({fingerprint(r) for r in test}),
            'class_order':order, 'files':files}

if __name__ == '__main__':
    import os
    os.chdir(ROOT)
    report = {'split_seed':20261023, 'budgets':BUDGETS, 'seeds':SEEDS,
              'normalization':'casefold + collapse whitespace; SHA256', 'datasets':{}}
    for ds in ['ag_news','banking77','clinc150']:
        report['datasets'][ds] = build(ds)
        print(ds, {k:v for k,v in report['datasets'][ds].items() if k not in ['files','class_order']}, flush=True)
    (ROOT/'reports').mkdir(exist_ok=True)
    (ROOT/'reports/split_audit.json').write_text(json.dumps(report, indent=2)+'\n')
