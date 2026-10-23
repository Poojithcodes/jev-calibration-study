import hashlib
import json
from pathlib import Path

import pytest
from jevcal import encoder_data as E


@pytest.fixture
def sample(tmp_path, monkeypatch):
    for folder in ['manifests', 'configs/labels', 'data/raw/ag_news']:
        (tmp_path / folder).mkdir(parents=True)
    (tmp_path / 'configs/labels/ag_news.json').write_text(json.dumps({'labels':[{'name':'A'}, {'name':'B'}]}))
    contents = {'train_b1_s13':['ag_news:train:0','ag_news:train:1'],
                'dev_ids':['ag_news:train:2'], 'calibration_ids':['ag_news:train:3'], 'test_ids':['ag_news:test:0']}
    for name, ids in contents.items():
        (tmp_path / f'manifests/ag_news_{name}.txt').write_text('\n'.join(ids)+'\n')
    raw = b'fixture raw bytes'
    (tmp_path / 'data/raw/ag_news/train.csv').write_bytes(raw)
    (tmp_path / 'data/raw/SOURCES.json').write_text(json.dumps({'ag_news/train.csv': {'sha256':hashlib.sha256(raw).hexdigest()}}))
    rows = [dict(example_id=f'ag_news:train:{i}',text=f'text {i}',label='A' if i%2==0 else 'B') for i in range(4)]
    calls = []
    def loader(dataset, split, root):
        calls.append(split)
        assert split == 'train'
        return rows
    monkeypatch.setattr(E, 'load_split', loader)
    return tmp_path, rows, calls


def test_only_train_snapshot_loaded(sample):
    root, _, calls = sample
    order, train, dev = E.load_development_run(root,'ag_news',1,13)
    assert order == ['A','B'] and len(train) == 2 and len(dev) == 1
    assert calls == ['train']


def test_corrupt_raw_rejected_before_loading(sample):
    root, _, calls = sample
    (root / 'data/raw/ag_news/train.csv').write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        E.load_development_run(root,'ag_news',1,13)
    assert calls == []


def test_manifest_overlap_rejected(sample):
    root, _, calls = sample
    (root / 'manifests/ag_news_dev_ids.txt').write_text('ag_news:train:0\n')
    with pytest.raises(ValueError, match='overlap'):
        E.load_development_run(root,'ag_news',1,13)
    assert calls == []


def test_normalized_text_overlap_rejected(sample):
    root, rows, _ = sample
    rows[2]['text'] = ' TEXT   0 '
    with pytest.raises(ValueError, match='text overlaps'):
        E.load_development_run(root,'ag_news',1,13)


def test_unbalanced_training_rejected(sample):
    root, rows, _ = sample
    rows[1]['label'] = 'A'
    with pytest.raises(ValueError, match='class counts'):
        E.load_development_run(root,'ag_news',1,13)
