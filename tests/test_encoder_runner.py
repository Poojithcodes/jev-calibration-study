import hashlib
import json
import pytest
from jevcal import encoder_runner as E


def test_final_gate_blocks_before_any_data_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(E,'load_development_run',lambda *args: pytest.fail('Data loaded before gate'))
    with pytest.raises(ValueError,match='allow-test'):
        E.load_final_rows(tmp_path,'ag_news',8,13,None,False)
    with pytest.raises(ValueError,match='freeze-record'):
        E.load_final_rows(tmp_path,'ag_news',8,13,None,True)


def test_freeze_rejects_draft_and_changed_fingerprints(tmp_path, monkeypatch):
    path=tmp_path/'freeze.json'
    monkeypatch.setattr(E,'freeze_fingerprint',lambda root: {'hash':'current'})
    path.write_text(json.dumps(dict(status='draft')))
    with pytest.raises(ValueError,match='not frozen'):
        E.require_freeze(tmp_path,path,True)
    path.write_text(json.dumps(dict(status='frozen',frozen_at='2026-10-27',fingerprint={'hash':'old'})))
    with pytest.raises(ValueError,match='differs'):
        E.require_freeze(tmp_path,path,True)


def test_clinc_final_membership_and_oos_flags(tmp_path,monkeypatch):
    (tmp_path/'manifests').mkdir()
    (tmp_path/'data/raw/clinc150').mkdir(parents=True)
    (tmp_path/'data/raw/clinc150/data_full.json').write_bytes(b'fixture')
    (tmp_path/'data/raw/SOURCES.json').write_text(json.dumps({'clinc150/data_full.json':{'sha256':hashlib.sha256(b'fixture').hexdigest()}}))
    (tmp_path/'manifests/clinc150_calibration_ids.txt').write_text('clinc150:val:0\n')
    (tmp_path/'manifests/clinc150_test_ids.txt').write_text('clinc150:test:0\nclinc150:oos_test:0\n')
    monkeypatch.setattr(E,'require_freeze',lambda *args: {})
    monkeypatch.setattr(E,'load_development_run',lambda *args: (['A','B'],[dict(text='training')],[]))
    calls=[]
    def loader(dataset,split,root):
        calls.append(split)
        return [dict(example_id=f'clinc150:{split}:0',text=f'heldout {split}',label='oos' if split=='oos_test' else 'A')]
    monkeypatch.setattr(E,'load_split',loader)
    order,rows=E.load_final_rows(tmp_path,'clinc150',8,13,None,True)
    assert order==['A','B'] and len(rows)==3
    assert [r['split'] for r in rows]==['calibration','test','test']
    assert [r['is_oos'] for r in rows]==[False,False,True]
    assert 'oos_train' not in calls and 'oos_val' not in calls
    (tmp_path/'data/raw/clinc150/data_full.json').write_bytes(b'corrupt')
    calls.clear()
    with pytest.raises(ValueError,match='hash mismatch'):
        E.load_final_rows(tmp_path,'clinc150',8,13,None,True)
    assert calls==[]


def test_unplanned_run_and_tampered_checkpoint_rejected(tmp_path):
    (tmp_path/'reports').mkdir()
    (tmp_path/'reports/encoder_grid_plan.json').write_text(json.dumps({'runs':[dict(model='setfit',dataset='banking77',budget=8,seed=13)]}))
    with pytest.raises(ValueError,match='agreed'):
        E.require_grid_run(tmp_path,'setfit','banking77',64,13)
    checkpoint=tmp_path/'checkpoint'
    checkpoint.mkdir()
    (checkpoint/'weights').write_bytes(b'original')
    before=E.checkpoint_hashes(checkpoint)
    (checkpoint/'weights').write_bytes(b'changed')
    assert E.checkpoint_hashes(checkpoint)!=before


def test_final_cli_writes_valid_contract_and_refuses_overwrite(tmp_path, monkeypatch):
    import importlib.util
    import sys
    import numpy as np
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('encoder_cli',Path(__file__).parents[1]/'scripts/run_encoder.py')
    cli=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    (tmp_path/'configs').mkdir()
    (tmp_path/'docs').mkdir()
    (tmp_path/'configs/modernbert_selected.json').write_text(json.dumps(dict(max_length=256,micro_batch_size=2)))
    freeze=tmp_path/'freeze.json'; freeze.write_text('{}')
    monkeypatch.setattr(cli,'require_grid_run',lambda *args: None)
    monkeypatch.setattr(cli,'require_freeze',lambda *args: {})
    monkeypatch.setattr(cli,'freeze_fingerprint',lambda *args: {'fixture':'stable'})
    monkeypatch.setattr(cli,'load_development_run',lambda *args: (['A','B'],[{'text':'train'}],[]))
    rows=[dict(example_id='clinc150:val:0',text='cal',label='A',dataset='clinc150',split='calibration',is_oos=False),
          dict(example_id='clinc150:oos_test:0',text='oos',label='oos',dataset='clinc150',split='test',is_oos=True)]
    monkeypatch.setattr(cli,'load_final_rows',lambda *args: (['A','B'],rows))
    monkeypatch.setattr(cli,'predict_logits',lambda *args: np.array([[2.,1.],[0.,0.]]))
    def fake_training(command,cwd,check):
        out=tmp_path/'reports/pilots/modernbert/clinc150__b8__s13__runner-final-fixture'
        (out/'checkpoint').mkdir(parents=True)
        (out/'checkpoint/weights').write_bytes(b'fixture')
        (out/'result.json').write_text(json.dumps(dict(train_seconds=1)))
    monkeypatch.setattr(cli.subprocess,'run',fake_training)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys,'argv',['run_encoder','--root',str(tmp_path),'--model','modernbert','--dataset','clinc150',
                                  '--budget','8','--seed','13','--phase','final','--run-tag','fixture',
                                  '--allow-test','--freeze-record',str(freeze)])
    cli.main()
    from jevcal.contract import read_jsonl, validate
    result=read_jsonl(tmp_path/'preds/modernbert/clinc150__b8__s13.jsonl')
    assert not validate(result,['A','B'],{r['example_id'] for r in rows})
    assert result[1]['is_oos'] and result[1]['logits']==[0.,0.]
    with pytest.raises(FileExistsError,match='overwrite'):
        cli.main()


def test_fresh_requires_hash_and_joins_ag_news_file(tmp_path,monkeypatch):
    (tmp_path/'manifests').mkdir()
    (tmp_path/'data/raw/ag_news').mkdir(parents=True)
    sources={}
    for split in ['train','test']:
        name=f'ag_news/{split}.csv'
        (tmp_path/'data/raw'/name).write_bytes(b'fixture')
        sources[name]={'sha256':hashlib.sha256(b'fixture').hexdigest()}
    (tmp_path/'data/raw/SOURCES.json').write_text(json.dumps(sources))
    (tmp_path/'manifests/ag_news_calibration_ids.txt').write_text('ag_news:train:0\n')
    (tmp_path/'manifests/ag_news_test_ids.txt').write_text('ag_news:test:0\n')
    monkeypatch.setattr(E,'load_development_run',lambda *args: (['A','B'],[dict(text='training only')],[]))
    monkeypatch.setattr(E,'load_split',lambda dataset,split,root: [dict(example_id=f'ag_news:{split}:0',text=split,label='A')])
    freeze={}
    monkeypatch.setattr(E,'require_freeze',lambda *args: freeze)
    with pytest.raises(ValueError,match='adjudicated'):
        E.load_final_rows(tmp_path,'ag_news',8,13,None,True,True)
    (tmp_path/'data/fresh').mkdir()
    path=tmp_path/'data/fresh/ag_fresh_2026.csv'
    path.write_text('example_id,text,label\nag_fresh:test:0,new headline,A\n')
    freeze['ag_fresh_sha256']=E.digest(path)
    _,rows=E.load_final_rows(tmp_path,'ag_news',8,13,None,True,True)
    assert len(rows)==3 and rows[-1]['dataset']=='ag_fresh'
