"""One manifest-backed entry point for development and frozen final encoder runs."""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from jevcal.encoder_data import load_development_run, shared_hashes
from jevcal.encoder_runner import (checkpoint_hashes, digest, freeze_fingerprint,
                                    load_final_rows, require_freeze, require_grid_run)


def predict_logits(model_name, checkpoint, rows, order, max_length, batch_size):
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    if model_name == 'modernbert':
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, local_files_only=True, attn_implementation='sdpa', reference_compile=False).to('cuda')
        if [model.config.id2label[i] for i in range(len(order))] != order:
            raise ValueError('Checkpoint class order differs')
        model.eval()
        parts = []
        with torch.inference_mode():
            for offset in range(0,len(rows),batch_size):
                inputs = tokenizer([r['text'] for r in rows[offset:offset+batch_size]],
                                   truncation=True,max_length=max_length,padding=True,return_tensors='pt').to('cuda')
                parts.append(model(**inputs).logits.float().cpu().numpy())
        return np.concatenate(parts)
    from setfit import SetFitModel
    model = SetFitModel.from_pretrained(str(checkpoint),device='cuda')
    model.model_body.max_seq_length = max_length
    columns = [list(model.model_head.classes_).index(i) for i in range(len(order))]
    parts = []
    for offset in range(0,len(rows),batch_size):
        embeddings = model.model_body.encode([r['text'] for r in rows[offset:offset+batch_size]],
                        batch_size=batch_size,normalize_embeddings=model.normalize_embeddings,show_progress_bar=False)
        parts.append(model.model_head.decision_function(embeddings)[:,columns])
    return np.concatenate(parts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--model', choices=['setfit','modernbert'], required=True)
    p.add_argument('--dataset', choices=['ag_news','banking77','clinc150'], required=True)
    p.add_argument('--budget', type=int, required=True)
    p.add_argument('--seed', type=int, choices=[13,42,87], required=True)
    p.add_argument('--phase', choices=['development','final'], default='development')
    p.add_argument('--run-tag', required=True)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--predict-only', action='store_true', help='Reuse a completed, provenance-verified runner checkpoint')
    p.add_argument('--freeze-record', type=Path)
    p.add_argument('--allow-test', action='store_true')
    p.add_argument('--include-fresh', action='store_true')
    a = p.parse_args()
    if not a.run_tag or not all(c.isalnum() or c in '-_' for c in a.run_tag):
        p.error('Invalid run tag')
    if a.phase != 'final' and (a.allow_test or a.include_fresh or a.predict_only):
        p.error('Test, fresh and predict-only options require final mode')
    root = a.root.resolve()
    os.chdir(root)
    os.environ['HF_HUB_OFFLINE']='1'
    require_grid_run(root,a.model,a.dataset,a.budget,a.seed)
    if a.phase=='final':
        require_freeze(root,a.freeze_record,a.allow_test)
    order, train, dev = load_development_run(root,a.dataset,a.budget,a.seed)
    config_path = root/f'configs/{a.model}_selected.json'
    cfg = json.loads(config_path.read_text())
    # SetFit pilot fixes these settings; fail loudly if the selected configuration drifts.
    if a.model=='setfit':
        fixed = dict(embedding_epochs=1,num_iterations=20,body_learning_rate=2e-5,max_length=256,
                     warmup_proportion=0.1,head_C=1.0,head_max_iter=1000,normalize_embeddings=False,
                     use_amp=False,gradient_checkpointing=False)
        if any(cfg[k]!=v for k,v in fixed.items()):
            raise ValueError('SetFit selected settings differ from implemented pilot settings')
    tag = f'runner-{a.phase}-{a.run_tag}'
    run_id = f'{a.dataset}__b{a.budget}__s{a.seed}'
    out = root/f'reports/pilots/{a.model}/{run_id}__{tag}'
    predictions = root/f'preds/{a.model}/{run_id}.jsonl'
    provenance = dict(model=a.model,dataset=a.dataset,budget=a.budget,seed=a.seed,phase=a.phase,
                      config_sha256=digest(config_path),fingerprint=freeze_fingerprint(root))
    if a.dry_run:
        print(json.dumps(dict(**provenance,train_examples=len(train),dev_examples=len(dev),
                             output=str(out),final_prediction_file=str(predictions) if a.phase=='final' else None),indent=2))
        return
    if a.phase=='final' and predictions.exists():
        raise FileExistsError(f'Refusing to overwrite {predictions}')
    if a.predict_only:
        prior = json.loads((out/'runner_provenance.json').read_text())
        if prior['run'] != provenance or prior['checkpoint_hashes'] != checkpoint_hashes(out/'checkpoint'):
            raise ValueError('Checkpoint provenance does not match this run')
    else:
        if out.exists():
            raise FileExistsError(f'Refusing to overwrite {out}; use a new tag')
        command = [sys.executable,'-u',str(root/f'scripts/run_{a.model}_pilot.py'),
                   '--root',str(root),'--dataset',a.dataset,'--budget',str(a.budget),'--seed',str(a.seed),'--run-tag',tag]
        if a.model=='modernbert':
            command += ['--config',str(config_path)]
        else:
            command += ['--model-id',cfg['model_id'],'--revision',cfg['revision'],
                        '--micro-batch-size',str(cfg['micro_batch_size']),
                        '--gradient-accumulation-steps',str(cfg['gradient_accumulation_steps'])]
        subprocess.run(command,cwd=root,check=True)
        if provenance['fingerprint'] != freeze_fingerprint(root):
            raise ValueError('Inputs changed during training')
        (out/'runner_provenance.json').write_text(json.dumps(dict(run=provenance,
                       checkpoint_hashes=checkpoint_hashes(out/'checkpoint')),indent=2),encoding='utf-8')
    if a.phase=='development':
        print(f'Development run complete: {out}')
        return
    # The backend's dev diagnostic is logged but never selects epochs/checkpoints.
    # Only now, after training and a second freeze check, load final rows.
    order, rows = load_final_rows(root,a.dataset,a.budget,a.seed,a.freeze_record,a.allow_test,a.include_fresh)
    from scipy.special import softmax
    import numpy as np
    from jevcal.contract import validate, write_jsonl
    logits = predict_logits(a.model,out/'checkpoint',rows,order,cfg['max_length'],cfg['micro_batch_size'])
    if logits.shape != (len(rows),len(order)) or not np.isfinite(logits).all():
        raise ValueError('Invalid final logits')
    probs = softmax(logits.astype('float64'),axis=1)
    result = json.loads((out/'result.json').read_text())
    records = [dict(example_id=r['example_id'],dataset=r['dataset'],split=r['split'],model=a.model,
                    label_config=None,budget=a.budget,seed=a.seed,true_label=r['label'],is_oos=r['is_oos'],
                    class_order=order,class_probabilities=prob.tolist(),logits=logit.tolist(),
                    predicted_label=order[int(prob.argmax())],checkpoint=(out/'checkpoint').relative_to(root).as_posix(),
                    train_seconds=result['train_seconds'],config=cfg)
               for r,logit,prob in zip(rows,logits,probs)]
    errors = validate(records,order,{r['example_id'] for r in rows})
    if errors:
        raise ValueError(errors)
    require_freeze(root,a.freeze_record,a.allow_test)
    predictions.parent.mkdir(parents=True,exist_ok=True)
    temporary = predictions.with_suffix('.jsonl.tmp')
    if temporary.exists():
        raise FileExistsError(temporary)
    write_jsonl(records,temporary)
    temporary.rename(predictions)
    (out/'final_result.json').write_text(json.dumps(dict(status='completed_final_predictions',
        prediction_file=predictions.relative_to(root).as_posix(),rows=len(records),
        prediction_sha256=digest(predictions),freeze_record_sha256=digest(a.freeze_record),
        include_fresh=a.include_fresh,shared_hashes_unchanged=True),indent=2),encoding='utf-8')
    with (root/'docs/run_ledger.csv').open('a',newline='',encoding='utf-8') as f:
        csv.writer(f).writerow([datetime.now(timezone(timedelta(hours=5, minutes=30))).date(),'Poojith',a.dataset,a.model,'',
            a.budget,a.seed,predictions.relative_to(root).as_posix(),(out/'checkpoint').relative_to(root).as_posix(),
            len(records),result['train_seconds'],0,True,'Frozen final prediction generation; no test-based selection'])
    print(f'Final predictions validated: {predictions}')


if __name__=='__main__':
    main()
