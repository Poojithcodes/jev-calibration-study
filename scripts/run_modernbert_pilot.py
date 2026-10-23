"""Train ModernBERT on shared IDs; evaluate development only, never official test."""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--dataset', choices=['ag_news', 'banking77', 'clinc150'], default='ag_news')
    p.add_argument('--budget', type=int, default=8)
    p.add_argument('--seed', type=int, default=13)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/modernbert_pilot.json')
    p.add_argument('--run-tag', default='', help='Separate artifact directory for a numerical-validation repeat')
    a = p.parse_args()
    root = a.root.resolve()
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'
    import numpy as np
    import torch
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
    from jevcal.encoder_data import load_development_run, shared_hashes
    from jevcal.contract import validate, write_jsonl

    before = shared_hashes(root)
    order, train, dev = load_development_run(root, a.dataset, a.budget, a.seed)
    if a.run_tag and not all(c.isalnum() or c in '-_' for c in a.run_tag):
        p.error('--run-tag must contain only letters, digits, hyphens, or underscores')
    cfg = json.loads(a.config.read_text())
    cfg.update(dataset=a.dataset, budget=a.budget, seed=a.seed)
    suffix = f'__{a.run_tag}' if a.run_tag else ''
    out = root / f'reports/pilots/modernbert/{a.dataset}__b{a.budget}__s{a.seed}{suffix}'
    out.mkdir(parents=True, exist_ok=False)
    save = lambda name, value: (out / name).write_text(json.dumps(value, indent=2), encoding='utf-8')
    save('config.json', cfg)
    save('shared_hashes_before.json', before)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    set_seed(a.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    save('environment.json', dict(python=sys.version, executable=sys.executable,
         packages={k: importlib.metadata.version(k) for k in ['torch', 'transformers', 'numpy', 'scikit-learn']},
         cuda_runtime=torch.version.cuda, gpu=torch.cuda.get_device_name(0)))
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_id'], revision=cfg['revision'], local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg['model_id'], revision=cfg['revision'], local_files_only=True,
        num_labels=len(order), id2label=dict(enumerate(order)), label2id={s:i for i,s in enumerate(order)},
        attn_implementation=cfg['attention'], reference_compile=False).to('cuda')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    train_tokens = tokenizer([r['text'] for r in train], truncation=True, max_length=cfg['max_length'])

    def batch(indices):
        inputs = tokenizer.pad([{k: v[i] for k,v in train_tokens.items()} for i in indices],
                               padding=True, return_tensors='pt')
        return {k:v.to('cuda') for k,v in inputs.items()}, torch.tensor([order.index(train[i]['label']) for i in indices], device='cuda')

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        (no_decay if param.ndim < 2 or name.endswith('bias') else decay).append(param)
    optimizer = torch.optim.AdamW([{'params':decay, 'weight_decay':cfg['weight_decay']},
                                  {'params':no_decay, 'weight_decay':0.0}], lr=cfg['learning_rate'])
    effective_batch = cfg['micro_batch_size'] * cfg['gradient_accumulation_steps']
    total_steps = math.ceil(len(train) / effective_batch) * cfg['epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer, math.ceil(total_steps * cfg['warmup_ratio']), total_steps)
    scaler = torch.amp.GradScaler('cuda', init_scale=cfg.get('initial_loss_scale', 65536.0))
    history = []
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    print(f'Training {len(train)} examples for {cfg["epochs"]} epochs / {total_steps} optimizer steps', flush=True)
    rng = np.random.default_rng(a.seed)
    skipped = 0
    for epoch in range(cfg['epochs']):
        model.train()
        shuffled = rng.permutation(len(train)).tolist()
        total_loss = 0.0
        for offset in range(0, len(train), effective_batch):
            group = shuffled[offset:offset + effective_batch]
            optimizer.zero_grad(set_to_none=True)
            for j in range(0, len(group), cfg['micro_batch_size']):
                indices = group[j:j + cfg['micro_batch_size']]
                inputs, labels = batch(indices)
                with torch.autocast('cuda', dtype=torch.float16):
                    loss = model(**inputs, labels=labels).loss
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite training loss')
                total_loss += float(loss.detach()) * len(indices)
                scaler.scale(loss * (len(indices) / len(group))).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['max_grad_norm'])
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < old_scale:
                skipped += 1
            else:
                scheduler.step()
        item = dict(epoch=epoch+1, training_loss=total_loss/len(train), learning_rate=scheduler.get_last_lr()[0])
        history.append(item)
        print(json.dumps(item), flush=True)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated()
    save('training_history.json', history)
    checkpoint = out / 'checkpoint'
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)

    def predict(current, rows):
        current.eval()
        logits = []
        with torch.inference_mode():
            for offset in range(0, len(rows), cfg['micro_batch_size']):
                inputs = tokenizer([r['text'] for r in rows[offset:offset+cfg['micro_batch_size']]],
                                   truncation=True, max_length=cfg['max_length'], padding=True, return_tensors='pt').to('cuda')
                # FP32 evaluation makes checkpoint round-trip comparison straightforward.
                logits.append(current(**inputs).logits.float().cpu())
        return torch.cat(logits)

    logits = predict(model, dev)
    del optimizer, scheduler, model, decay, no_decay, param, loss
    gc.collect()
    torch.cuda.empty_cache()
    restored = AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True,
                                                                 attn_implementation='sdpa', reference_compile=False).to('cuda')
    restored_logits = predict(restored, dev[:2])
    torch.testing.assert_close(restored_logits, logits[:2], rtol=1e-5, atol=1e-5)
    if not torch.isfinite(logits).all():
        raise RuntimeError('Non-finite development logits')
    probs = torch.softmax(logits.double(), dim=1).numpy()
    records = [dict(example_id=r['example_id'], dataset=a.dataset, split='dev', model='modernbert',
                    label_config=None, budget=a.budget, seed=a.seed, true_label=r['label'], is_oos=False,
                    class_order=order, class_probabilities=prob.tolist(), predicted_label=order[int(prob.argmax())],
                    logits=logit.tolist(), checkpoint=checkpoint.relative_to(root).as_posix(), train_seconds=seconds, config=cfg)
               for r, prob, logit in zip(dev, probs, logits)]
    problems = validate(records, order, {r['example_id'] for r in dev})
    if problems:
        raise ValueError(problems)
    write_jsonl(records, out / 'development_predictions.jsonl')
    after = shared_hashes(root)
    if before != after:
        raise RuntimeError('Shared artifacts changed')
    save('shared_hashes_after.json', after)
    gold, pred = [[r[k] for r in records] for k in ['true_label', 'predicted_label']]
    result = dict(status='completed_development_only', train_examples=len(train), dev_examples=len(dev),
                  development_accuracy=accuracy_score(gold,pred),
                  development_macro_f1=f1_score(gold,pred,labels=order,average='macro',zero_division=0),
                  confusion_matrix=confusion_matrix(gold,pred,labels=order).tolist(), class_order=order,
                  train_seconds=seconds, peak_cuda_allocated_bytes=peak, skipped_optimizer_steps=skipped,
                  checkpoint_reload_passed=True, validate_problems=problems, shared_hashes_unchanged=True,
                  official_test_evaluated=False, calibration_evaluated=False)
    save('result.json', result)
    with (root / 'docs/run_ledger.csv').open('a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow([datetime.now(timezone(timedelta(hours=5, minutes=30))).date(), 'Poojith', a.dataset, 'modernbert', '',
                              a.budget,a.seed,(out/'development_predictions.jsonl').relative_to(root).as_posix(),
                              checkpoint.relative_to(root).as_posix(),len(dev),seconds,0,True,
                              f'Development-only pilot; checkpoint reload verified; no final evaluation. Variant: {a.run_tag or "initial"}.'])
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
