"""Development-only SetFit pilot; never loads official test examples."""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
from collections import Counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--dataset', choices=['ag_news', 'banking77', 'clinc150'], default='ag_news')
    parser.add_argument('--budget', type=int, default=8)
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--model-id', default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--revision')
    parser.add_argument('--run-tag', default='')
    parser.add_argument('--gradient-checkpointing', action='store_true')
    parser.add_argument('--micro-batch-size', type=int, default=16)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
    args = parser.parse_args()
    if args.micro_batch_size < 1 or args.gradient_accumulation_steps < 1:
        parser.error('Batch size and accumulation steps must be positive')
    if args.run_tag and not all(c.isalnum() or c in '-_' for c in args.run_tag):
        parser.error('Invalid run tag')
    root = args.root.resolve()
    os.chdir(root)
    sys.path.insert(0, str(root / 'src'))
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    if args.revision:
        os.environ['HF_HUB_OFFLINE'] = '1'
    import numpy as np
    import torch
    from datasets import Dataset
    from huggingface_hub import HfApi, snapshot_download
    from scipy.special import softmax
    from setfit import SetFitModel, Trainer, TrainingArguments
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import set_seed
    from jevcal.contract import validate, write_jsonl
    from jevcal.data import load_split
    from jevcal.encoder_data import load_development_run

    def hashes():
        paths = list((root / 'manifests').glob('*')) + [root / 'reports/split_audit.json', root / 'data/raw/SOURCES.json']
        return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}

    before = hashes()
    suffix = f'__{args.run_tag}' if args.run_tag else ''
    out = root / f'reports/pilots/setfit/{args.dataset}__b{args.budget}__s{args.seed}{suffix}'
    out.mkdir(parents=True, exist_ok=False)
    (out / 'shared_hashes_before.json').write_text(json.dumps(before, indent=2))
    assert torch.cuda.is_available(), 'CUDA computation required for this pilot'
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    a = torch.randn(128, 128, device='cuda', requires_grad=True)
    a.square().mean().backward()
    torch.cuda.synchronize()
    assert torch.isfinite(a.grad).all()
    environment = {
        'python': sys.version, 'executable': sys.executable,
        'packages': {p: importlib.metadata.version(p) for p in ['torch', 'setfit', 'sentence-transformers', 'transformers', 'datasets', 'scikit-learn', 'huggingface-hub']},
        'cuda_runtime': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
        'cuda_forward_backward_passed': True,
    }
    (out / 'environment.json').write_text(json.dumps(environment, indent=2))
    print(json.dumps(environment), flush=True)
    order, train, dev = load_development_run(root, args.dataset, args.budget, args.seed)
    dev_ids = [r['example_id'] for r in dev]
    model_id = args.model_id
    revision = args.revision or HfApi().model_info(model_id).sha
    config = {
        'model_id': model_id, 'revision': revision, 'dataset': args.dataset, 'budget': args.budget, 'seed': args.seed,
        'embedding_epochs': 1, 'batch_size': args.micro_batch_size,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'effective_batch_size': args.micro_batch_size * args.gradient_accumulation_steps, 'num_iterations': 20,
        'body_learning_rate': 2e-5, 'max_length': 256, 'warmup_proportion': 0.1,
        'loss': 'CosineSimilarityLoss', 'head': 'LogisticRegression',
        'head_params': {'C': 1.0, 'max_iter': 1000, 'random_state': args.seed},
        'normalize_embeddings': False, 'use_amp': False,
        'gradient_checkpointing': args.gradient_checkpointing,
        'evaluation': 'development only; no early stopping or hyperparameter search',
        'status': 'pilot settings; not a frozen full-grid protocol',
    }
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    # SetFit 1.2 does not forward revision to SentenceTransformer. Resolve the
    # exact snapshot first so the backbone and recorded revision cannot diverge.
    snapshot = snapshot_download(model_id, revision=revision, local_files_only=bool(args.revision))
    model = SetFitModel.from_pretrained(snapshot, device='cuda',
                                        head_params=config['head_params'], normalize_embeddings=False)
    model.model_body.max_seq_length = 256
    if args.gradient_checkpointing:
        model.model_body[0].auto_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    dataset = Dataset.from_dict({'text': [r['text'] for r in train], 'label': [order.index(r['label']) for r in train]})
    training_args = TrainingArguments(
        output_dir=str(out / 'training'), batch_size=args.micro_batch_size, num_epochs=1,
        num_iterations=20, body_learning_rate=2e-5, max_length=256,
        warmup_proportion=0.1, seed=args.seed, use_amp=False, report_to='none',
        eval_strategy='no', save_strategy='no', logging_steps=10, show_progress_bar=False,
    )
    (out / 'training_args.json').write_text(training_args.to_json_string())
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    trainer.st_trainer.args.gradient_accumulation_steps = args.gradient_accumulation_steps
    (out / 'transformers_training_args.json').write_text(trainer.st_trainer.args.to_json_string())
    assert model.model_body.device.type == 'cuda'
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    trainer.train()
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - start
    optimizer_steps = trainer.st_trainer.state.global_step
    (out / 'training_history.json').write_text(json.dumps(trainer.st_trainer.state.log_history, indent=2))
    checkpoint = out / 'checkpoint'
    model.save_pretrained(str(checkpoint))
    embeddings = model.model_body.encode([r['text'] for r in dev], batch_size=16,
                                         normalize_embeddings=model.normalize_embeddings, show_progress_bar=False)
    columns = [list(model.model_head.classes_).index(i) for i in range(len(order))]
    logits = model.model_head.decision_function(embeddings)[:, columns]
    probs = softmax(logits, axis=1)
    np.testing.assert_allclose(probs, model.model_head.predict_proba(embeddings)[:, columns], atol=1e-6)
    peak_memory = torch.cuda.max_memory_allocated()
    model.model_body.to('cpu')
    del trainer, model, embeddings
    gc.collect()
    torch.cuda.empty_cache()
    restored = SetFitModel.from_pretrained(str(checkpoint), device='cuda')
    restored_embeddings = restored.model_body.encode([r['text'] for r in dev[:2]], batch_size=16,
                                                   normalize_embeddings=restored.normalize_embeddings, show_progress_bar=False)
    restored_logits = restored.model_head.decision_function(restored_embeddings)[:, columns]
    np.testing.assert_allclose(restored_logits, logits[:2], rtol=1e-4, atol=1e-4)
    records = []
    for r, logit, prob in zip(dev, logits, probs):
        records.append(dict(example_id=r['example_id'], dataset=args.dataset, split='dev', model='setfit',
                            label_config=None, budget=args.budget, seed=args.seed, true_label=r['label'], is_oos=False,
                            class_order=order, class_probabilities=prob.tolist(),
                            predicted_label=order[int(np.argmax(prob))], logits=logit.tolist(),
                            checkpoint=str(checkpoint.relative_to(root)), train_seconds=train_seconds, config=config))
    problems = validate(records, order, set(dev_ids))
    assert not problems, problems
    prediction_path = out / 'development_predictions.jsonl'
    write_jsonl(records, prediction_path)
    after = hashes()
    assert before == after, 'Shared manifests or source/audit metadata changed'
    (out / 'shared_hashes_after.json').write_text(json.dumps(after, indent=2))
    result = {
        'status': 'completed_development_only', 'train_examples': len(train), 'dev_examples': len(dev),
        'development_accuracy': accuracy_score([r['true_label'] for r in records], [r['predicted_label'] for r in records]),
        'development_macro_f1': f1_score([r['true_label'] for r in records], [r['predicted_label'] for r in records], average='macro'),
        'train_seconds': train_seconds, 'peak_cuda_allocated_bytes': peak_memory,
        'checkpoint_reload_passed': True,
        'optimizer_steps_completed': optimizer_steps,
        'validate_problems': problems, 'shared_hashes_unchanged': before == after,
        'official_test_evaluated': False, 'calibration_evaluated': False,
    }
    (out / 'result.json').write_text(json.dumps(result, indent=2))
    with (root / 'docs/run_ledger.csv').open('a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow([datetime.now(timezone(timedelta(hours=5, minutes=30))).date(), 'Poojith', args.dataset, 'setfit', '', args.budget, args.seed,
                               str(prediction_path.relative_to(root)), str(checkpoint.relative_to(root)),
                               len(records), train_seconds, 0, True,
                               f'Development-only pilot; not final evaluation. Variant: {args.run_tag or "initial"}.'])
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
