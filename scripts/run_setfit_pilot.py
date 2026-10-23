"""Development-only SetFit pilot; never loads official test examples."""
import argparse
import csv
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
    args = parser.parse_args()
    root = args.root.resolve()
    os.chdir(root)
    sys.path.insert(0, str(root / 'src'))
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    import numpy as np
    import torch
    from datasets import Dataset
    from huggingface_hub import HfApi
    from scipy.special import softmax
    from setfit import SetFitModel, Trainer, TrainingArguments
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import set_seed
    from jevcal.contract import validate, write_jsonl
    from jevcal.data import load_split

    def hashes():
        paths = list((root / 'manifests').glob('*')) + [root / 'reports/split_audit.json', root / 'data/raw/SOURCES.json']
        return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}

    before = hashes()
    out = root / 'reports/pilots/setfit/ag_news__b8__s13'
    out.mkdir(parents=True, exist_ok=False)
    (out / 'shared_hashes_before.json').write_text(json.dumps(before, indent=2))
    assert torch.cuda.is_available(), 'CUDA computation required for this pilot'
    set_seed(13)
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
    order = [r['name'] for r in json.loads((root / 'configs/labels/ag_news.json').read_text())['labels']]

    def ids(name):
        values = (root / 'manifests' / name).read_text().splitlines()
        assert len(values) == len(set(values))
        return values

    train_ids = ids('ag_news_train_b8_s13.txt')
    dev_ids = ids('ag_news_dev_ids.txt')
    cal_ids = ids('ag_news_calibration_ids.txt')
    test_ids = ids('ag_news_test_ids.txt')
    assert not set(train_ids) & (set(dev_ids) | set(cal_ids) | set(test_ids))
    assert not set(dev_ids) & (set(cal_ids) | set(test_ids))
    # Only the supplied training snapshot is loaded, for both train and development.
    rows = {r['example_id']: r for r in load_split('ag_news', 'train')}
    train, dev = [[rows[i] for i in group] for group in [train_ids, dev_ids]]
    assert Counter(r['label'] for r in train) == {c: 8 for c in order}
    assert Counter(r['label'] for r in dev) == {c: 25 for c in order}
    norm = lambda r: ' '.join(r['text'].casefold().split())
    assert not {norm(r) for r in train} & {norm(r) for r in dev}
    model_id = 'sentence-transformers/all-MiniLM-L6-v2'
    revision = HfApi().model_info(model_id).sha
    config = {
        'model_id': model_id, 'revision': revision, 'budget': 8, 'seed': 13,
        'embedding_epochs': 1, 'batch_size': 16, 'num_iterations': 20,
        'body_learning_rate': 2e-5, 'max_length': 256, 'warmup_proportion': 0.1,
        'loss': 'CosineSimilarityLoss', 'head': 'LogisticRegression',
        'head_params': {'C': 1.0, 'max_iter': 1000, 'random_state': 13},
        'normalize_embeddings': False, 'use_amp': False,
        'evaluation': 'development only; no early stopping or hyperparameter search',
        'status': 'pilot settings; not a frozen full-grid protocol',
    }
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    model = SetFitModel.from_pretrained(model_id, revision=revision, device='cuda',
                                        head_params=config['head_params'], normalize_embeddings=False)
    model.model_body.max_seq_length = 256
    dataset = Dataset.from_dict({'text': [r['text'] for r in train], 'label': [order.index(r['label']) for r in train]})
    training_args = TrainingArguments(
        output_dir=str(out / 'training'), batch_size=16, num_epochs=1,
        num_iterations=20, body_learning_rate=2e-5, max_length=256,
        warmup_proportion=0.1, seed=13, use_amp=False, report_to='none',
        eval_strategy='no', save_strategy='no', logging_steps=10, show_progress_bar=False,
    )
    (out / 'training_args.json').write_text(training_args.to_json_string())
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    assert model.model_body.device.type == 'cuda'
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    trainer.train()
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - start
    checkpoint = out / 'checkpoint'
    model.save_pretrained(str(checkpoint))
    embeddings = model.model_body.encode([r['text'] for r in dev], batch_size=16,
                                         normalize_embeddings=model.normalize_embeddings, show_progress_bar=False)
    columns = [list(model.model_head.classes_).index(i) for i in range(len(order))]
    logits = model.model_head.decision_function(embeddings)[:, columns]
    probs = softmax(logits, axis=1)
    np.testing.assert_allclose(probs, model.model_head.predict_proba(embeddings)[:, columns], atol=1e-6)
    records = []
    for r, logit, prob in zip(dev, logits, probs):
        records.append(dict(example_id=r['example_id'], dataset='ag_news', split='dev', model='setfit',
                            label_config=None, budget=8, seed=13, true_label=r['label'], is_oos=False,
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
        'train_seconds': train_seconds, 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
        'validate_problems': problems, 'shared_hashes_unchanged': before == after,
        'official_test_evaluated': False, 'calibration_evaluated': False,
    }
    (out / 'result.json').write_text(json.dumps(result, indent=2))
    with (root / 'docs/run_ledger.csv').open('a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(['2026-10-23', 'Poojith', 'ag_news', 'setfit', '', 8, 13,
                               str(prediction_path.relative_to(root)), str(checkpoint.relative_to(root)),
                               len(records), train_seconds, 0, True,
                               'Development-only pilot; not final evaluation. See pilot config and result.'])
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
