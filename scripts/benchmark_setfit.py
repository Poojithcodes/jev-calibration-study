"""Time fixed-iteration MPNet updates on training data only; no evaluation/head fit."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['clinc150','banking77'],default='clinc150')
    p.add_argument('--budget',type=int,default=8)
    p.add_argument('--steps',type=int,default=200)
    p.add_argument('--warmup',type=int,default=20)
    p.add_argument('--run-tag',default='timing200')
    a=p.parse_args()
    if a.steps<2 or a.warmup<0 or not a.run_tag or not all(c.isalnum() or c in '-_' for c in a.run_tag):
        p.error('Invalid timing counts or tag')
    os.chdir(ROOT)
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TOKENIZERS_PARALLELISM']='false'
    import numpy as np
    import torch
    from datasets import Dataset
    from huggingface_hub import snapshot_download
    from setfit import SetFitModel, Trainer, TrainingArguments
    from transformers import TrainerCallback,set_seed
    from jevcal.encoder_data import load_development_run,shared_hashes
    from jevcal.encoder_runner import require_grid_run
    from jevcal.setfit_sampling import bounded_trainer_class
    require_grid_run(ROOT,'setfit',a.dataset,a.budget,13)
    before=shared_hashes(ROOT)
    order,train,_=load_development_run(ROOT,a.dataset,a.budget,13)
    cfg=json.loads((ROOT/'configs/setfit_selected.json').read_text())
    total_steps=len(train)*40//16
    if a.steps+a.warmup>total_steps:
        p.error('Timing count exceeds the full training schedule')
    out=ROOT/f'reports/benchmarks/setfit/{a.dataset}__b{a.budget}__s13__{a.run_tag}'
    out.mkdir(parents=True,exist_ok=False)
    save=lambda name,obj:(out/name).write_text(json.dumps(obj,indent=2),encoding='utf-8')
    save('shared_hashes_before.json',before)
    save('plan.json',dict(dataset=a.dataset,budget=a.budget,seed=13,train_examples=len(train),
                         measured_updates=a.steps,warmup_updates=a.warmup,full_schedule_updates=total_steps,
                         config=cfg,head_fitted=False,evaluation=False))
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    set_seed(13)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    # Match the pilot's pre-model CUDA verification RNG consumption.
    probe=torch.randn(128,128,device='cuda',requires_grad=True)
    probe.square().mean().backward()
    snapshot=snapshot_download(cfg['model_id'],revision=cfg['revision'],local_files_only=True)
    model=SetFitModel.from_pretrained(snapshot,device='cuda',head_params={'C':1.,'max_iter':1000,'random_state':13},normalize_embeddings=False)
    model.model_body.max_seq_length=cfg['max_length']
    save('environment.json',dict(python=sys.version,gpu=torch.cuda.get_device_name(0),
          cuda=torch.version.cuda,packages={k:importlib.metadata.version(k) for k in ['torch','setfit','sentence-transformers','transformers']}))
    args=TrainingArguments(output_dir=str(out/'training'),batch_size=cfg['micro_batch_size'],num_epochs=1,
          num_iterations=20,body_learning_rate=2e-5,max_length=256,warmup_proportion=.1,seed=13,
          use_amp=False,report_to='none',eval_strategy='no',save_strategy='no',logging_strategy='no',show_progress_bar=False)
    dataset=Dataset.from_dict({'text':[r['text'] for r in train],'label':[order.index(r['label']) for r in train]})
    trainer=bounded_trainer_class(Trainer)(model=model,args=args,train_dataset=dataset)
    trainer.st_trainer.args.gradient_accumulation_steps=8
    trainer.st_trainer.args.disable_tqdm=True
    timings=[]
    class Timing(TrainerCallback):
        def on_step_begin(self,args,state,control,**kwargs):
            torch.cuda.synchronize()
            self.started=time.perf_counter()
        def on_step_end(self,args,state,control,**kwargs):
            torch.cuda.synchronize()
            timings.append(time.perf_counter()-self.started)
            if len(timings)%10==0:
                save('progress.json',dict(completed_updates=len(timings),target_updates=a.steps+a.warmup,latest_seconds=timings[-1]))
                print(f'Update {len(timings)}/{a.steps+a.warmup}: {timings[-1]:.3f}s',flush=True)
            if len(timings)>=a.steps+a.warmup:
                control.should_training_stop=True
            return control
    trainer.st_trainer.add_callback(Timing())
    torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    trainer.train_embeddings([r['text'] for r in train],[order.index(r['label']) for r in train])
    torch.cuda.synchronize()
    elapsed=time.perf_counter()-start
    measured=np.asarray(timings[a.warmup:])
    if len(measured)!=a.steps or trainer.st_trainer.state.max_steps!=total_steps:
        raise RuntimeError('Timing count or full training schedule changed')
    after=shared_hashes(ROOT)
    if before!=after:
        raise RuntimeError('Shared files changed')
    save('shared_hashes_after.json',after)
    result=dict(status='completed_training_only_benchmark',completed_updates=len(timings),measured_updates=len(measured),
                warmup_updates=a.warmup,full_schedule_updates=total_steps,training_pool_examples=len(train),
                mean_update_seconds=float(measured.mean()),median_update_seconds=float(np.median(measured)),
                p90_update_seconds=float(np.percentile(measured,90)),
                first_half_mean_seconds=float(measured[:len(measured)//2].mean()),
                second_half_mean_seconds=float(measured[len(measured)//2:].mean()),
                total_embedding_elapsed_seconds=elapsed,
                non_update_elapsed_seconds=elapsed-sum(timings),
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),update_seconds=timings,
                shared_hashes_unchanged=True,development_evaluated=False,calibration_evaluated=False,
                official_test_evaluated=False,checkpoint_saved=False,head_fitted=False,
                timing_scope='Synchronized optimizer updates; excludes batch fetching before on_step_begin, pair construction and head fit')
    save('result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='update_seconds'},indent=2),flush=True)


if __name__=='__main__':
    main()
