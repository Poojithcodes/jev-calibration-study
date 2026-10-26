"""Training-time estimates from measured updates and the existing run grid."""
import math
from .training_schedule import training_schedule


def forecast(runs, clinc_seconds, ag_seconds, setfit_seconds, modernbert_min_updates=0,
             setfit_clinc_seconds=None, setfit_banking_seconds=None):
    estimates = []
    for run in runs:
        n = run['training_examples']
        if run['model'] == 'modernbert':
            _, steps = training_schedule(n, 16, 10, modernbert_min_updates)
            seconds = ag_seconds if run['dataset'] == 'ag_news' else clinc_seconds
            basis = 'AG News bf16 pilot' if run['dataset'] == 'ag_news' else (
                'CLINC150 b64 benchmark' if run['dataset'] == 'clinc150' else 'CLINC proxy; Banking77 not timed')
        else:
            steps = math.ceil(n * 40 / 16)
            seconds = setfit_seconds
            basis = 'MPNet AG News proxy; other datasets not timed'
            if run['dataset']=='clinc150' and setfit_clinc_seconds is not None:
                seconds=setfit_clinc_seconds
                basis='MPNet CLINC150 b8 measured updates; larger budgets extrapolated'
            elif run['dataset']=='banking77' and setfit_banking_seconds is not None:
                seconds=setfit_banking_seconds
                basis='MPNet Banking77 b8 measured updates; b16 extrapolated'
        estimates.append(dict(**run, estimated_optimizer_steps=steps,
                              seconds_per_update_assumption=seconds, timing_basis=basis,
                              estimated_training_hours=steps * seconds / 3600))
    totals = {model: sum(r['estimated_training_hours'] for r in estimates if r['model']==model)
              for model in ['setfit','modernbert']}
    hours = sum(totals.values())
    return dict(status='provisional_training_only_forecast', runs=estimates,
                hours_by_model=totals, total_training_hours=hours,
                hours_with_20_percent_planning_buffer=hours*1.2,
                days_at_daily_gpu_hours={str(h):hours*1.2/h for h in [8,16,24]},
                assumptions=[f'ModernBERT 10 epochs or minimum {modernbert_min_updates} updates, rounded to whole epochs; effective batch 16',
                             'SetFit 1 epoch, 20 pair iterations, effective batch 16',
                             'All 60 final runs remain; pilots are not deducted',
                             '20 percent buffer is a planning assumption, not a confidence interval',
                             'Model loading, pair preparation, classifier fitting, prediction, calibration, checkpoint I/O and reruns excluded from update-based estimates',
                             'ModernBERT Banking77 uses a CLINC proxy; SetFit uses dataset benchmarks only where provided, otherwise the AG proxy'])
