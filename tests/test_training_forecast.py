from jevcal.training_forecast import forecast
import pytest


def test_forecast_counts_optimizer_updates_and_keeps_proxy_labels():
    runs = [dict(dataset='clinc150',model='modernbert',training_examples=9600),
            dict(dataset='banking77',model='modernbert',training_examples=616),
            dict(dataset='ag_news',model='setfit',training_examples=32)]
    result = forecast(runs,1,4,2)
    assert [r['estimated_optimizer_steps'] for r in result['runs']] == [6000,390,80]
    assert result['total_training_hours'] == pytest.approx((6000+390+160)/3600)
    assert 'proxy' in result['runs'][1]['timing_basis']
    assert result['hours_with_20_percent_planning_buffer'] == result['total_training_hours']*1.2


def test_forecast_accounts_for_minimum_update_rule():
    runs = [dict(dataset='ag_news', model='modernbert', training_examples=64)]
    result = forecast(runs, 1, 4, 2, modernbert_min_updates=50)
    assert result['runs'][0]['estimated_optimizer_steps'] == 52
