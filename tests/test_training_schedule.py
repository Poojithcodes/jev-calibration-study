import pytest
from jevcal.training_schedule import training_schedule


def test_small_pool_gets_minimum_and_large_pool_keeps_ten_epochs():
    assert training_schedule(32, 16, 10, 50) == (25, 50)
    assert training_schedule(32, 16, 10, 100) == (50, 100)
    assert training_schedule(9600, 16, 10, 100) == (10, 6000)
    assert training_schedule(64, 16, 10, 50) == (13, 52)


def test_partial_batch_counts_as_update_and_empty_pool_rejected():
    assert training_schedule(17, 16, 10, 0) == (10, 20)
    with pytest.raises(ValueError):
        training_schedule(0, 16)
