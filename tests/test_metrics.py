import numpy as np
import pytest

from jevcal import metrics as m


def test_perfect_confident_predictions():
    P = np.eye(3)[[0, 1, 2, 0]]
    y = np.array([0, 1, 2, 0])
    assert m.accuracy(P, y) == 1.0
    assert m.ece(P, y) == pytest.approx(0.0)
    assert m.brier(P, y) == pytest.approx(0.0)
    assert m.aurc(P, y) == pytest.approx(0.0)
    assert m.coverage_at_risk(P, y, 0.01) == 1.0


def test_ece_hand_computed():
    # 4 examples at confidence 0.8, 3 correct -> |0.75-0.8| = 0.05
    P = np.array([[0.8, 0.2]] * 4)
    y = np.array([0, 0, 0, 1])
    assert m.ece(P, y, n_bins=15) == pytest.approx(0.05)
    assert m.ece(P, y, n_bins=2, scheme="mass") == pytest.approx(0.05)


def test_brier_and_nll_hand_computed():
    P = np.array([[0.7, 0.3], [0.4, 0.6]])
    y = np.array([0, 0])
    # (0.3^2+0.3^2 + 0.6^2+0.6^2)/2 = (0.18+0.72)/2
    assert m.brier(P, y) == pytest.approx(0.45)
    assert m.nll(P, y) == pytest.approx(-(np.log(0.7) + np.log(0.4)) / 2, rel=1e-2)


def test_nll_floor_handles_exact_zero():
    P = np.array([[1.0, 0.0]])
    y = np.array([1])
    assert np.isfinite(m.nll(P, y, eps=1e-3))


def test_risk_coverage_orders_by_confidence():
    P = np.array([[0.9, 0.1], [0.8, 0.2], [0.6, 0.4], [0.55, 0.45]])
    y = np.array([0, 0, 1, 1])  # two most confident right, two least confident wrong
    cov, risk = m.risk_coverage(P, y)
    assert list(cov) == [0.25, 0.5, 0.75, 1.0]
    assert list(risk) == pytest.approx([0, 0, 1 / 3, 0.5])
    assert m.coverage_at_risk(P, y, 0.05) == 0.5


def test_oos_metrics():
    P = np.array([[0.95, 0.05], [0.9, 0.1], [0.5, 0.5], [0.55, 0.45]])
    is_oos = [0, 0, 1, 1]
    assert m.oos_auroc(P, is_oos) == 1.0
    assert m.fpr_at_95_tpr(P, is_oos) == 0.0


def test_temperature_recovers_known_value():
    rng = np.random.default_rng(0)
    true_logits = rng.normal(size=(5000, 4)) * 2
    p_true = m.apply_temperature(true_logits, 1.0)
    y = np.array([rng.choice(4, p=p) for p in p_true])
    t = m.fit_temperature(true_logits * 3, y)  # model 3x overconfident
    assert t == pytest.approx(3.0, rel=0.1)


def test_paired_bootstrap_zero_for_identical_systems():
    P = np.array([[0.7, 0.3], [0.2, 0.8], [0.6, 0.4]])
    y = np.array([0, 1, 1])
    point, lo, hi = m.paired_bootstrap(m.accuracy, P, P, y, n_boot=200)
    assert point == lo == hi == 0.0
