"""Metrics frozen by the protocol.

Conventions (state these in the paper):
- probs: (N, K) array, rows sum to 1. y: (N,) integer class indices.
- Confidence = maximum class probability (MSP), never Jev's `confidence` field.
- ECE: 15 equal-width bins by default; equal-mass bins as a sensitivity check.
- Brier: multiclass, sum over classes of squared error, averaged over examples (range 0..2).
- NLL: probabilities floored at `eps` then renormalised; report the floor.
- Selective risk = error rate among accepted examples, accepting the most confident first.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

DEFAULT_EPS = 1e-3  # Jev rounds to 0.01, so many probabilities are exactly 0


def _check(probs, y):
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y, dtype=int)
    if probs.ndim != 2 or len(probs) != len(y):
        raise ValueError("probs must be (N, K) and match y")
    return probs, y


def accuracy(probs, y) -> float:
    probs, y = _check(probs, y)
    return float((probs.argmax(1) == y).mean())


def macro_f1(probs, y) -> float:
    probs, y = _check(probs, y)
    return float(f1_score(y, probs.argmax(1), average="macro", labels=np.arange(probs.shape[1]),
                          zero_division=0))


def ece(probs, y, n_bins: int = 15, scheme: str = "width") -> float:
    probs, y = _check(probs, y)
    conf = probs.max(1)
    correct = (probs.argmax(1) == y).astype(float)
    n = len(y)
    if scheme == "width":
        edges = np.linspace(0, 1, n_bins + 1)
        idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
        groups = [np.where(idx == b)[0] for b in range(n_bins)]
    elif scheme == "mass":
        # Quantile edges, assigned by value so tied confidences (Jev rounds to 0.01)
        # always share a bin instead of being split by row order.
        edges = np.unique(np.quantile(conf, np.linspace(0, 1, n_bins + 1)))
        idx = np.digitize(conf, edges[1:-1], right=True)
        groups = [np.where(idx == b)[0] for b in range(len(edges))]
    else:
        raise ValueError("scheme must be 'width' or 'mass'")
    return float(sum(len(g) / n * abs(correct[g].mean() - conf[g].mean()) for g in groups if len(g)))


def brier(probs, y) -> float:
    probs, y = _check(probs, y)
    onehot = np.eye(probs.shape[1])[y]
    return float(((probs - onehot) ** 2).sum(1).mean())


def nll(probs, y, eps: float = DEFAULT_EPS) -> float:
    probs, y = _check(probs, y)
    p = np.clip(probs, eps, 1.0)
    p = p / p.sum(1, keepdims=True)
    return float(-np.log(p[np.arange(len(y)), y]).mean())


def risk_coverage(probs, y):
    """Returns (coverage, risk) arrays, accepting examples in order of decreasing confidence."""
    probs, y = _check(probs, y)
    conf = probs.max(1)
    order = np.argsort(-conf, kind="stable")
    errors = (probs.argmax(1) != y)[order].astype(float)
    k = np.arange(1, len(y) + 1)
    return k / len(y), np.cumsum(errors) / k


def aurc(probs, y) -> float:
    """Area under the risk-coverage curve (lower is better)."""
    _, risk = risk_coverage(probs, y)
    return float(risk.mean())


def coverage_at_risk(probs, y, max_risk: float) -> float:
    """Largest fraction of examples that can be accepted with selective error <= max_risk."""
    cov, risk = risk_coverage(probs, y)
    ok = np.where(risk <= max_risk)[0]
    return float(cov[ok.max()]) if len(ok) else 0.0


def oos_scores(probs) -> np.ndarray:
    """OOS score = 1 - MSP; higher means more likely out-of-scope."""
    return 1.0 - np.asarray(probs, dtype=float).max(1)


def oos_auroc(probs, is_oos) -> float:
    return float(roc_auc_score(np.asarray(is_oos, dtype=int), oos_scores(probs)))


def fpr_at_95_tpr(probs, is_oos) -> float:
    fpr, tpr, _ = roc_curve(np.asarray(is_oos, dtype=int), oos_scores(probs))
    return float(fpr[np.searchsorted(tpr, 0.95, side="left")])


# ---- temperature scaling ------------------------------------------------------
def fit_temperature(logits, y, bounds=(0.05, 20.0)) -> float:
    """One positive temperature minimising calibration-set NLL.
    Encoders: pass logits. Jev: pass log(clip(probs, eps)) via probs_to_logits."""
    logits = np.asarray(logits, dtype=float)
    y = np.asarray(y, dtype=int)

    def loss(log_t):
        z = logits / np.exp(log_t)
        z = z - z.max(1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(1, keepdims=True))
        return -logp[np.arange(len(y)), y].mean()

    res = minimize_scalar(loss, bounds=(np.log(bounds[0]), np.log(bounds[1])), method="bounded")
    return float(np.exp(res.x))


def probs_to_logits(probs, eps: float = DEFAULT_EPS) -> np.ndarray:
    p = np.clip(np.asarray(probs, dtype=float), eps, 1.0)
    return np.log(p / p.sum(1, keepdims=True))


def apply_temperature(logits, t: float) -> np.ndarray:
    z = np.asarray(logits, dtype=float) / t
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


# ---- statistics ---------------------------------------------------------------
def paired_bootstrap(metric, probs_a, probs_b, y, n_boot: int = 2000, seed: int = 0):
    """CI for metric(A) - metric(B), resampling the SAME example IDs for both systems."""
    probs_a, y = _check(probs_a, y)
    probs_b, _ = _check(probs_b, y)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        diffs[b] = metric(probs_a[i], y[i]) - metric(probs_b[i], y[i])
    point = metric(probs_a, y) - metric(probs_b, y)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def summary(probs, y, eps: float = DEFAULT_EPS) -> dict:
    return {
        "n": int(len(y)),
        "accuracy": accuracy(probs, y),
        "macro_f1": macro_f1(probs, y),
        "ece_15_width": ece(probs, y, 15, "width"),
        "ece_15_mass": ece(probs, y, 15, "mass"),
        "brier": brier(probs, y),
        "nll": nll(probs, y, eps),
        "aurc": aurc(probs, y),
        "coverage_at_1pct_risk": coverage_at_risk(probs, y, 0.01),
        "coverage_at_5pct_risk": coverage_at_risk(probs, y, 0.05),
    }
