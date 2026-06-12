"""Statistical primitives used by every gate: AUROC, bootstrap CIs, agreement.

All randomness flows through an explicit ``numpy.random.Generator`` so audits are
reproducible from (bundle, config, seed) alone.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as _scipy_stats
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class CI:
    """A point estimate with a two-sided bootstrap confidence interval."""

    point: float
    lo: float
    hi: float
    level: float = 0.95

    def excludes(self, value: float) -> bool:
        return self.lo > value or self.hi < value

    def to_dict(self) -> dict:
        return {"point": self.point, "lo": self.lo, "hi": self.hi, "level": self.level}


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """AUROC; 0.5 when only one class is present (no information either way)."""
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return 0.5
    return float(roc_auc_score(labels, scores))


def bootstrap_auroc(
    labels: np.ndarray,
    scores: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = 2000,
    level: float = 0.95,
) -> CI:
    """Bootstrap CI for AUROC, resampling examples with replacement."""
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    point = auroc(labels, scores)
    n = len(labels)
    reps = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        reps[b] = auroc(labels[idx], scores[idx])
    lo, hi = _percentile_ci(reps, level)
    return CI(point, lo, hi, level)


def bootstrap_auroc_diff(
    labels: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = 2000,
    level: float = 0.95,
) -> CI:
    """Bootstrap CI for AUROC(a) - AUROC(b) on the same examples (paired resampling)."""
    labels = np.asarray(labels)
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)
    point = auroc(labels, scores_a) - auroc(labels, scores_b)
    n = len(labels)
    reps = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        reps[b] = auroc(labels[idx], scores_a[idx]) - auroc(labels[idx], scores_b[idx])
    lo, hi = _percentile_ci(reps, level)
    return CI(point, lo, hi, level)


def bootstrap_mean(
    values: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = 2000,
    level: float = 0.95,
) -> CI:
    """Bootstrap CI for the mean of a sample."""
    values = np.asarray(values, dtype=float)
    point = float(values.mean())
    n = len(values)
    idx = rng.integers(0, n, (n_boot, n))
    reps = values[idx].mean(axis=1)
    lo, hi = _percentile_ci(reps, level)
    return CI(point, lo, hi, level)


def bootstrap_abs_mean_diff(
    values_a: np.ndarray,
    values_b: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = 2000,
    level: float = 0.95,
) -> CI:
    """Bootstrap CI for |mean(a)| - |mean(b)| with independent resampling per group.

    Used to test whether the probe arm's behavioral effect exceeds a control
    arm's, without assuming the control moves behavior in any particular
    direction.
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    point = abs(a.mean()) - abs(b.mean())
    reps = np.empty(n_boot)
    for i in range(n_boot):
        ra = a[rng.integers(0, len(a), len(a))]
        rb = b[rng.integers(0, len(b), len(b))]
        reps[i] = abs(ra.mean()) - abs(rb.mean())
    lo, hi = _percentile_ci(reps, level)
    return CI(point, lo, hi, level)


def dose_response(alphas: np.ndarray, effects: np.ndarray) -> tuple[float, float]:
    """Spearman rho and p-value of behavioral effect vs steering strength alpha.

    A causally load-bearing direction should produce a monotone trend across the
    alpha grid, not an isolated blip at one alpha.
    """
    alphas = np.asarray(alphas, dtype=float)
    effects = np.asarray(effects, dtype=float)
    if len(np.unique(alphas)) < 3:
        return 0.0, 1.0
    rho, p = _scipy_stats.spearmanr(alphas, effects)
    if np.isnan(rho):
        return 0.0, 1.0
    return float(rho), float(p)


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa for two binary raters; 1.0 if both raters are constant and equal."""
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    po = float(np.mean(a == b))
    pa = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    if pa >= 1.0:
        return 1.0 if po == 1.0 else 0.0
    return float((po - pa) / (1 - pa))


def _percentile_ci(reps: np.ndarray, level: float) -> tuple[float, float]:
    tail = (1.0 - level) / 2.0
    return float(np.quantile(reps, tail)), float(np.quantile(reps, 1.0 - tail))
