"""
datasets.py
Dataset generation utilities for APMR Regression.
"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np

def make_regression(
    n_samples: int = 200,
    n_features: int = 10,
    noise: float = 0.1,
    outlier_fraction: float = 0.0,
    outlier_scale: float = 20.0,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    rng = np.random.default_rng(random_state)

    A = rng.standard_normal((n_samples, n_features))

    x_true = rng.standard_normal(n_features)

    b = A @ x_true

    b += noise * rng.standard_normal(n_samples)

    if outlier_fraction > 0:
        n_outliers = int(outlier_fraction * n_samples)
        indices = rng.choice(n_samples, size=n_outliers, replace=False)

        b[indices] += (outlier_scale* rng.standard_normal(n_outliers))

    return A, b, x_true


def train_test_split(
    A: np.ndarray,
    b: np.ndarray,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
):

    rng = np.random.default_rng(random_state)

    n = len(b)

    indices = np.arange(n)

    rng.shuffle(indices)

    n_test = int(test_size * n)

    test_idx = indices[:n_test]

    train_idx = indices[n_test:]

    return (A[train_idx], A[test_idx], b[train_idx], b[test_idx])


def add_outliers(
    b: np.ndarray,
    fraction: float = 0.2,
    scale: float = 20.0,
    random_state: Optional[int] = None,
):
    rng = np.random.default_rng(random_state)

    corrupted = b.copy()

    n = len(b)

    m = int(fraction * n)

    idx = rng.choice(n, size=m, replace=False)

    corrupted[idx] += scale * (1.0 + np.abs(rng.standard_t(df=1, size=m)))

    return corrupted


def mean_squared_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    
    return float(
        np.mean((y_true - y_pred) ** 2)
    )


def median_absolute_residual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    
    return float(np.median(np.abs(y_true - y_pred)))


def mean_absolute_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    
    return float(np.mean(np.abs(y_true - y_pred)))