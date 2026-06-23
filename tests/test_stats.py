"""Phase 4 — cross-seed statistics (median + IQR, Mann-Whitney U)."""

from __future__ import annotations

import math
import warnings

import numpy as np

from uav.evaluation.stats import mann_whitney, median_iqr


def test_median_iqr_known_sample():
    # [1..10]: median 5.5; Q1 3.25, Q3 7.75 (linear-interpolated) -> IQR 4.5.
    med, iqr = median_iqr(np.arange(1, 11, dtype=float))
    assert math.isclose(med, 5.5, rel_tol=1e-12)
    assert math.isclose(iqr, 4.5, rel_tol=1e-12)


def test_median_iqr_drops_nan():
    med, iqr = median_iqr(np.array([1.0, 2.0, 3.0, np.nan]))
    assert math.isclose(med, 2.0, rel_tol=1e-12)
    assert math.isfinite(iqr)


def test_median_iqr_all_nan_returns_nan():
    med, iqr = median_iqr(np.array([np.nan, np.nan]))
    assert math.isnan(med) and math.isnan(iqr)


def test_mann_whitney_separated_samples_significant():
    a = np.arange(1, 11, dtype=float)
    b = np.arange(101, 111, dtype=float)        # cleanly separated
    _, p = mann_whitney(a, b)
    assert p < 0.05


def test_mann_whitney_identical_samples_not_significant():
    a = np.arange(1, 11, dtype=float)
    _, p = mann_whitney(a, a.copy())
    assert p > 0.05


def test_mann_whitney_all_tied_not_significant():
    # Zero pooled variance -> p = 1.0 (no detectable difference), no exception.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # SciPy tie-correction warning
        u, p = mann_whitney(np.array([5.0, 5.0, 5.0]), np.array([5.0, 5.0, 5.0]))
    assert math.isclose(p, 1.0, rel_tol=1e-12)
    assert p > 0.05


def test_mann_whitney_empty_group_is_nan():
    u, p = mann_whitney(np.array([np.nan, np.nan]), np.array([1.0, 2.0, 3.0]))
    assert math.isnan(u) and math.isnan(p)
