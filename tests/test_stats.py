"""Phase 4 — cross-seed statistics (median + IQR, Mann-Whitney U)."""

from __future__ import annotations

import math
import warnings

import numpy as np

from uav.evaluation.stats import holm, mann_whitney, median_iqr


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


def test_holm_step_down_stops_at_first_failure():
    # m=3, alpha=0.05. Sorted ascending: 0.01, 0.03, 0.04.
    #   k=1: 0.01 <= 0.05/3 = 0.0167  -> reject; adjusted 3*0.01 = 0.03
    #   k=2: 0.03 >  0.05/2 = 0.025    -> STOP; everything from here is kept.
    # Monotone adjusted p: [0.03, 0.06, 0.06] in input order [0.01, 0.04, 0.03].
    reject, adjusted = holm([0.01, 0.04, 0.03])
    assert reject == [True, False, False]
    assert math.isclose(adjusted[0], 0.03, rel_tol=1e-12)
    assert math.isclose(adjusted[1], 0.06, rel_tol=1e-12)
    assert math.isclose(adjusted[2], 0.06, rel_tol=1e-12)


def test_holm_all_tiny_pvalues_all_reject():
    # The project's GD/IGD headline: p ~ 3e-11 everywhere -> survives any family.
    reject, adjusted = holm([3.0e-11, 3.0e-11, 3.0e-11, 3.0e-11])
    assert reject == [True, True, True, True]
    assert all(a <= 0.05 for a in adjusted)


def test_holm_single_pvalue_reduces_to_raw_test():
    # m=1: Holm is just the uncorrected test (adjusted == raw, clamped to 1).
    assert holm([0.049]) == ([True], [0.049])
    assert holm([0.051])[0] == [False]


def test_holm_is_order_independent():
    # Permuting the inputs permutes the outputs identically (family = the set).
    base_reject, base_adj = holm([0.001, 0.02, 0.6, 0.04])
    perm_reject, perm_adj = holm([0.6, 0.04, 0.001, 0.02])
    assert perm_reject == [base_reject[2], base_reject[3], base_reject[0], base_reject[1]]
    for got, exp in zip(perm_adj, [base_adj[2], base_adj[3], base_adj[0], base_adj[1]]):
        assert math.isclose(got, exp, rel_tol=1e-12)
