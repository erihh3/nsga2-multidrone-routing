"""Cross-seed statistics.

Every reported figure is n=30 (the fixed seeds), summarized by **median + IQR**
(robust to the outliers stochastic search produces) and compared between
algorithms with the **Mann-Whitney U** test — non-parametric, so it assumes
nothing about normality. Never quote a single-seed number.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import mannwhitneyu

ALPHA = 0.05


def median_iqr(samples: np.ndarray) -> tuple[float, float]:
    """Return ``(median, IQR)`` for a sample (n=30 in this project).

    IQR is the interquartile range ``Q3 - Q1`` (linear-interpolated percentiles),
    the robust spread we report alongside the median. NaNs are dropped first so a
    degenerate-front ``spacing`` (NaN for <3 points) doesn't poison the summary;
    an all-NaN or empty sample returns ``(nan, nan)``.
    """
    arr = np.asarray(samples, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    median = float(np.median(arr))
    q1, q3 = np.percentile(arr, [25, 75])
    return median, float(q3 - q1)


def mann_whitney(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sided Mann-Whitney U comparing two algorithms on one metric.

    Returns ``(U, p)``. The null hypothesis is that the two samples come from the
    same distribution; a small ``p`` means the algorithms differ significantly on
    this metric across seeds. NaNs are dropped from each group first (so a
    degenerate-front ``spacing`` of NaN doesn't poison the test). An empty group
    after dropping returns ``(nan, nan)``. Fully-tied data (zero pooled variance,
    e.g. every seed gives the same NPS) yields ``p = 1.0`` — correctly read as "no
    detectable difference" — though SciPy emits a tie-correction warning.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        return float("nan"), float("nan")
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    return float(u), float(p)


def holm(pvalues, alpha: float = ALPHA) -> tuple[list[bool], list[float]]:
    """Holm (step-down Bonferroni) correction for a *family* of hypotheses.

    Controls the family-wise error rate (the probability of *any* false rejection)
    across the ``m = len(pvalues)`` tests handed in. Holm is uniformly more powerful
    than plain Bonferroni: sort the p-values ascending and reject the k-th smallest
    (1-based) while ``p_(k) <= alpha / (m - k + 1)``, stopping at the first failure
    (every larger p-value is then kept). The *family* is exactly the list passed in —
    the caller fixes the granularity (e.g. the 7 metrics of one instance, or all 28
    metric x instance tests) by slicing the p-values accordingly; there is no
    canonical family for a single pairwise comparison spread over a metric x problem
    grid, so the choice is the caller's.

    Returns ``(reject, adjusted)``, both parallel to the input order. ``adjusted`` is
    the monotone Holm-adjusted p-value (statsmodels convention,
    ``max_{j<=k} min(1, (m - j + 1) * p_(j))``) and ``reject[i]`` is
    ``adjusted[i] <= alpha`` — equivalent to the step-down rule above. Inputs must be
    finite; the reporting layer drops any NaN before forming the family.
    """
    p = [float(x) for x in pvalues]
    m = len(p)
    if m == 0:
        return [], []
    order = sorted(range(m), key=lambda i: p[i])     # indices, ascending by p-value
    adjusted = [0.0] * m
    running = 0.0                                    # running max enforces monotonicity
    for k, idx in enumerate(order):                  # k is the 0-based rank
        val = min(1.0, (m - k) * p[idx])             # factor (m - k) == m - (k+1) + 1
        running = max(running, val)
        adjusted[idx] = running
    reject = [adjusted[i] <= alpha for i in range(m)]
    return reject, adjusted
