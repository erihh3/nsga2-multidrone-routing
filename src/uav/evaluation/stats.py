"""Cross-seed statistics.

Every reported figure is n=10 (the fixed seeds), summarized by **median + IQR**
(robust to the outliers stochastic search produces) and compared between
algorithms with the **Mann-Whitney U** test — non-parametric, so it assumes
nothing about normality. Never quote a single-seed number.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import mannwhitneyu


def median_iqr(samples: np.ndarray) -> tuple[float, float]:
    """Return ``(median, IQR)`` for a sample (n=10 in this project).

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
