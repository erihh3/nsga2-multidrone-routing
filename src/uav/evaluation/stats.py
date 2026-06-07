"""Cross-seed statistics.

Mann-Whitney U per metric across the 10 seeds (non-parametric — do not assume
normality). Report median + IQR, n=10. Never quote a single-seed number.

Phase 4. Stubs.
"""

from __future__ import annotations

import numpy as np


def median_iqr(samples: np.ndarray) -> tuple[float, float]:
    """Return (median, IQR) for an n=10 sample."""
    raise NotImplementedError("Phase 4: median + IQR.")


def mann_whitney(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Return (U statistic, p-value) comparing two algorithms on one metric."""
    raise NotImplementedError("Phase 4: Mann-Whitney U.")
