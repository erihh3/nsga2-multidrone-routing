"""Seed control for reproducibility.

10 fixed seeds per (instance, algorithm). Seeds must be set for both ``random``
and ``numpy``, and — critically — *inside* any worker process if multiprocessing
is used, otherwise every worker inherits the parent's RNG state and the runs
correlate.
"""

from __future__ import annotations

import random

import numpy as np


def set_all_seeds(seed: int) -> None:
    """Seed Python's ``random`` and NumPy's legacy global RNG.

    Call this at the top of every run() and inside every worker.
    """
    random.seed(seed)
    np.random.seed(seed)
