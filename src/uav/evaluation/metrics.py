"""Quality indicators: hypervolume, IGD, GD, spacing, NPS.

HV reference point = 1.1 x the normalized nadir. All metrics operate on
normalized objectives (see reference_front.py).

Phase 4. Stubs.
"""

from __future__ import annotations

import numpy as np


def hypervolume(points: np.ndarray, ref_point: np.ndarray) -> float:
    raise NotImplementedError("Phase 4: hypervolume.")


def igd(front: np.ndarray, reference: np.ndarray) -> float:
    raise NotImplementedError("Phase 4: IGD.")


def gd(front: np.ndarray, reference: np.ndarray) -> float:
    raise NotImplementedError("Phase 4: GD.")


def spacing(front: np.ndarray) -> float:
    raise NotImplementedError("Phase 4: spacing.")


def nps(front: np.ndarray) -> int:
    """Number of Pareto solutions."""
    raise NotImplementedError("Phase 4: NPS.")
