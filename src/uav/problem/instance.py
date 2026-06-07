"""mTSPLIB instance loader + precomputed distance matrix.

Distance convention is load-bearing: TSPLIB ``EUC_2D`` defines
``d(i,j) = nint(sqrt((xi-xj)^2 + (yi-yj)^2))`` — rounded to the *nearest integer*.
Using raw floats would desync our numbers from the CPLEX-verified optima we cite.

Precompute the full O(N^2) matrix once (trivial at N<=99) so the optimizer inner
loops never call sqrt.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Instance:
    name: str            # e.g. "eil51-k3"
    coords: np.ndarray   # shape (N+1, 2), index 0 = depot
    dist: np.ndarray     # shape (N+1, N+1), nint EUC_2D, precomputed
    n_pois: int          # N (depot excluded)
    k: int               # number of drones (3)
    depot: int = 0


def load_instance(tsp_path: str, k: int) -> Instance:
    """Parse an mTSPLIB ``.tsp`` file into coords + nint distance matrix.

    Phase 1. Stub until the instance files are on disk.
    """
    raise NotImplementedError("Phase 1: implement mTSPLIB EUC_2D loader.")
