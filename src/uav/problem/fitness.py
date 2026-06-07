"""The single fitness function. Neither algorithm gets its own.

Objectives (both minimized), per the formal formulation:

    makespan f1 = max_k (1/v) * sum_i d(r_k[i], r_k[i+1])          [seconds]
    energy   f2 = (alpha*m + beta) * (1/v) * sum_k sum_i d(...)    [joules]

Under Dorling's linear model at constant mass and velocity, per-segment power is
constant, so f2 is proportional to *total* tour length. Compute per-drone
distances once: makespan is their max/v, energy is a scalar times their sum.
One pass, no duplicated geometry.

Locked parameters (never change — sensitivity analysis is out of scope):
alpha=46.7 W/kg, beta=26.9 W, m=2.0 kg, v_cruise=15 m/s.
"""

from __future__ import annotations

import numpy as np

# Locked UAV parameters (Dorling 2017 linear model). Do not edit.
ALPHA = 46.7      # W/kg
BETA = 26.9       # W
MASS = 2.0        # kg
V_CRUISE = 15.0   # m/s


def route_distance(route: list[int], dist: np.ndarray) -> float:
    """Sum of edge distances along a depot-bookended route.

    Phase 1. Stub.
    """
    raise NotImplementedError("Phase 1: route distance.")


def evaluate(
    routes: list[list[int]],
    dist: np.ndarray,
    alpha: float = ALPHA,
    beta: float = BETA,
    mass: float = MASS,
    v_cruise: float = V_CRUISE,
) -> tuple[float, float]:
    """Return ``(makespan_seconds, energy_joules)`` in a single pass.

    Phase 1. **Hard gate**: prove this against a hand calculation (depot + 4 POIs
    on a unit grid, 2 drones) before writing any optimizer.
    """
    raise NotImplementedError("Phase 1: makespan + Dorling energy.")
