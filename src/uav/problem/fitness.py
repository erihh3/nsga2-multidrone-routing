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

    Reads consecutive pairs from the precomputed matrix; no geometry here, so it
    stays cheap inside the optimizer inner loop.
    """
    return float(sum(dist[route[i], route[i + 1]] for i in range(len(route) - 1)))


def evaluate(
    routes: list[list[int]],
    dist: np.ndarray,
    alpha: float = ALPHA,
    beta: float = BETA,
    mass: float = MASS,
    v_cruise: float = V_CRUISE,
) -> tuple[float, float]:
    """Return ``(makespan_seconds, energy_joules)`` in a single pass.

    makespan = max per-drone travel time = max_k d_k / v.
    energy   = (alpha*mass + beta)/v * sum_k d_k.

    Per the formulation, under Dorling's linear model at constant mass and
    velocity the per-segment power ``alpha*mass + beta`` is constant, so f2 is that
    constant divided by v times the *total* distance. We therefore compute each
    drone's distance exactly once: makespan is their max/v, energy a scalar times
    their sum. One pass, no duplicated geometry.
    """
    per_drone_d = [route_distance(r, dist) for r in routes]
    makespan = max(per_drone_d) / v_cruise
    energy = (alpha * mass + beta) / v_cruise * sum(per_drone_d)
    return makespan, energy


class CountingEvaluator:
    """The single shared eval counter both optimizers route through.

    A thin callable over the one ``evaluate()`` that tallies how many times it is
    invoked. NSGA-II and MOPSO each build *one* instance per ``run()`` and call it
    in place of ``evaluate(routes, dist)``; ``n_calls`` is then the run's measured
    evaluation count.

    Why this exists (CLAUDE.md invariant): fitness must be counted in exactly one
    place. Before this, each optimizer kept its own ad-hoc tally (NSGA-II added
    ``len(invalid)`` per generation, MOPSO did ``n_evals += 1`` in the loop). Two
    hand-maintained counters are two chances to count differently and silently
    break the *measured* budget parity the comparison rests on. One wrapper, one
    counter, no counting inside the optimizer loops.

    The wrapper owns no objective or distance logic — it forwards verbatim to the
    shared ``evaluate()`` — so it cannot become an algorithm-specific fitness.
    """

    def __init__(
        self,
        dist: np.ndarray,
        alpha: float = ALPHA,
        beta: float = BETA,
        mass: float = MASS,
        v_cruise: float = V_CRUISE,
    ) -> None:
        self.dist = dist
        self.alpha = alpha
        self.beta = beta
        self.mass = mass
        self.v_cruise = v_cruise
        self.n_calls = 0

    def __call__(self, routes: list[list[int]]) -> tuple[float, float]:
        self.n_calls += 1
        return evaluate(
            routes, self.dist, self.alpha, self.beta, self.mass, self.v_cruise
        )

    def reset(self) -> None:
        """Zero the counter (e.g. to reuse one evaluator across calibration runs)."""
        self.n_calls = 0
