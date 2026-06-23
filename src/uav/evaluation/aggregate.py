"""Tie the evaluation pieces together: fronts -> normalized metrics -> stats table.

Given the per-seed fronts of both algorithms on one instance, this builds the
union reference front, normalizes every front against it, computes the geometric
metrics per (algorithm, seed), carries CT (wall-clock) and the measured n_evals
through, and summarizes each metric as median + IQR per algorithm with a
Mann-Whitney U p-value between the two.

Decoupled from the optimizers on purpose: it consumes only ``Run`` records
(seed + final-front ``Solution``s + wall-clock + n_evals), so it serves both the
live Phase-4 eil51 script and the Phase-5 runner reading saved JSON, blind to
which algorithm produced a front.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uav.evaluation import metrics as M
from uav.evaluation.reference_front import (
    build_reference_front,
    normalize,
    reference_bounds,
)
from uav.evaluation.stats import mann_whitney, median_iqr
from uav.solution import Solution

# Geometric metrics + the two carried-through reported quantities (CT, n_evals).
# CT == wall_clock_s (Wibowo & Sopha's computation-time metric).
METRIC_KEYS = ("nps", "spacing", "gd", "igd", "dm", "hv", "ct", "n_evals")


@dataclass
class Run:
    """One seeded run's reportable payload (the slice metrics need)."""

    seed: int
    front: list[Solution]
    wall_clock_s: float
    n_evals: int


def _front_objs(front: list[Solution]) -> np.ndarray:
    """Distinct objective vectors of a front (shape ``(n, 2)``).

    Quality indicators are defined on the set of distinct non-dominated objective
    vectors, so duplicates are collapsed here. This matters for a *fair* co-equal
    comparison: NSGA-II's ``final_front`` is the whole non-dominated population and
    routinely carries dozens of objective-identical clones (different genotypes,
    same (makespan, energy)), whereas MOPSO's archive is already deduplicated. Left
    raw, those clones would inflate NSGA-II's NPS to ~100, force its Schott spacing
    to 0 (zero nearest-neighbour gaps between clones), and shrink its GD (the
    distance sum is divided by an inflated point count). Collapsing to distinct
    objective vectors — identically for both algorithms — removes that artifact.
    The persisted per-run JSON keeps the *raw* fronts; only the metrics use the
    distinct set.
    """
    if not front:
        return np.empty((0, 2), dtype=float)
    objs = np.array([s.objectives for s in front], dtype=float)
    return np.unique(objs, axis=0)


def run_metrics(front_norm: np.ndarray, ref_norm: np.ndarray, ref_point: np.ndarray) -> dict:
    """All geometric metrics for one already-normalized front."""
    return {
        "nps": M.nps(front_norm),
        "spacing": M.spacing(front_norm),
        "gd": M.gd(front_norm, ref_norm),
        "igd": M.igd(front_norm, ref_norm),
        "dm": M.maximum_spread(front_norm),
        "hv": M.hypervolume(front_norm, ref_point),
    }


def aggregate(runs_by_algo: dict[str, list[Run]]) -> dict:
    """Compute per-run metrics + the cross-seed comparison table.

    Args:
        runs_by_algo: ``{algorithm_name: [Run, ...]}``. Typically two algorithms
            (NSGA-II, MOPSO), 10 runs each.

    Returns a dict with:
        - ``reference_size``: number of points on the union reference front.
        - ``ref_min`` / ``ref_max``: the normalization bounds (raw objectives).
        - ``per_run``: ``{algo: [{seed, nps, spacing, gd, igd, dm, hv, ct,
          n_evals}, ...]}`` on normalized objectives.
        - ``summary``: ``{metric: {algo: {median, iqr}, ..., "p": <mwU p-value>}}``
          (the p-value present only when exactly two algorithms are compared).
    """
    all_fronts = [r.front for runs in runs_by_algo.values() for r in runs]
    reference = build_reference_front(all_fronts)
    ref_min, ref_max = reference_bounds(reference)
    ref_norm = normalize(_front_objs(reference), ref_min, ref_max)
    # Normalized nadir is (1, 1); the HV reference point is 1.1 x that.
    ref_point = np.array([1.1, 1.1])

    per_run: dict[str, list[dict]] = {}
    for algo, runs in runs_by_algo.items():
        rows = []
        for r in runs:
            objs = _front_objs(r.front)
            fnorm = normalize(objs, ref_min, ref_max) if objs.size else objs
            row = {"seed": r.seed}
            row.update(run_metrics(fnorm, ref_norm, ref_point))
            row["ct"] = r.wall_clock_s
            row["n_evals"] = r.n_evals
            rows.append(row)
        per_run[algo] = rows

    algos = list(runs_by_algo.keys())
    summary: dict[str, dict] = {}
    for key in METRIC_KEYS:
        entry: dict = {}
        for algo in algos:
            vals = np.array([row[key] for row in per_run[algo]], dtype=float)
            med, iqr = median_iqr(vals)
            entry[algo] = {"median": med, "iqr": iqr}
        if len(algos) == 2:
            a = np.array([row[key] for row in per_run[algos[0]]], dtype=float)
            b = np.array([row[key] for row in per_run[algos[1]]], dtype=float)
            u, p = mann_whitney(a, b)
            entry["U"], entry["p"] = u, p
        summary[key] = entry

    return {
        "reference_size": len(reference),
        "ref_min": ref_min.tolist(),
        "ref_max": ref_max.tolist(),
        "per_run": per_run,
        "summary": summary,
    }
