"""Reference front + objective normalization.

Reference front: the union of all non-dominated solutions across both algorithms
x all seeds for an instance, re-filtered to non-dominated. This is the standard
proxy used when no true Pareto front is known (GD/IGD need *some* reference) —
disclose it as such in the paper.

Normalization: makespan (tens-hundreds of s) and energy (large J) live on wildly
different scales, and the energy axis spans only ~1% of its magnitude. Every
Euclidean/area metric would otherwise be governed by makespan's range and
energy's raw magnitude. Mapping each objective to ``[0,1]`` via the reference
front's per-objective min/max puts both axes on equal footing *before* any
distance-based metric (GD, IGD, spacing, DM, HV).
"""

from __future__ import annotations

import warnings

import numpy as np

from uav.solution import Solution


def _dominates(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """True iff objective vector ``a`` dominates ``b`` (both minimized)."""
    return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])


def build_reference_front(fronts: list[list[Solution]]) -> list[Solution]:
    """Union of all fronts, deduplicated by objective vector and re-filtered to
    non-dominated.

    Args:
        fronts: one list of ``Solution`` per (algorithm, seed) run. Mixed freely —
            the reference front is blind to which optimizer produced each point.

    Returns:
        The non-dominated set over the union, one representative ``Solution`` per
        distinct objective vector, sorted by makespan (f1) ascending for stable,
        readable output.
    """
    # Collapse to one representative per objective vector first (cheap dedup).
    by_obj: dict[tuple[float, float], Solution] = {}
    for front in fronts:
        for sol in front:
            by_obj.setdefault(sol.objectives, sol)

    candidates = list(by_obj.items())
    keep = [
        sol
        for oi, sol in candidates
        if not any(_dominates(oj, oi) for oj, _ in candidates if oj != oi)
    ]
    keep.sort(key=lambda s: s.objectives)
    return keep


def reference_bounds(ref_front: list[Solution]) -> tuple[np.ndarray, np.ndarray]:
    """Per-objective ``(min, max)`` over the reference front, each shape ``(2,)``.

    These bounds define the normalization box. Raises on an empty front (there is
    nothing to normalize against).
    """
    if not ref_front:
        raise ValueError("reference front is empty; cannot derive bounds")
    objs = np.array([s.objectives for s in ref_front], dtype=float)
    return objs.min(axis=0), objs.max(axis=0)


def normalize(
    objs: np.ndarray, ref_min: np.ndarray, ref_max: np.ndarray
) -> np.ndarray:
    """Map objective vectors into ``[0,1]`` via the reference front's min/max.

        normalized = (objs - ref_min) / (ref_max - ref_min)

    Points on the reference extremes land at 0 and 1; the thin energy axis is
    stretched to fill ``[0,1]`` so it weighs equally with makespan in every
    distance metric (this is the whole point of normalizing).

    **Near-zero-range guard (project invariant).** A genuinely degenerate axis
    (every reference point sharing one value, so ``ref_max - ref_min`` ~ 0) would
    divide by ~0 and blow up to inf. Such an axis is collapsed to 0 with a warning
    instead. The guard triggers only on a *numerically* zero range — the ~1%
    energy span is far above the threshold and normalizes normally.
    """
    objs = np.asarray(objs, dtype=float)
    ref_min = np.asarray(ref_min, dtype=float)
    ref_max = np.asarray(ref_max, dtype=float)
    rng = ref_max - ref_min

    # Degenerate iff the span is negligible relative to the axis magnitude.
    scale = np.maximum.reduce([np.abs(ref_min), np.abs(ref_max), np.ones_like(rng)])
    degenerate = rng <= 1e-9 * scale

    safe_rng = np.where(degenerate, 1.0, rng)   # avoid the divide; overwrite below
    out = (objs - ref_min) / safe_rng
    if np.any(degenerate):
        warnings.warn(
            f"normalize: degenerate objective axis (near-zero range) at "
            f"{np.flatnonzero(degenerate).tolist()}; collapsed to 0.",
            RuntimeWarning,
            stacklevel=2,
        )
        out[..., degenerate] = 0.0
    return out
