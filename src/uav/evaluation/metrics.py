"""Quality indicators for the NSGA-II vs MOPSO comparison.

The comparison reports the five Wibowo & Sopha (2021) metrics — **NPS, CT,
spacing, GD, DM** — plus **HV** and **IGD** as extra coverage. CT (computation
time) is not a front-geometry quantity; it is each run's wall-clock
(``RunResult.wall_clock_s``), aggregated in ``stats.py`` / ``aggregate.py``. The
six functions here are the geometric indicators.

**Every distance-based metric (spacing, GD, IGD, DM, HV) operates on NORMALIZED
objectives** (see ``reference_front.normalize``). The energy axis spans only ~1%
in absolute terms while makespan spans tens of seconds, so without normalization
makespan would dominate every Euclidean distance and HV would be governed by
energy's raw magnitude. Callers must pass objectives already mapped into the
reference front's ``[0,1]`` box; these functions do not normalize internally.

Conventions (both objectives MINIMIZED):
- ``front`` / ``points`` / ``reference`` are ``np.ndarray`` of shape ``(n, 2)``.
- "front" = an algorithm's (one seed's) non-dominated objective set; "reference"
  = the union reference front (``reference_front.build_reference_front``).

References (unverified DOIs; flag in the paper):
- Schott, "Fault tolerant design using single and multicriteria GA optimization",
  MSc thesis MIT, 1995 — spacing.
- Van Veldhuizen & Lamont, "Multiobjective evolutionary algorithm research: a
  history and analysis", 1998 — generational distance (GD).
- Zitzler & Thiele, "Multiobjective evolutionary algorithms: a comparative case
  study and the strength Pareto approach", IEEE TEVC 1999 — hypervolume, maximum
  spread.
"""

from __future__ import annotations

import numpy as np


# --- helpers --------------------------------------------------------------------

def _as2d(points: np.ndarray) -> np.ndarray:
    """Coerce to a float ``(n, 2)`` array (tolerates an empty input)."""
    arr = np.asarray(points, dtype=float)
    if arr.size == 0:
        return arr.reshape(0, 2)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"expected shape (n, 2), got {arr.shape}")
    return arr


def _min_dist_to_set(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """For each row of ``a``, the minimum Euclidean distance to any row of ``b``.

    Shapes: ``a`` is ``(na, 2)``, ``b`` is ``(nb, 2)`` -> returns ``(na,)``.
    """
    # (na, nb) pairwise Euclidean distances via broadcasting.
    diff = a[:, None, :] - b[None, :, :]
    d = np.sqrt((diff ** 2).sum(axis=2))
    return d.min(axis=1)


def _nondominated_mask(points: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated rows (minimization). O(n^2), n is tiny."""
    n = points.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        pi = points[i]
        for j in range(n):
            if i == j:
                continue
            pj = points[j]
            # pj dominates pi?
            if (pj[0] <= pi[0] and pj[1] <= pi[1]) and (pj[0] < pi[0] or pj[1] < pi[1]):
                keep[i] = False
                break
    return keep


# --- metrics --------------------------------------------------------------------

def nps(front: np.ndarray) -> int:
    """Number of Pareto solutions: the count of points on the front.

    More is generally better (a richer trade-off set), though it must be read
    alongside spacing/spread — many clustered points are not informative.
    """
    return int(_as2d(front).shape[0])


def spacing(front: np.ndarray) -> float:
    """Schott's spacing: uniformity of the gaps between adjacent front points.

    For each point, ``d_i`` is the L1 (Manhattan) distance to its nearest
    neighbour on the front; with ``d_bar`` their mean,

        S = sqrt( (1 / (n - 1)) * sum_i (d_i - d_bar)^2 ).

    Lower is better (0 = perfectly even spacing). **Returns ``np.nan`` for fewer
    than 3 points** (project invariant): with <3 points the spread of
    nearest-neighbour gaps is undefined/degenerate, and returning 0 would
    masquerade as a *perfect* score. NaN forces the degenerate case to be handled
    explicitly downstream rather than silently rewarded.
    """
    f = _as2d(front)
    n = f.shape[0]
    if n < 3:
        return float("nan")
    # L1 nearest-neighbour distance per point (Schott uses the city-block metric).
    l1 = np.abs(f[:, None, :] - f[None, :, :]).sum(axis=2)
    np.fill_diagonal(l1, np.inf)
    d = l1.min(axis=1)
    d_bar = d.mean()
    return float(np.sqrt(((d - d_bar) ** 2).sum() / (n - 1)))


def gd(front: np.ndarray, reference: np.ndarray) -> float:
    """Generational distance (Van Veldhuizen): convergence of a front to the
    reference set.

        GD = sqrt( sum_{a in front} dist(a, reference)^2 ) / |front|,

    where ``dist`` is the Euclidean distance from ``a`` to its nearest reference
    point. Lower is better (0 = every front point lies on the reference front).
    Returns ``np.nan`` for an empty front or empty reference.
    """
    f = _as2d(front)
    r = _as2d(reference)
    if f.shape[0] == 0 or r.shape[0] == 0:
        return float("nan")
    d = _min_dist_to_set(f, r)
    return float(np.sqrt((d ** 2).sum()) / f.shape[0])


def igd(front: np.ndarray, reference: np.ndarray) -> float:
    """Inverted generational distance: coverage of the reference set by a front.

        IGD = (1 / |reference|) * sum_{r in reference} dist(r, front),

    the mean Euclidean distance from each reference point to its nearest front
    point. Penalizes gaps in the front (parts of the reference left uncovered) as
    well as poor convergence, so it captures convergence *and* diversity. Lower is
    better. Returns ``np.nan`` for an empty front or empty reference.
    """
    f = _as2d(front)
    r = _as2d(reference)
    if f.shape[0] == 0 or r.shape[0] == 0:
        return float("nan")
    d = _min_dist_to_set(r, f)
    return float(d.mean())


def maximum_spread(front: np.ndarray) -> float:
    """Maximum spread / diversity metric (DM): the diagonal of the front's
    objective-space bounding box.

        DM = sqrt( sum_m (max_m f_m - min_m f_m)^2 ).

    Larger is better (the front reaches further across objective space). Returns
    ``0.0`` for fewer than 2 points (a single point spans nothing) — graceful, not
    NaN, because a degenerate spread of 0 is a meaningful (bad) value here, unlike
    spacing where 0 would read as perfect.
    """
    f = _as2d(front)
    if f.shape[0] < 2:
        return 0.0
    span = f.max(axis=0) - f.min(axis=0)
    return float(np.sqrt((span ** 2).sum()))


def hypervolume(points: np.ndarray, ref_point: np.ndarray) -> float:
    """2D hypervolume dominated by ``points`` relative to ``ref_point`` (both
    objectives minimized).

    The reference point is the worst corner — by convention ``1.1 x`` the
    normalized nadir, i.e. ``(1.1, 1.1)`` in the normalized box. Larger HV is
    better. Implemented as an exact 2D staircase sweep: dominated points and
    points not better than the reference contribute nothing, so the input is first
    reduced to the non-dominated subset that strictly dominates ``ref_point``.
    Returns ``0.0`` when nothing dominates the reference.
    """
    p = _as2d(points)
    ref = np.asarray(ref_point, dtype=float)
    if p.shape[0] == 0:
        return 0.0
    # Keep only points that strictly dominate the reference on both axes.
    p = p[(p[:, 0] < ref[0]) & (p[:, 1] < ref[1])]
    if p.shape[0] == 0:
        return 0.0
    p = p[_nondominated_mask(p)]
    # Sort by f1 ascending; on a front f2 is then descending.
    p = p[np.argsort(p[:, 0])]
    area = 0.0
    prev_f2 = ref[1]
    for x, y in p:
        area += (ref[0] - x) * (prev_f2 - y)
        prev_f2 = y
    return float(area)
