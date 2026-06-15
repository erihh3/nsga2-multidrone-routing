"""Static route map for one chosen Pareto-optimal solution (K colored tours from
depot). Phase 6.

Generic and pure: it takes the K routes and the instance coordinates directly (not
a JSON), so the same function renders a solution loaded from disk *or* one typed in
by hand for a talk. An empty (depot-only) route ``[depot, depot]`` is skipped — it
is an idle drone in the variable-fleet model, not a tour. Co-equality: nothing here
knows which optimizer produced the routes.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: must precede pyplot
import matplotlib.pyplot as plt
import numpy as np

# Per-drone tour color (K=3, plus spares). Distinct from the pareto/convergence
# palettes so a route map is never confused with a front.
_DRONE_COLORS: tuple[str, ...] = ("#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
                                  "#ff7f00")


def _is_active(route) -> bool:
    """An active tour visits at least one POI: longer than ``[depot, depot]``."""
    return len(route) > 2


def _route_length(route, coords: np.ndarray) -> float:
    """Geometric (Euclidean) polyline length of a route, for display annotation."""
    pts = coords[np.asarray(route, dtype=int)]
    return float(np.sqrt(((pts[1:] - pts[:-1]) ** 2).sum(axis=1)).sum())


def plot_routes(routes, coords, ax=None, *, depot=0, title=None):
    """Draw the K depot-rooted tours of one solution on the instance coordinates.

    Args:
        routes: ``list[list[int]]`` — K routes, each ``[depot, ..., depot]``. Empty
            ``[depot, depot]`` routes (idle drones) are skipped.
        coords: ``(N+1, 2)`` array; index 0 is the depot. Node id -> ``coords[id]``.
        ax: optional Axes; created if omitted.
        depot: depot node index (default 0).
        title: optional axes title (the caller typically embeds makespan/energy).

    Returns:
        The Axes drawn into.
    """
    coords = np.asarray(coords, dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 6.0))

    # All POIs as a faint backdrop so skipped/idle nodes are still visible.
    ax.scatter(coords[1:, 0], coords[1:, 1], s=14, color="0.8", zorder=1,
               linewidths=0)

    n_active = 0
    for k, route in enumerate(routes):
        if not _is_active(route):
            continue
        n_active += 1
        color = _DRONE_COLORS[k % len(_DRONE_COLORS)]
        pts = coords[np.asarray(route, dtype=int)]
        length = _route_length(route, coords)
        ax.plot(pts[:, 0], pts[:, 1], "-", color=color, linewidth=1.4, zorder=2,
                label=f"drone {k}: {len(route) - 2} POIs, d≈{length:.0f}")
        # POI markers (exclude the two depot endpoints).
        ax.scatter(pts[1:-1, 0], pts[1:-1, 1], s=28, color=color,
                   edgecolors="white", linewidths=0.5, zorder=3)

    # Depot drawn last, on top, distinctly.
    ax.scatter(coords[depot, 0], coords[depot, 1], marker="*", s=260,
               color="black", edgecolors="white", linewidths=0.8, zorder=4,
               label="depot")

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title or f"Route map — {n_active} active drone(s)")
    ax.legend(fontsize=8, loc="best")
    return ax
