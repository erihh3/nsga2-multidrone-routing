"""Overlay both algorithms' Pareto fronts per instance (vector PDF). Phase 6.

Co-equality: this renders any number of arms identically — it never branches on
*which* algorithm produced a front. An arm contributes only a label and a marker
shape; the *color* encodes the variable-fleet metadata (number of active drones),
which is the headline story (NSGA-II spreads across fleet sizes 1/2/3; MOPSO
collapses to 3). Points are pooled across the 10 seeds by the caller and reduced
here to the union non-dominated front (never a single seed).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: must precede pyplot
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Reuse the project's single non-dominated routine (do not re-implement domination).
from uav.evaluation.metrics import _nondominated_mask

# Fleet size -> color. Shared by every arm so the legend reads "color == fleet
# size", not "color == algorithm". Keys are the only fleet sizes K=3 admits.
_FLEET_COLORS: dict[int, str] = {1: "#d62728", 2: "#1f77b4", 3: "#2ca02c"}
# Per-arm marker shapes, assigned in call order (the only per-algorithm visual).
_ALGO_MARKERS: tuple[str, ...] = ("o", "s", "^", "D")


def _union_front(points: np.ndarray) -> np.ndarray:
    """Indices of the non-dominated subset, ordered by makespan (for a step line)."""
    if len(points) == 0:
        return np.empty(0, dtype=int)
    keep = np.flatnonzero(_nondominated_mask(points))
    return keep[np.argsort(points[keep, 0])]


def plot_pareto(records_by_algo, ax=None, *, title=None):
    """Overlay each arm's union Pareto front, colored by active-drone count.

    Args:
        records_by_algo: ``{label: list[(makespan, energy, n_active)]}`` — points
            already pooled across seeds. ``n_active`` is the variable-fleet count
            (derived from routes by the caller, never read off a JSON key so it is
            consistent for both arms).
        ax: optional Axes to draw into; one is created if omitted.
        title: optional axes title.

    Returns:
        The Axes drawn into.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 4.5))

    seen_fleet: set[int] = set()
    algo_handles: list[Line2D] = []

    for i, (label, records) in enumerate(records_by_algo.items()):
        marker = _ALGO_MARKERS[i % len(_ALGO_MARKERS)]
        algo_handles.append(
            Line2D([], [], marker=marker, color="0.3", linestyle="none",
                   markersize=7, label=label)
        )
        if not records:
            continue

        arr = np.asarray(records, dtype=float)
        objs, fleet = arr[:, :2], arr[:, 2].astype(int)

        # Faint cloud of every pooled (dominated + non-dominated) point for context.
        ax.scatter(objs[:, 0], objs[:, 1], s=12, color="0.8", alpha=0.35,
                   zorder=1, linewidths=0)

        # Union non-dominated front: thin connecting step + generous fleet-colored
        # markers. The energy axis spans ~1%, so the front is short and stair-like
        # rather than a long smooth curve — markers carry the information.
        u = _union_front(objs)
        ufront = objs[u]
        ax.step(ufront[:, 0], ufront[:, 1], where="post", color="0.5",
                linewidth=0.8, zorder=2)
        for size in np.unique(fleet[u]):
            sel = u[fleet[u] == size]
            seen_fleet.add(int(size))
            ax.scatter(objs[sel, 0], objs[sel, 1], marker=marker, s=70,
                       color=_FLEET_COLORS.get(int(size), "0.2"),
                       edgecolors="black", linewidths=0.6, zorder=3)

    ax.set_xlabel("makespan $f_1$ (s)")
    ax.set_ylabel("total energy $f_2$ (J)")
    if title:
        ax.set_title(title)

    # Two-part legend: marker shape == algorithm, color == active-drone count.
    fleet_handles = [
        Line2D([], [], marker="o", color=_FLEET_COLORS[s], linestyle="none",
               markeredgecolor="black", markersize=7, label=f"{s} drone(s)")
        for s in sorted(seen_fleet)
    ]
    first = ax.legend(handles=algo_handles, title="algorithm", loc="upper right",
                      fontsize=8, title_fontsize=8)
    ax.add_artist(first)
    if fleet_handles:
        ax.legend(handles=fleet_handles, title="active drones", loc="lower left",
                  fontsize=8, title_fontsize=8)

    # The thin-energy axis is a real structural fact — annotate it so a reader does
    # not mistake the short front for poor sampling.
    ax.annotate("energy axis spans ~1% — trade-off lives in makespan",
                xy=(0.5, 0.01), xycoords="axes fraction", ha="center",
                va="bottom", fontsize=7, color="0.4")
    return ax
