"""FuncAnimation: K drones interpolated along segments at v_cruise, time-synced;
export MP4 (fallback GIF via pillow if ffmpeg missing). Phase 6.

Generic and pure (same ``(routes, coords)`` interface as ``plot_routes``): it
builds and returns a ``FuncAnimation``; *saving* (with the ffmpeg->pillow writer
fallback) is the driver's job, so this stays testable without a writer.

Time-sync is the point: every drone flies at the same ``v_cruise``, so at wall
time ``t`` a drone has covered ``v_cruise·t`` of its polyline (clamped at its end).
The longest tour defines the makespan, so that drone is the last to land — the
animation *shows* f1. Idle ``[depot, depot]`` routes are skipped.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: must precede pyplot
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from uav.problem.fitness import V_CRUISE
from uav.viz.routes import _DRONE_COLORS, _is_active


def _cumulative_lengths(pts: np.ndarray) -> np.ndarray:
    """Cumulative arc length at each vertex of a polyline (starts at 0)."""
    seg = np.sqrt(((pts[1:] - pts[:-1]) ** 2).sum(axis=1))
    return np.concatenate([[0.0], np.cumsum(seg)])


def _position_at(polyline, dist_covered: float):
    """Point at arc-length ``dist_covered`` along ``polyline`` (clamped to its ends).

    At ``0`` returns the start (depot); at/after the total length returns the final
    vertex (depot). Linear interpolation within the containing segment.
    """
    pts = np.asarray(polyline, dtype=float)
    cum = _cumulative_lengths(pts)
    total = cum[-1]
    if total <= 0:                       # degenerate (single point / zero length)
        return float(pts[0, 0]), float(pts[0, 1])
    d = min(max(dist_covered, 0.0), total)
    j = int(np.searchsorted(cum, d, side="right")) - 1
    j = min(max(j, 0), len(pts) - 2)     # clamp into a valid segment
    seg_len = cum[j + 1] - cum[j]
    frac = 0.0 if seg_len <= 0 else (d - cum[j]) / seg_len
    p = pts[j] + frac * (pts[j + 1] - pts[j])
    return float(p[0]), float(p[1])


def animate_routes(routes, coords, *, v_cruise: float = V_CRUISE, depot: int = 0,
                   n_frames: int = 200, fps: int = 20, title=None):
    """Build a time-synced drone animation for one solution.

    Args:
        routes: ``list[list[int]]`` K routes; idle ``[depot, depot]`` ones skipped.
        coords: ``(N+1, 2)`` coordinates; index 0 is the depot.
        v_cruise: cruise speed (m/s); makespan = max tour length / v_cruise.
        depot: depot node index.
        n_frames, fps: playback length is ``n_frames/fps`` s; positions are
            time-synced to the real makespan regardless of playback length.
        title: optional axes title.

    Returns:
        A ``FuncAnimation`` (the driver saves it with a writer fallback).
    """
    coords = np.asarray(coords, dtype=float)
    active = [(k, np.asarray(r, dtype=int)) for k, r in enumerate(routes)
              if _is_active(r)]

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    ax.scatter(coords[1:, 0], coords[1:, 1], s=14, color="0.85", zorder=1,
               linewidths=0)
    ax.scatter(coords[depot, 0], coords[depot, 1], marker="*", s=260,
               color="black", edgecolors="white", linewidths=0.8, zorder=5)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    drones = []  # per active drone: polyline pts, total length, trail line, marker
    max_len = 0.0
    for idx, (k, route) in enumerate(active):
        pts = coords[route]
        cum = _cumulative_lengths(pts)
        total = float(cum[-1])
        max_len = max(max_len, total)
        color = _DRONE_COLORS[k % len(_DRONE_COLORS)]
        ax.plot(pts[:, 0], pts[:, 1], "-", color=color, linewidth=0.7,
                alpha=0.3, zorder=2)                       # faint full route
        (trail,) = ax.plot([], [], "-", color=color, linewidth=2.0, zorder=3,
                           label=f"drone {k}")
        (marker,) = ax.plot([], [], "o", color=color, markersize=9,
                            markeredgecolor="white", zorder=4)
        drones.append({"pts": pts, "total": total, "trail": trail,
                       "marker": marker})

    makespan = max_len / v_cruise if max_len > 0 else 0.0
    ax.set_title(title or f"Drone animation — makespan {makespan:.1f}s")
    ax.legend(fontsize=8, loc="best")
    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top",
                        fontsize=9)

    def update(frame):
        t = makespan * frame / max(n_frames - 1, 1)        # wall time this frame
        covered = v_cruise * t
        artists = [time_text]
        for d in drones:
            x, y = _position_at(d["pts"], covered)
            d["marker"].set_data([x], [y])
            # Trail = polyline vertices already passed, plus the current point.
            cum = _cumulative_lengths(d["pts"])
            passed = d["pts"][cum <= min(covered, d["total"])]
            tx = np.append(passed[:, 0], x)
            ty = np.append(passed[:, 1], y)
            d["trail"].set_data(tx, ty)
            artists += [d["marker"], d["trail"]]
        time_text.set_text(f"t = {t:5.1f} s / {makespan:.1f} s")
        return artists

    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=1000 / fps, blit=False)
    return anim
