"""Phase 6 — smoke + unit tests for the visualization layer.

These run on the Agg backend with tiny synthetic data: they prove each figure
renders (a non-empty file / a Figure) without raising, and unit-test the pure
helpers (union extraction, segment interpolation, knee selection). No real sweep
is run — figures are reproducible from the persisted JSONs, which the driver
(`scripts/make_figures.py`) handles, not these tests.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from uav.viz.animation import _position_at, animate_routes
from uav.viz.convergence import _stack, plot_convergence
from uav.viz.pareto import _union_front, plot_pareto
from uav.viz.routes import _is_active, _route_length, plot_routes


def _toy_history(n_gens=5, scale=1.0):
    """One run's history: best <= mean <= worst, decreasing over generations."""
    return [
        {"gen": g,
         "best": [scale * (10 - g), scale * (100 - g)],
         "mean": [scale * (12 - g), scale * (120 - g)],
         "worst": [scale * (15 - g), scale * (150 - g)]}
        for g in range(n_gens)
    ]


# --- pareto --------------------------------------------------------------------

def test_union_front_keeps_only_nondominated_sorted():
    # (mk, en): (1,3) and (2,2) and (3,1) are mutually non-dominated; (4,4) is
    # dominated by all. Result must be sorted by makespan.
    pts = np.array([[3.0, 1.0], [1.0, 3.0], [4.0, 4.0], [2.0, 2.0]])
    u = _union_front(pts)
    assert list(pts[u, 0]) == [1.0, 2.0, 3.0]
    # the dominated (4,4) is gone
    assert 4.0 not in pts[u, 0]


def test_union_front_empty():
    assert len(_union_front(np.empty((0, 2)))) == 0


def test_plot_pareto_renders_pdf(tmp_path):
    records = {
        "NSGA-II": [(22.5, 4400.0, 3), (24.0, 4300.0, 2), (30.0, 4200.0, 1),
                    (26.0, 4500.0, 3)],
        "MOPSO": [(23.0, 4600.0, 3), (25.0, 4550.0, 3)],
    }
    ax = plot_pareto(records, title="toy")
    out = tmp_path / "pareto.pdf"
    ax.figure.savefig(out)
    plt.close(ax.figure)
    assert out.exists() and out.stat().st_size > 0


def test_plot_pareto_tolerates_empty_arm():
    ax = plot_pareto({"NSGA-II": [(1.0, 2.0, 3)], "MOPSO": []})
    plt.close(ax.figure)


# --- convergence ---------------------------------------------------------------

def test_stack_shape_and_values():
    hs = [_toy_history(n_gens=4), _toy_history(n_gens=4, scale=1.0)]
    s = _stack(hs, "mean", obj=0)  # makespan means
    assert s.shape == (2, 4)
    assert s[0, 0] == 12.0  # mean makespan at gen 0

def test_plot_convergence_renders_pdf(tmp_path):
    histories = {
        "NSGA-II": [_toy_history(6) for _ in range(3)],
        # different length per arm (parity iters) must not break plotting
        "MOPSO": [_toy_history(4) for _ in range(3)],
    }
    axes = plot_convergence(histories)
    out = tmp_path / "conv.pdf"
    axes[0].figure.savefig(out)
    plt.close(axes[0].figure)
    assert out.exists() and out.stat().st_size > 0


# --- routes --------------------------------------------------------------------

_COORDS = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])  # depot + 3 POIs


def test_is_active_skips_empty_route():
    assert not _is_active([0, 0])      # idle drone
    assert _is_active([0, 1, 0])       # one POI

def test_route_length_unit_square_leg():
    # depot(0,0) -> (1,0) -> depot : length 2
    assert _route_length([0, 1, 0], _COORDS) == pytest.approx(2.0)

def test_plot_routes_renders_and_skips_empty(tmp_path):
    routes = [[0, 1, 2, 0], [0, 3, 0], [0, 0]]  # third drone idle
    ax = plot_routes(routes, _COORDS, title="toy routes")
    out = tmp_path / "routes.pdf"
    ax.figure.savefig(out)
    plt.close(ax.figure)
    assert out.exists() and out.stat().st_size > 0


# --- animation -----------------------------------------------------------------

def test_position_at_endpoints():
    # depot(0,0) -> (1,0) -> (1,1): total length 2.
    poly = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    assert _position_at(poly, 0.0) == pytest.approx((0.0, 0.0))      # start (depot)
    assert _position_at(poly, 0.5) == pytest.approx((0.5, 0.0))      # mid first leg
    assert _position_at(poly, 2.0) == pytest.approx((1.0, 1.0))      # end
    assert _position_at(poly, 99.0) == pytest.approx((1.0, 1.0))     # clamped past end

def test_position_at_degenerate_zero_length():
    assert _position_at(np.array([[3.0, 3.0], [3.0, 3.0]]), 1.0) == pytest.approx((3.0, 3.0))

def test_animate_routes_saves_gif(tmp_path):
    routes = [[0, 1, 2, 0], [0, 3, 0], [0, 0]]  # one idle drone, skipped
    anim = animate_routes(routes, _COORDS, n_frames=5, fps=5)
    out = tmp_path / "anim.gif"
    anim.save(out, writer="pillow", fps=5)      # pillow always available (no ffmpeg)
    plt.close(anim._fig)
    assert out.exists() and out.stat().st_size > 0
