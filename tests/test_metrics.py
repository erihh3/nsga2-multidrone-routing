"""Phase 4 — quality metrics, validated against hand-computed values.

These tests prove the harness is correct *independently of MOPSO/NSGA-II output*:
every metric is checked on tiny synthetic fronts whose answer is computed by hand
here, plus the degenerate edge cases the project invariants call out (spacing NaN
for <3 points; graceful single-point handling).

All inputs are treated as already-normalized objectives (shape (n, 2),
minimization), exactly as the real pipeline feeds them.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from uav.evaluation.metrics import (
    gd,
    hypervolume,
    igd,
    maximum_spread,
    nps,
    spacing,
)


# --- NPS -------------------------------------------------------------------------

def test_nps_counts_points():
    assert nps(np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])) == 3
    assert nps(np.empty((0, 2))) == 0
    assert nps(np.array([[0.5, 0.5]])) == 1


# --- GD --------------------------------------------------------------------------

def test_gd_single_point_by_hand():
    # (0,1) is distance 1 from the only reference point (0,0). GD = sqrt(1^2)/1.
    front = np.array([[0.0, 1.0]])
    ref = np.array([[0.0, 0.0]])
    assert math.isclose(gd(front, ref), 1.0, rel_tol=1e-12)


def test_gd_multi_point_by_hand():
    # Distances to the single reference (0,0): (0,0)->0, (3,4)->5.
    # GD = sqrt(0^2 + 5^2) / 2 = 5/2.
    front = np.array([[0.0, 0.0], [3.0, 4.0]])
    ref = np.array([[0.0, 0.0]])
    assert math.isclose(gd(front, ref), 2.5, rel_tol=1e-12)


def test_gd_empty_is_nan():
    assert math.isnan(gd(np.empty((0, 2)), np.array([[0.0, 0.0]])))
    assert math.isnan(gd(np.array([[0.0, 0.0]]), np.empty((0, 2))))


# --- IGD -------------------------------------------------------------------------

def test_igd_by_hand():
    # Reference (0,0)->front 0, (3,4)->front 5. IGD = mean(0, 5) = 2.5.
    ref = np.array([[0.0, 0.0], [3.0, 4.0]])
    front = np.array([[0.0, 0.0]])
    assert math.isclose(igd(front, ref), 2.5, rel_tol=1e-12)


# --- DM (maximum spread) ---------------------------------------------------------

def test_maximum_spread_by_hand():
    # Bounding-box diagonal: span (3,4) -> sqrt(9+16) = 5.
    front = np.array([[0.0, 0.0], [3.0, 4.0]])
    assert math.isclose(maximum_spread(front), 5.0, rel_tol=1e-12)


def test_maximum_spread_single_point_is_zero():
    assert maximum_spread(np.array([[1.0, 1.0]])) == 0.0
    assert maximum_spread(np.empty((0, 2))) == 0.0


# --- spacing ---------------------------------------------------------------------

def test_spacing_by_hand():
    # Collinear points; L1 nearest-neighbour distances: 1, 1, 2. d_bar = 4/3.
    # S = sqrt( (1/(3-1)) * ((1-4/3)^2 + (1-4/3)^2 + (2-4/3)^2) )
    #   = sqrt( (1/2) * (1/9 + 1/9 + 4/9) ) = sqrt(1/3).
    front = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    assert math.isclose(spacing(front), math.sqrt(1.0 / 3.0), rel_tol=1e-12)


def test_spacing_two_points_is_nan():
    # Invariant: <3 points -> NaN (never 0, which would read as a perfect score).
    assert math.isnan(spacing(np.array([[0.0, 0.0], [1.0, 1.0]])))


def test_spacing_single_point_is_nan():
    assert math.isnan(spacing(np.array([[0.5, 0.5]])))


# --- hypervolume -----------------------------------------------------------------

def test_hv_single_point_by_hand():
    # Rectangle from (0,0) to ref (2,2): area 4.
    assert math.isclose(hypervolume(np.array([[0.0, 0.0]]), np.array([2.0, 2.0])),
                        4.0, rel_tol=1e-12)


def test_hv_two_points_by_hand():
    # Union of [0,2]x[1,2] and [1,2]x[0,2] minus overlap [1,2]x[1,2] = 2+2-1 = 3.
    front = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert math.isclose(hypervolume(front, np.array([2.0, 2.0])), 3.0, rel_tol=1e-12)


def test_hv_ignores_dominated_points():
    # (1.5,1.5) is dominated by both others and must not change the HV.
    front = np.array([[0.0, 1.0], [1.0, 0.0], [1.5, 1.5]])
    assert math.isclose(hypervolume(front, np.array([2.0, 2.0])), 3.0, rel_tol=1e-12)


def test_hv_ignores_points_worse_than_reference():
    # (3,3) does not dominate the reference (2,2); only (0,1) contributes: 2*1 = 2.
    front = np.array([[0.0, 1.0], [3.0, 3.0]])
    assert math.isclose(hypervolume(front, np.array([2.0, 2.0])), 2.0, rel_tol=1e-12)


def test_hv_empty_is_zero():
    assert hypervolume(np.empty((0, 2)), np.array([1.1, 1.1])) == 0.0


# --- degenerate single-point front: every metric stays graceful ------------------

def test_single_point_front_is_handled_gracefully():
    front = np.array([[0.3, 0.7]])
    ref = np.array([[0.0, 0.0], [1.0, 1.0]])
    assert nps(front) == 1
    assert maximum_spread(front) == 0.0
    assert math.isnan(spacing(front))           # <3 points
    assert math.isfinite(gd(front, ref))        # convergence still well-defined
    assert math.isfinite(igd(front, ref))
    assert hypervolume(front, np.array([1.1, 1.1])) >= 0.0
