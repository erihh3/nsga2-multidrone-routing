"""Phase 4 — reference front + normalization.

Validates the union/non-dominated re-filter and the normalization box (including
the near-zero-range guard the energy axis motivates), on toy ``Solution`` sets
with hand-known answers.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from uav.evaluation.reference_front import (
    build_reference_front,
    normalize,
    reference_bounds,
)
from uav.solution import Solution


def _sol(mk: float, en: float) -> Solution:
    # Routes are irrelevant to reference-front logic (it reads only objectives).
    return Solution(routes=((0, 1, 0),), makespan=mk, energy=en)


# --- union + non-dominated re-filter --------------------------------------------

def test_build_reference_front_unions_and_filters():
    front_a = [_sol(1.0, 4.0), _sol(2.0, 2.0), _sol(3.0, 1.0)]
    front_b = [_sol(2.5, 2.5), _sol(1.0, 4.0)]   # (2.5,2.5) dominated; (1,4) duplicate
    ref = build_reference_front([front_a, front_b])

    objs = [s.objectives for s in ref]
    # (2.5,2.5) is dominated by (2,2); the duplicate (1,4) collapses to one.
    assert objs == [(1.0, 4.0), (2.0, 2.0), (3.0, 1.0)]   # sorted by makespan


def test_build_reference_front_single_solution():
    ref = build_reference_front([[_sol(5.0, 5.0)]])
    assert len(ref) == 1
    assert ref[0].objectives == (5.0, 5.0)


def test_reference_bounds():
    ref = build_reference_front([[_sol(1.0, 4.0), _sol(2.0, 2.0), _sol(3.0, 1.0)]])
    lo, hi = reference_bounds(ref)
    assert lo.tolist() == [1.0, 1.0]
    assert hi.tolist() == [3.0, 4.0]


def test_reference_bounds_empty_raises():
    with pytest.raises(ValueError):
        reference_bounds([])


# --- normalization ---------------------------------------------------------------

def test_normalize_maps_extremes_to_unit_box():
    ref_min = np.array([0.0, 0.0])
    ref_max = np.array([10.0, 100.0])
    objs = np.array([[0.0, 0.0], [10.0, 100.0], [5.0, 50.0]])
    out = normalize(objs, ref_min, ref_max)
    assert np.allclose(out, [[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])


def test_normalize_thin_axis_is_stretched_not_flagged():
    # The ~1% energy span is NOT degenerate: it must stretch to fill [0,1].
    ref_min = np.array([10.0, 4080.0])
    ref_max = np.array([14.0, 4130.0])
    objs = np.array([[12.0, 4105.0]])
    out = normalize(objs, ref_min, ref_max)
    assert math.isclose(out[0, 0], 0.5, rel_tol=1e-9)
    assert math.isclose(out[0, 1], 0.5, rel_tol=1e-9)


def test_normalize_guards_degenerate_axis():
    # Axis 0 has zero range -> must collapse to 0 with a warning, not divide by 0.
    ref_min = np.array([0.0, 0.0])
    ref_max = np.array([0.0, 5.0])
    objs = np.array([[0.0, 3.0]])
    with pytest.warns(RuntimeWarning):
        out = normalize(objs, ref_min, ref_max)
    assert out[0, 0] == 0.0                       # degenerate axis collapsed
    assert math.isclose(out[0, 1], 0.6, rel_tol=1e-12)
    assert np.all(np.isfinite(out))               # never inf/nan
