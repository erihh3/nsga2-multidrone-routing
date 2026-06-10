"""Phase 3 — MOPSO.

Unit checks on the PSO-local machinery (Pareto domination, the adaptive grid +
sparse-cell leader selection, the reflective velocity bound) plus a reduced-budget
integration run proving the loop yields a feasible, internally non-dominated,
deterministic archive with the expected eval tally.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from uav.algorithms.mopso import (
    MOPSO,
    _dominates,
    _draw_leader,
    _grid_coords,
    _leader_sampler,
    _nondominated,
    _truncate,
    _turbulence_fraction,
)
from uav.experiment.config import Budget, Hyperparams
from uav.problem.instance import load_instance
from uav.seeds import set_all_seeds

_EIL51 = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, "instances", "eil51.tsp")
)
requires_eil51 = pytest.mark.skipif(
    not os.path.exists(_EIL51), reason="eil51.tsp not downloaded"
)


# --- domination -----------------------------------------------------------------

def test_dominates_minimization():
    assert _dominates((1.0, 1.0), (2.0, 2.0))          # strictly better on both
    assert _dominates((1.0, 2.0), (1.0, 3.0))          # equal on f1, better f2
    assert not _dominates((1.0, 3.0), (2.0, 2.0))      # trade-off: neither wins
    assert not _dominates((1.0, 1.0), (1.0, 1.0))      # identical: no domination


def test_nondominated_filters_and_dedups():
    mk = lambda o: (np.zeros(3), o, None)              # noqa: E731 (position unused here)
    members = [mk((1.0, 3.0)), mk((2.0, 2.0)), mk((3.0, 1.0)), mk((2.5, 2.5)),
               mk((1.0, 3.0))]
    front = _nondominated(members)
    objs = sorted(m[1] for m in front)
    # (2.5, 2.5) is dominated by (2,2); the duplicate (1,3) is collapsed.
    assert objs == [(1.0, 3.0), (2.0, 2.0), (3.0, 1.0)]


# --- adaptive grid + leader selection ------------------------------------------

def test_grid_coords_in_range_and_degenerate_axis():
    objs = np.array([[10.0, 4000.0], [12.0, 4000.0], [14.0, 4000.0]])
    coords = _grid_coords(objs, divisions=30)
    assert coords.shape == objs.shape
    assert coords.min() >= 0 and coords.max() <= 29
    # Energy axis is degenerate (constant) -> all collapse to cell 0 on that axis.
    assert np.all(coords[:, 1] == 0)
    # Makespan axis spreads across distinct cells.
    assert len(set(coords[:, 0].tolist())) == 3


@requires_eil51
def test_leader_selection_returns_archive_member():
    set_all_seeds(0)
    archive = [
        (np.random.random(5), (10.0 + i, 4000.0 - i), None) for i in range(6)
    ]
    groups, probs = _leader_sampler(archive, divisions=10)
    assert abs(float(probs.sum()) - 1.0) < 1e-9
    for _ in range(50):
        leader = _draw_leader(archive, groups, probs)
        assert any(leader is m for m in archive)       # identity (positions are arrays)


def test_turbulence_fraction_respects_floor():
    iters, rate, floor = 500, 0.5, 0.1
    # Early: above the floor and near the initial rate.
    assert _turbulence_fraction(1, iters, rate, floor) > floor
    assert _turbulence_fraction(1, iters, rate, floor) <= rate
    # At the end: pinned at the floor, never 0 (which would kill late diversity).
    assert _turbulence_fraction(iters, iters, rate, floor) == floor
    # Monotone non-increasing, never below the floor.
    prev = 1.0
    for t in range(0, iters + 1):
        f = _turbulence_fraction(t, iters, rate, floor)
        assert f >= floor
        assert f <= prev + 1e-12
        prev = f


def test_truncate_respects_capacity():
    set_all_seeds(1)
    archive = [(np.random.random(3), (float(i), float(100 - i)), None)
               for i in range(50)]
    out = _truncate(list(archive), max_size=10, divisions=10)
    assert len(out) == 10


# --- swarm feasibility ----------------------------------------------------------

@requires_eil51
def test_decoded_swarm_is_feasible():
    inst = load_instance(_EIL51, k=3)
    set_all_seeds(0)
    from uav.problem.decode import decode_random_key
    n, k, depot = inst.n_pois, inst.k, inst.depot
    for _ in range(50):
        keys = np.random.uniform(0.0, 1.0, size=n + k)
        assert keys.min() >= 0.0 and keys.max() <= 1.0
        routes = decode_random_key(keys, n, k, depot)
        assert len(routes) == k
        pois = sorted(p for r in routes for p in r if p != depot)
        assert pois == list(range(1, n + 1))           # every POI exactly once
        assert all(r[0] == depot and r[-1] == depot for r in routes)
        assert all(len(r) >= 3 for r in routes)        # >=1 POI per drone


# --- integration run ------------------------------------------------------------

def _front_is_nondominated(front):
    objs = [s.objectives for s in front]
    for a in objs:
        for b in objs:
            if b is a:
                continue
            if b[0] <= a[0] and b[1] <= a[1] and (b[0] < a[0] or b[1] < a[1]):
                return False
    return True


@requires_eil51
def test_reduced_budget_run_is_feasible_and_nondominated():
    inst = load_instance(_EIL51, k=3)
    hp = Hyperparams(swarm=20, iters=10)
    res = MOPSO(inst, Budget(), hp).run(seed=0)

    assert len(res.history) == hp.iters + 1            # iter 0 .. iters
    # PSO re-evaluates every particle every iteration (+ the initial swarm).
    assert res.n_evals == hp.swarm * (hp.iters + 1)
    assert res.wall_clock_s >= 0
    assert len(res.final_front) >= 1

    n = inst.n_pois
    for s in res.final_front:
        assert len(s.routes) == inst.k
        pois = sorted(p for r in s.routes for p in r if p != inst.depot)
        assert pois == list(range(1, n + 1))
        assert all(r[0] == inst.depot and r[-1] == inst.depot for r in s.routes)
        assert all(len(r) >= 3 for r in s.routes)
        assert np.isfinite(s.makespan) and s.makespan > 0
        assert np.isfinite(s.energy) and s.energy > 0

    assert _front_is_nondominated(res.final_front)
    assert len(res.final_front) <= hp.archive_size


@requires_eil51
def test_same_seed_is_deterministic():
    inst = load_instance(_EIL51, k=3)
    hp = Hyperparams(swarm=20, iters=10)
    a = MOPSO(inst, Budget(), hp).run(seed=7)
    b = MOPSO(inst, Budget(), hp).run(seed=7)
    objs_a = sorted(s.objectives for s in a.final_front)
    objs_b = sorted(s.objectives for s in b.final_front)
    assert objs_a == objs_b


@requires_eil51
def test_history_tracks_per_objective_stats():
    inst = load_instance(_EIL51, k=3)
    res = MOPSO(inst, Budget(), Hyperparams(swarm=20, iters=10)).run(seed=0)
    for gs in res.history:
        for i in range(2):
            assert gs.best[i] <= gs.mean[i] + 1e-9
            assert gs.mean[i] <= gs.worst[i] + 1e-9
