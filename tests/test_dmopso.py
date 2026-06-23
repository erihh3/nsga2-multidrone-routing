"""Attempt C — discrete (swap-sequence) MOPSO.

Unit checks on the swap-sequence algebra (round-trip difference/apply, coefficient
scaling, position validity after a full update) and the shared-decoder contract for
a hybrid (permutation + cut-keys) particle, plus a reduced-budget integration run
proving the loop yields a feasible, internally non-dominated, deterministic archive
with the deterministic ``swarm*(iters+1)`` eval tally (measured-eval parity).
"""

from __future__ import annotations

import os
import random

import numpy as np
import pytest

from uav.algorithms.dmopso import (
    DiscreteMOPSO,
    apply_ss,
    clamp_ss,
    concat_ss,
    difference,
    scale_ss,
    swap_op,
)
from uav.experiment.config import Budget, Hyperparams
from uav.problem.decode import decode_discrete
from uav.problem.instance import load_instance
from uav.seeds import set_all_seeds

_EIL51 = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, "instances", "eil51.tsp")
)
requires_eil51 = pytest.mark.skipif(
    not os.path.exists(_EIL51), reason="eil51.tsp not downloaded"
)


# --- swap-sequence algebra ------------------------------------------------------

def test_swap_op_and_apply():
    perm = [0, 1, 2, 3]
    assert apply_ss(perm, [swap_op(0, 3)]) == [3, 1, 2, 0]
    # apply returns a NEW list; the input is untouched.
    assert perm == [0, 1, 2, 3]


def test_difference_round_trip():
    # difference(a, b) transforms b into a: apply_ss(b, a (-) b) == a, always.
    rng = random.Random(0)
    for _ in range(500):
        n = rng.randint(1, 30)
        a = list(range(n)); rng.shuffle(a)
        b = list(range(n)); rng.shuffle(b)
        ss = difference(a, b)
        assert apply_ss(b, ss) == a


def test_difference_identity_is_empty():
    a = [3, 1, 0, 2]
    assert difference(a, a) == []


def test_scale_ss_keeps_all_when_alpha_ge_one():
    ss = [(0, 1), (2, 3), (1, 4)]
    assert scale_ss(1.0, ss) == ss
    assert scale_ss(1.5, ss) == ss          # PSO coefficients exceed 1 -> clamp
    assert scale_ss(0.0, ss) == []
    assert scale_ss(-0.5, ss) == []


def test_scale_ss_expected_length_is_proportional():
    # Statistical: keeping each swap with prob ~alpha gives length ~ alpha*len(ss).
    set_all_seeds(0)
    ss = [swap_op(i, i + 1) for i in range(100)]
    alpha = 0.4
    lengths = [len(scale_ss(alpha, ss)) for _ in range(400)]
    assert abs(np.mean(lengths) - alpha * len(ss)) < 3.0   # ~40, generous tol


def test_concat_ss_appends():
    assert concat_ss([(0, 1)], [(2, 3), (1, 1)]) == [(0, 1), (2, 3), (1, 1)]


def test_clamp_ss_caps_length():
    ss = [(0, 1), (1, 2), (2, 3), (3, 4)]
    assert clamp_ss(ss, 2) == [(0, 1), (1, 2)]   # keeps the first ``max_len``
    assert clamp_ss(ss, 0) == []
    assert clamp_ss(ss, 10) == ss                # cap above length is a no-op


# --- decoder contract (shared phenotype) ---------------------------------------

@requires_eil51
def test_discrete_particle_decodes_to_feasible_routes():
    inst = load_instance(_EIL51, k=3)
    n, k, depot = inst.n_pois, inst.k, inst.depot
    set_all_seeds(0)
    for _ in range(50):
        perm = list(map(int, np.random.permutation(n)))
        kappa = np.random.uniform(0.0, 1.0, size=k)
        routes = decode_discrete(perm, kappa, n, k, depot)
        assert len(routes) == k
        pois = sorted(p for r in routes for p in r if p != depot)
        assert pois == list(range(1, n + 1))            # every POI exactly once
        assert all(r[0] == depot and r[-1] == depot for r in routes)
        assert all(len(r) >= 2 for r in routes)         # empty (idle) drone allowed
        counts = [len(r) - 2 for r in routes]
        assert min(counts) >= 0                         # c_k >= 0 (variable fleet)


@requires_eil51
def test_idle_drone_is_reachable():
    # The count axis must be able to park a drone (c_k == 0), as in random-key MOPSO.
    inst = load_instance(_EIL51, k=3)
    n, k, depot = inst.n_pois, inst.k, inst.depot
    set_all_seeds(0)
    seen_idle = False
    for _ in range(300):
        perm = list(map(int, np.random.permutation(n)))
        kappa = np.random.uniform(0.0, 1.0, size=k)
        counts = [len(r) - 2 for r in decode_discrete(perm, kappa, n, k, depot)]
        if 0 in counts:
            seen_idle = True
            break
    assert seen_idle


# --- position validity after a full update -------------------------------------

@requires_eil51
def test_full_update_keeps_valid_permutations():
    # A full reduced-budget run never produces an invalid order — every archive
    # member's routes visit each POI exactly once, with no repair anywhere.
    inst = load_instance(_EIL51, k=3)
    n = inst.n_pois
    res = DiscreteMOPSO(inst, Budget(), Hyperparams(swarm=20, iters=10)).run(seed=0)
    for s in res.final_front:
        pois = sorted(p for r in s.routes for p in r if p != inst.depot)
        assert pois == list(range(1, n + 1))


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
    res = DiscreteMOPSO(inst, Budget(), hp).run(seed=0)

    assert len(res.history) == hp.iters + 1
    # One eval per particle per iteration (+ the initial swarm): exact parity tally.
    assert res.n_evals == hp.swarm * (hp.iters + 1)
    assert res.wall_clock_s >= 0
    assert len(res.final_front) >= 1

    n = inst.n_pois
    for s in res.final_front:
        assert len(s.routes) == inst.k
        pois = sorted(p for r in s.routes for p in r if p != inst.depot)
        assert pois == list(range(1, n + 1))
        assert all(r[0] == inst.depot and r[-1] == inst.depot for r in s.routes)
        assert 1 <= s.n_active_drones <= inst.k
        assert np.isfinite(s.makespan) and s.makespan > 0
        assert np.isfinite(s.energy) and s.energy > 0

    assert _front_is_nondominated(res.final_front)
    assert len(res.final_front) <= hp.archive_size


@requires_eil51
def test_velocity_clamp_enables_convergence():
    # Regression for the no-convergence bug: without the discrete velocity clamp
    # the order re-randomized each step and best-energy did a flat random walk.
    # With the per-pull clamp the swarm-best energy must improve materially.
    inst = load_instance(_EIL51, k=3)
    res = DiscreteMOPSO(inst, Budget(), Hyperparams(swarm=30, iters=60)).run(seed=0)
    initial_best = res.history[0].best[1]
    final_best = min(h.best[1] for h in res.history)
    assert final_best < 0.95 * initial_best     # observed ~0.88; flat walk would be ~1.0


@requires_eil51
def test_same_seed_is_deterministic():
    inst = load_instance(_EIL51, k=3)
    hp = Hyperparams(swarm=20, iters=10)
    a = DiscreteMOPSO(inst, Budget(), hp).run(seed=7)
    b = DiscreteMOPSO(inst, Budget(), hp).run(seed=7)
    objs_a = sorted(s.objectives for s in a.final_front)
    objs_b = sorted(s.objectives for s in b.final_front)
    assert objs_a == objs_b
