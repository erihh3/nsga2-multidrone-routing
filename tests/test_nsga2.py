"""Phase 2 — NSGA-II.

Operator-level feasibility (the two-part chromosome must never produce an
infeasible split), plus a reduced-budget integration run proving the loop yields
a feasible, internally non-dominated, deterministic front.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from uav.algorithms.nsga2 import (
    NSGA2,
    _cx_counts,
    _init_individual,
    _mut_counts,
    _random_composition,
    _repair_counts,
)
from uav.experiment.config import Budget, Hyperparams
from uav.problem.instance import load_instance

_EIL51 = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, "instances", "eil51.tsp")
)
requires_eil51 = pytest.mark.skipif(
    not os.path.exists(_EIL51), reason="eil51.tsp not downloaded"
)


# --- counts feasibility ---------------------------------------------------------

def _valid_counts(counts, n, k):
    # Variable fleet: a valid split sums to N with every count >= 0 (empty drones
    # allowed). Negative counts remain invalid.
    return len(counts) == k and sum(counts) == n and min(counts) >= 0


def test_repair_counts_random_and_degenerate():
    import random
    random.seed(0)
    n, k = 50, 3
    # Degenerate inputs that must be repaired to validity.
    for raw in ([0, 0, 50], [50, 0, 0], [-5, 100, 1], [1, 1, 1], [20, 20, 20]):
        assert _valid_counts(_repair_counts(list(raw), n, k), n, k)
    # Random noise.
    for _ in range(500):
        raw = [random.randint(-3, 60) for _ in range(k)]
        assert _valid_counts(_repair_counts(raw, n, k), n, k)


def test_cx_and_mut_counts_preserve_feasibility():
    import random
    random.seed(1)
    n, k = 50, 3
    for _ in range(500):
        c1 = _repair_counts([random.randint(1, 40) for _ in range(k)], n, k)
        c2 = _repair_counts([random.randint(1, 40) for _ in range(k)], n, k)
        d1, d2 = _cx_counts(c1, c2, n, k)
        assert _valid_counts(d1, n, k) and _valid_counts(d2, n, k)
        assert _valid_counts(_mut_counts(c1, n, k), n, k)


def test_mut_counts_consolidation_empties_a_donor(monkeypatch):
    # Force the consolidation branch (random.random() < 0.5): a non-empty donor's
    # whole count is dumped onto another drone, so the sum is preserved and a new
    # zero appears. This is what makes the 1-/2-drone (energy-end) region reachable
    # in a single mutation rather than ~N/K accepted single-POI shifts.
    import random
    monkeypatch.setattr(random, "random", lambda: 0.0)   # always consolidate
    n, k = 50, 3
    for _ in range(200):
        out = _mut_counts([10, 20, 20], n, k)            # no zeros in the input
        assert sum(out) == n
        assert min(out) >= 0
        assert 0 in out                                  # a drone was deactivated


def test_mut_counts_can_reach_a_single_active_drone():
    # Over many mutations, the operator must be able to drive a balanced split all
    # the way down to one active drone (the energy extreme). Pure single-POI drift
    # would make this practically unreachable.
    import random
    random.seed(3)
    n, k = 50, 3
    counts = _repair_counts([17, 17, 16], n, k)
    reached_single = False
    for _ in range(200):
        counts = _mut_counts(counts, n, k)
        assert sum(counts) == n and min(counts) >= 0
        if sum(1 for c in counts if c > 0) == 1:
            reached_single = True
            break
    assert reached_single


def test_random_composition_allows_zeros_and_sums_to_n():
    import random
    random.seed(4)
    n, k = 50, 3
    saw_zero = False
    for _ in range(500):
        comp = _random_composition(n, k)
        assert len(comp) == k and sum(comp) == n and min(comp) >= 0
        saw_zero = saw_zero or (0 in comp)
    assert saw_zero                                      # zeros are representable


def test_init_individual_is_a_valid_permutation():
    import random
    random.seed(2)
    n, k = 50, 3
    for _ in range(50):
        ind = _init_individual(n, k)
        assert sorted(ind) == list(range(n))      # permutation of 0..N-1
        assert _valid_counts(ind.counts, n, k)


# --- integration run ------------------------------------------------------------

def _front_is_nondominated(front):
    objs = [s.objectives for s in front]
    for a in objs:
        for b in objs:
            if b is a:
                continue
            # b strictly dominates a?  (both <= and at least one <)
            if b[0] <= a[0] and b[1] <= a[1] and (b[0] < a[0] or b[1] < a[1]):
                return False
    return True


@requires_eil51
def test_reduced_budget_run_is_feasible_and_nondominated():
    inst = load_instance(_EIL51, k=3)
    hp = Hyperparams(pop=20, gens=12)
    res = NSGA2(inst, Budget(), hp).run(seed=0)

    assert len(res.history) == hp.gens + 1          # gen 0 .. gens
    assert res.n_evals > hp.pop
    assert res.wall_clock_s >= 0
    assert len(res.final_front) >= 1

    N = inst.n_pois
    for s in res.final_front:
        assert len(s.routes) == inst.k
        pois = sorted(p for r in s.routes for p in r if p != inst.depot)
        assert pois == list(range(1, N + 1))         # every POI exactly once
        assert all(r[0] == inst.depot and r[-1] == inst.depot for r in s.routes)
        assert all(len(r) >= 2 for r in s.routes)    # >=0 POIs (empty = [depot,depot])
        assert 1 <= s.n_active_drones <= inst.k       # variable fleet, >=1 flies
        assert np.isfinite(s.makespan) and s.makespan > 0
        assert np.isfinite(s.energy) and s.energy > 0

    assert _front_is_nondominated(res.final_front)


@requires_eil51
def test_same_seed_is_deterministic():
    inst = load_instance(_EIL51, k=3)
    hp = Hyperparams(pop=20, gens=12)
    a = NSGA2(inst, Budget(), hp).run(seed=7)
    b = NSGA2(inst, Budget(), hp).run(seed=7)
    objs_a = sorted(s.objectives for s in a.final_front)
    objs_b = sorted(s.objectives for s in b.final_front)
    assert objs_a == objs_b


@requires_eil51
def test_history_tracks_per_objective_stats():
    inst = load_instance(_EIL51, k=3)
    res = NSGA2(inst, Budget(), Hyperparams(pop=20, gens=10)).run(seed=0)
    for gs in res.history:
        # best <= mean <= worst componentwise, for both objectives.
        for i in range(2):
            assert gs.best[i] <= gs.mean[i] + 1e-9
            assert gs.mean[i] <= gs.worst[i] + 1e-9
