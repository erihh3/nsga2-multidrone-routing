"""Phase 1 — decoders.

The central done-when: a two-part chromosome and a random-key vector that *should*
decode to the same routes must produce identical partitions. Plus the phenotype
invariants (every POI once, depot-bookended) and the feasibility check.

Variable fleet (c_k >= 0): a drone may stay at the depot. An empty route decodes
to [depot, depot] (zero cost); only negative counts are rejected.
"""

from __future__ import annotations

import numpy as np
import pytest

from uav.problem.decode import decode_random_key, decode_two_part

DEPOT = 0


def _all_pois(routes):
    return sorted(p for r in routes for p in r if p != DEPOT)


# --- two-part basics ------------------------------------------------------------

def test_two_part_segments_and_bookends():
    # 6 POIs (ids 1..6), counts [2,1,3].
    routes = decode_two_part([3, 1, 5, 2, 6, 4], [2, 1, 3], DEPOT)
    assert routes == [[0, 3, 1, 0], [0, 5, 0], [0, 2, 6, 4, 0]]
    assert all(r[0] == DEPOT and r[-1] == DEPOT for r in routes)
    assert all(len(r) >= 3 for r in routes)            # depot + >=1 POI + depot
    assert _all_pois(routes) == [1, 2, 3, 4, 5, 6]


def test_two_part_zero_count_is_an_empty_route():
    # Variable fleet: a zero count is legal — that drone stays at the depot and
    # decodes to [depot, depot]. (Previously this raised.)
    routes = decode_two_part([1, 2, 3], [2, 0, 1], DEPOT)
    assert routes == [[0, 1, 2, 0], [0, 0], [0, 3, 0]]
    assert _all_pois(routes) == [1, 2, 3]


def test_two_part_rejects_infeasible_counts():
    with pytest.raises(ValueError):
        decode_two_part([1, 2, 3], [2, -1, 2], DEPOT)  # a negative count
    with pytest.raises(ValueError):
        decode_two_part([1, 2, 3], [2, 2], DEPOT)      # counts sum != N


# --- the cross-decoder equivalence (the key test) -------------------------------

def test_decoders_agree_on_matched_genotypes():
    n_pois, k = 6, 3
    # Target: visit order [3,1,5,2,6,4], counts [2,1,3] -> same routes both ways.

    # Random keys whose argsort yields that order: assign ascending key values to
    # the POIs in the desired visiting sequence. POI ids are 1..6 -> indices 0..5.
    perm = [3, 1, 5, 2, 6, 4]
    position_keys = np.empty(n_pois)
    for rank, poi_id in enumerate(perm):
        position_keys[poi_id - 1] = rank * 0.1          # strictly increasing
    # Split keys proportional to [2,1,3] out of 6 -> exact apportionment.
    split_keys = np.array([2.0, 1.0, 3.0])
    keys = np.concatenate([position_keys, split_keys])

    rk_routes = decode_random_key(keys, n_pois, k, DEPOT)
    tp_routes = decode_two_part(perm, [2, 1, 3], DEPOT)
    assert rk_routes == tp_routes


def test_decoders_agree_on_single_active_drone():
    # The energy-end extreme: all POIs on one drone, the other two empty. The
    # two-part vector (N,0,0) and the random-key split that apportions to it must
    # decode to the *identical* partition (two [depot, depot] routes included).
    n_pois, k = 6, 3
    perm = [3, 1, 5, 2, 6, 4]
    position_keys = np.empty(n_pois)
    for rank, poi_id in enumerate(perm):
        position_keys[poi_id - 1] = rank * 0.1
    split_keys = np.array([1.0, 0.0, 0.0])              # -> counts [6, 0, 0]
    keys = np.concatenate([position_keys, split_keys])

    rk_routes = decode_random_key(keys, n_pois, k, DEPOT)
    tp_routes = decode_two_part(perm, [6, 0, 0], DEPOT)
    assert rk_routes == tp_routes
    assert tp_routes == [[0, 3, 1, 5, 2, 6, 4, 0], [0, 0], [0, 0]]
    assert _all_pois(tp_routes) == [1, 2, 3, 4, 5, 6]   # every POI exactly once


# --- random-key invariants + apportionment --------------------------------------

def test_random_key_phenotype_invariants():
    rng = np.random.default_rng(0)
    n_pois, k = 50, 3
    for _ in range(200):
        keys = rng.random(n_pois + k)
        routes = decode_random_key(keys, n_pois, k, DEPOT)
        assert len(routes) == k
        assert _all_pois(routes) == list(range(1, n_pois + 1))   # each POI once
        assert all(len(r) >= 2 for r in routes)                  # >=0 POIs (>=2 nodes)
        assert all(r[0] == DEPOT and r[-1] == DEPOT for r in routes)


def test_random_key_allows_zero_counts():
    # A split heavily skewed to drone 0 now floors the others to zero — legal
    # (variable fleet), no longer repaired up to >=1.
    n_pois, k = 5, 3
    # proportions ~ [0.96, 0.02, 0.02] -> raw ~ [4.8, 0.1, 0.1].
    keys = np.concatenate([np.linspace(0, 1, n_pois), np.array([48.0, 1.0, 1.0])])
    routes = decode_random_key(keys, n_pois, k, DEPOT)
    counts = [len(r) - 2 for r in routes]
    assert min(counts) >= 0
    assert 0 in counts                                  # at least one idle drone
    assert sum(counts) == n_pois


def test_random_key_degenerate_keys_fall_back_to_uniform():
    # All-zero split keys would make proportions 0/0; the guard falls back to a
    # uniform split rather than producing NaN. Counts must still be a valid split.
    n_pois, k = 5, 3
    keys = np.concatenate([np.linspace(0, 1, n_pois), np.zeros(k)])
    routes = decode_random_key(keys, n_pois, k, DEPOT)
    counts = [len(r) - 2 for r in routes]
    assert sum(counts) == n_pois
    assert min(counts) >= 0
    assert all(np.isfinite(c) for c in counts)


def test_random_key_wrong_length_rejected():
    with pytest.raises(ValueError):
        decode_random_key(np.zeros(10), n_pois=50, k=3, depot=DEPOT)
