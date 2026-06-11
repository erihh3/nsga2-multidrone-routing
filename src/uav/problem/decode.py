"""Genotype -> shared phenotype.

Both decoders must emit the *identical* structure: a list of ``K`` routes, each
``[depot, p_i, ..., depot]``. After decoding, the two algorithms are
indistinguishable to the rest of the system — that is the whole point.
"""

from __future__ import annotations

import numpy as np


def _partition_to_routes(order: list[int], counts: list[int], depot: int) -> list[list[int]]:
    """Slice an ordered POI list into K depot-bookended routes.

    This is the *single* place a (visit order, per-drone counts) pair becomes the
    shared phenotype. Both decoders funnel through it, so the two algorithms are
    guaranteed structurally identical downstream — not identical by coincidence.

    Variable-fleet invariant: counts sum to N and each c_k >= 0. A zero count is
    legal — that drone stays at the depot and slices to an empty segment, i.e. the
    route ``[depot, depot]`` (zero distance, zero cost in ``evaluate``). The number
    of active UAVs is thus a decision variable, not a fixed K.
    """
    if sum(counts) != len(order):
        raise ValueError(f"counts sum {sum(counts)} != number of POIs {len(order)}")
    if min(counts) < 0:
        raise ValueError(f"counts must be >= 0; got counts={counts}")

    routes: list[list[int]] = []
    cursor = 0
    for c in counts:
        segment = order[cursor:cursor + c]      # empty when c == 0 -> [depot, depot]
        routes.append([depot, *segment, depot])
        cursor += c
    return routes


def decode_two_part(perm: list[int], counts: list[int], depot: int) -> list[list[int]]:
    """NSGA-II genotype -> routes.

    ``perm`` is a permutation of POI ids; ``counts`` (length K) sums to N with
    each >= 0. Part 1 fixes the global visit order, Part 2 the per-drone load; the
    two are read straight through into contiguous segments.
    """
    return _partition_to_routes(list(perm), list(counts), depot)


def _counts_from_keys(weight_keys: np.ndarray, n_pois: int, k: int) -> list[int]:
    """Turn the K split-keys into per-drone counts summing to N, each >= 0.

    **Stars-and-bars cut-points** — the *same* construction NSGA-II uses in
    ``nsga2._random_composition``. The first ``K-1`` keys are read as independent
    cut positions on the integer line ``[0, N]`` (``round(key * N)``); sorted, they
    split ``N`` into ``K`` contiguous gaps, which are the per-drone counts.

    Why this and not proportional (largest-remainder) apportionment: a zero count
    must be *reachable* for the fleet size to be a real decision variable. Under
    apportionment a drone goes idle only if its share falls below ``1/N`` — a tiny
    corner of the key space the swarm never reaches, so MOPSO collapsed to all-K
    active drones (measured: 0/40 idle-drone solutions). With independent cuts a
    zero is a first-class outcome (two cuts coincide, or a cut hits ``0``/``N``),
    and because each cut moves continuously with its key, PSO can drive a key to
    create and *hold* an idle drone. This also gives MOPSO the same fleet-sampling
    density as the two-part chromosome, so the comparison is fair on the fleet axis.

    The ``K``-th fleet key is unused — ``K-1`` cuts fully determine ``K`` counts —
    but the genotype keeps length ``N+K`` so the PSO velocity/position arrays and
    ``decode_random_key``'s length contract are unaffected. Reachability is total:
    any composition is hit by cut fractions = its cumulative counts / N. All-zero
    keys give cuts ``[0, 0]`` -> counts ``[0, ..., 0, N]`` (a valid 1-drone split),
    so there is no divide-by-zero to guard against.
    """
    w = np.clip(np.asarray(weight_keys, dtype=np.float64)[: k - 1], 0.0, 1.0)
    cuts = sorted(int(round(c)) for c in w * n_pois)   # K-1 positions in [0, N]
    bounds = [0, *cuts, n_pois]
    return [bounds[i + 1] - bounds[i] for i in range(k)]


def decode_random_key(keys: np.ndarray, n_pois: int, k: int, depot: int) -> list[list[int]]:
    """MOPSO genotype -> routes.

    ``keys`` has length N+K: the first N are sorted (argsort) to induce the visit
    order over POI ids; the last K are apportioned into per-drone counts. POI ids
    are every node except the depot (1..N when depot==0).
    """
    keys = np.asarray(keys, dtype=np.float64)
    if keys.shape[0] != n_pois + k:
        raise ValueError(f"expected {n_pois + k} keys, got {keys.shape[0]}")

    poi_ids = [node for node in range(n_pois + 1) if node != depot]
    order = [poi_ids[j] for j in np.argsort(keys[:n_pois], kind="stable")]
    counts = _counts_from_keys(keys[n_pois:], n_pois, k)
    return _partition_to_routes(order, counts, depot)
