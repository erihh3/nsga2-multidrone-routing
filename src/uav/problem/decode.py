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
    """
    if sum(counts) != len(order):
        raise ValueError(f"counts sum {sum(counts)} != number of POIs {len(order)}")
    if min(counts) < 1:
        raise ValueError(f"every drone must visit >=1 POI; got counts={counts}")

    routes: list[list[int]] = []
    cursor = 0
    for c in counts:
        segment = order[cursor:cursor + c]
        routes.append([depot, *segment, depot])
        cursor += c
    return routes


def decode_two_part(perm: list[int], counts: list[int], depot: int) -> list[list[int]]:
    """NSGA-II genotype -> routes.

    ``perm`` is a permutation of POI ids; ``counts`` (length K) sums to N with
    each >= 1. Part 1 fixes the global visit order, Part 2 the per-drone load; the
    two are read straight through into contiguous segments.
    """
    return _partition_to_routes(list(perm), list(counts), depot)


def _counts_from_keys(weight_keys: np.ndarray, n_pois: int, k: int) -> list[int]:
    """Turn the K split-keys into per-drone counts summing to N, each >= 1.

    Largest-remainder apportionment, then a guaranteed repair: this is what makes
    the random-key phenotype space *exactly* the two-part one (every valid (order,
    counts) partition is reachable, and no decode is ever infeasible).
    """
    w = np.asarray(weight_keys, dtype=np.float64)
    total = w.sum()
    # Degenerate keys (all zero / negative) fall back to a uniform split.
    proportions = (w / total) if total > 0 else np.full(k, 1.0 / k)

    raw = proportions * n_pois
    counts = np.floor(raw).astype(int)
    # Distribute the leftover units by largest fractional remainder.
    leftover = n_pois - int(counts.sum())
    if leftover > 0:
        order = np.argsort(-(raw - counts))           # descending remainder
        for i in range(leftover):
            counts[order[i % k]] += 1

    # Enforce >= 1 by stealing from the currently largest count.
    counts = counts.tolist()
    for i in range(k):
        if counts[i] == 0:
            donor = int(np.argmax(counts))
            counts[donor] -= 1
            counts[i] += 1
    return counts


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
