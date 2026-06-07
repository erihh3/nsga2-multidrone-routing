"""Genotype -> shared phenotype.

Both decoders must emit the *identical* structure: a list of ``K`` routes, each
``[depot, p_i, ..., depot]``. After decoding, the two algorithms are
indistinguishable to the rest of the system — that is the whole point.
"""

from __future__ import annotations

import numpy as np


def decode_two_part(perm: list[int], counts: list[int], depot: int) -> list[list[int]]:
    """NSGA-II genotype. ``perm`` = permutation of POI ids; ``counts`` sums to N,
    each >= 1. Slice ``perm`` into K contiguous segments of length ``counts[k]``,
    bookend each with ``depot``.

    Phase 1. Stub.
    """
    raise NotImplementedError("Phase 1: two-part decode.")


def decode_random_key(keys: np.ndarray, n_pois: int, k: int, depot: int) -> list[list[int]]:
    """MOPSO genotype. ``keys`` length N+K: first N -> argsort -> visit order;
    last K -> split into per-drone counts.

    Split rule (must match the two-part phenotype space exactly): normalize the
    last K keys, scale by N, floor, distribute leftover by largest fractional
    remainder, then steal from the largest count to guarantee every count >= 1.

    Phase 1. Stub.
    """
    raise NotImplementedError("Phase 1: random-key decode.")
