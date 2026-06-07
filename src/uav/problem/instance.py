"""mTSPLIB instance loader + precomputed distance matrix.

Distance convention is load-bearing: TSPLIB ``EUC_2D`` defines
``d(i,j) = nint(sqrt((xi-xj)^2 + (yi-yj)^2))`` — rounded to the *nearest integer*.
Using raw floats would desync our numbers from the CPLEX-verified optima we cite.

Precompute the full O(N^2) matrix once (trivial at N<=99) so the optimizer inner
loops never call sqrt.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass
class Instance:
    name: str            # e.g. "eil51-k3"
    coords: np.ndarray   # shape (N+1, 2), index 0 = depot
    dist: np.ndarray     # shape (N+1, N+1), nint EUC_2D, precomputed
    n_pois: int          # N (depot excluded)
    k: int               # number of drones (3)
    depot: int = 0


def _euc_2d_matrix(coords: np.ndarray) -> np.ndarray:
    """Full nint-rounded Euclidean distance matrix.

    TSPLIB EUC_2D rounds to the *nearest* integer via ``nint(x) = floor(x + 0.5)``
    (not Python's banker's rounding). We compute pairwise distances vectorized,
    then apply that exact convention so our numbers line up with published optima.
    """
    diff = coords[:, None, :] - coords[None, :, :]          # (n, n, 2)
    raw = np.sqrt((diff ** 2).sum(axis=-1))                 # (n, n) float
    dist = np.floor(raw + 0.5).astype(np.float64)           # nint, kept as float
    np.fill_diagonal(dist, 0.0)                             # guard against fp noise
    return dist


def load_instance(tsp_path: str, k: int) -> Instance:
    """Parse an mTSPLIB ``.tsp`` file into coords + nint distance matrix.

    Convention: node 1 in the file (1-indexed) is the depot, stored at index 0;
    the remaining ``DIMENSION - 1`` nodes are the POIs. ``k`` is the drone count,
    carried as a parameter — it is not part of the file.

    The parser tolerates the formatting drift across mTSPLIB files: header keys
    with or without a space before the colon (``DIMENSION :`` vs ``DIMENSION:``),
    integer or float coordinates, and irregular leading/internal whitespace in the
    coordinate rows. It splits on the colon for headers and on arbitrary
    whitespace for coordinates rather than assuming fixed columns.
    """
    with open(tsp_path, "r") as fh:
        lines = fh.readlines()

    header: dict[str, str] = {}
    coord_rows: list[tuple[int, float, float]] = []
    in_coords = False

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith("NODE_COORD_SECTION"):
            in_coords = True
            continue
        if upper in ("EOF", "-1"):
            break
        if not in_coords:
            # Header row "KEY : VALUE" (spacing around ':' varies between files).
            if ":" in line:
                key, _, value = line.partition(":")
                header[key.strip().upper()] = value.strip()
            continue
        # Coordinate row: "<id> <x> <y>", any whitespace, int or float coords.
        parts = line.split()
        node_id = int(parts[0])
        x, y = float(parts[1]), float(parts[2])
        coord_rows.append((node_id, x, y))

    ewt = header.get("EDGE_WEIGHT_TYPE", "").upper()
    if ewt != "EUC_2D":
        raise ValueError(
            f"{tsp_path}: only EUC_2D is supported (got {ewt!r}); the nint "
            "rounding convention is specific to EUC_2D."
        )

    dimension = int(header["DIMENSION"])
    if len(coord_rows) != dimension:
        raise ValueError(
            f"{tsp_path}: DIMENSION={dimension} but parsed {len(coord_rows)} "
            "coordinate rows."
        )

    # Order by the file's node id so index 0 == node 1 == depot, regardless of the
    # order rows happened to appear in.
    coord_rows.sort(key=lambda r: r[0])
    coords = np.array([[x, y] for _, x, y in coord_rows], dtype=np.float64)

    name = header.get("NAME", os.path.splitext(os.path.basename(tsp_path))[0])
    return Instance(
        name=f"{name}-k{k}",
        coords=coords,
        dist=_euc_2d_matrix(coords),
        n_pois=dimension - 1,   # depot (node 1) excluded
        k=k,
        depot=0,
    )
