"""Phase 1 — instance loader.

Two layers: a real mTSPLIB file (eil51) for the documented done-when assertions,
and a synthetic file written to tmp to nail down the EUC_2D nint rounding and the
parser's tolerance of formatting drift without depending on disk data.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from uav.problem.instance import load_instance

INSTANCES_DIR = os.path.join(
    os.path.dirname(__file__), os.pardir, "instances"
)


def _instance_path(name: str) -> str:
    return os.path.abspath(os.path.join(INSTANCES_DIR, f"{name}.tsp"))


# --- real-file done-when checks -------------------------------------------------

@pytest.mark.skipif(
    not os.path.exists(_instance_path("eil51")),
    reason="eil51.tsp not downloaded",
)
def test_eil51_shape_and_matrix():
    inst = load_instance(_instance_path("eil51"), k=3)
    assert inst.name == "eil51-k3"
    assert inst.k == 3
    assert inst.depot == 0
    assert inst.n_pois == 50                     # DIMENSION 51, depot excluded
    assert inst.coords.shape == (51, 2)
    assert inst.dist.shape == (51, 51)
    # Distance matrix invariants.
    assert np.array_equal(inst.dist, inst.dist.T)      # symmetric
    assert np.all(np.diag(inst.dist) == 0)             # zero diagonal
    assert np.all(inst.dist == np.round(inst.dist))    # integer-valued (nint)
    assert np.all(inst.dist >= 0)


@pytest.mark.skipif(
    not os.path.exists(_instance_path("berlin52")),
    reason="berlin52.tsp not downloaded",
)
def test_berlin52_float_coords_and_no_space_header():
    # berlin52 uses "DIMENSION:" (no space) and float coordinates — the parser
    # must handle both. First node is the depot.
    inst = load_instance(_instance_path("berlin52"), k=3)
    assert inst.n_pois == 51
    assert inst.coords.shape == (52, 2)
    assert np.allclose(inst.coords[0], [565.0, 575.0])  # node 1 == depot


# --- synthetic file: pin down rounding + parsing --------------------------------

def test_euc_2d_nint_rounding(tmp_path):
    # depot (0,0), and a node at (1,1): raw dist sqrt(2)=1.4142 -> nint -> 1.
    # node at (0,3): dist 3 -> 3. node at (2,2): sqrt(8)=2.828 -> nint -> 3.
    content = (
        "NAME : toy\n"
        "TYPE : TSP\n"
        "DIMENSION : 4\n"
        "EDGE_WEIGHT_TYPE : EUC_2D\n"
        "NODE_COORD_SECTION\n"
        "1 0 0\n"
        "2 1 1\n"
        "3 0 3\n"
        "4 2 2\n"
        "EOF\n"
    )
    p = tmp_path / "toy.tsp"
    p.write_text(content)
    inst = load_instance(str(p), k=2)

    assert inst.n_pois == 3
    assert inst.dist[0, 1] == round(math.sqrt(2) + 0.0)  # floor(1.414+0.5)=1
    assert inst.dist[0, 1] == 1
    assert inst.dist[0, 2] == 3
    assert inst.dist[0, 3] == 3                          # floor(2.828+0.5)=3
    assert inst.dist[1, 3] == 1                          # (1,1)->(2,2): sqrt2 ->1


def test_parser_tolerates_whitespace_and_unsorted(tmp_path):
    # No space before colon, irregular leading/internal whitespace, rows out of
    # order — index 0 must still be node 1.
    content = (
        "NAME: messy\n"
        "DIMENSION: 3\n"
        "EDGE_WEIGHT_TYPE: EUC_2D\n"
        "NODE_COORD_SECTION\n"
        "  3   10   0\n"
        " 1  0   0\n"
        "2 5  0\n"
        "EOF\n"
    )
    p = tmp_path / "messy.tsp"
    p.write_text(content)
    inst = load_instance(str(p), k=2)

    assert np.allclose(inst.coords[0], [0, 0])   # node 1 -> index 0 (depot)
    assert np.allclose(inst.coords[1], [5, 0])
    assert np.allclose(inst.coords[2], [10, 0])
    assert inst.dist[0, 2] == 10


def test_rejects_non_euc2d(tmp_path):
    content = (
        "DIMENSION : 2\n"
        "EDGE_WEIGHT_TYPE : GEO\n"
        "NODE_COORD_SECTION\n"
        "1 0 0\n"
        "2 1 1\n"
        "EOF\n"
    )
    p = tmp_path / "geo.tsp"
    p.write_text(content)
    with pytest.raises(ValueError, match="EUC_2D"):
        load_instance(str(p), k=2)
