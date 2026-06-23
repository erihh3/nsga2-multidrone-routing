"""Phase 4 — the aggregation glue (fronts -> normalized metrics -> table).

A small two-algorithm example with a hand-known reference front, checking the
wiring: reference construction, per-run metrics on normalized objectives, and the
summary table shape (median/IQR per algo + a Mann-Whitney p-value for two algos).
"""

from __future__ import annotations

import warnings

from uav.evaluation.aggregate import Run, aggregate
from uav.solution import Solution


def _sol(mk: float, en: float) -> Solution:
    return Solution(routes=((0, 1, 0),), makespan=mk, energy=en)


def test_aggregate_two_algorithms():
    runs = {
        "nsga2": [Run(0, [_sol(1.0, 4.0), _sol(2.0, 2.0), _sol(3.0, 1.0)], 1.0, 100)],
        "mopso": [Run(0, [_sol(2.0, 4.0), _sol(4.0, 2.0)], 2.0, 120)],
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # single-seed Mann-Whitney warnings
        out = aggregate(runs)

    # (2,4) is dominated by (2,2) and (4,2) by (3,1); reference is the other three.
    assert out["reference_size"] == 3
    assert out["ref_min"] == [1.0, 1.0]
    assert out["ref_max"] == [3.0, 4.0]

    a_rows = out["per_run"]["nsga2"]
    assert len(a_rows) == 1
    assert a_rows[0]["nps"] == 3
    assert a_rows[0]["ct"] == 1.0
    assert a_rows[0]["n_evals"] == 100

    assert out["per_run"]["mopso"][0]["nps"] == 2

    # Summary carries every metric, with a p-value because two algorithms compared.
    for key in ("nps", "spacing", "gd", "igd", "dm", "hv", "ct", "n_evals"):
        assert "nsga2" in out["summary"][key] and "mopso" in out["summary"][key]
        assert "p" in out["summary"][key]
    assert out["summary"]["nps"]["nsga2"]["median"] == 3
    assert out["summary"]["ct"]["mopso"]["median"] == 2.0
