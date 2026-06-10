"""Runner filters + resume (the §7 gate).

Smoke-level integration: a single filtered cell runs exactly once and logs exactly
one row; a second identical invocation resumes (skips it). Plus the co-equality
guard that MOPSO refuses to run without NSGA-II's measured parity budget. Budgets
are shrunk via ``hp_overrides`` so the tests stay fast.
"""

from __future__ import annotations

import csv
import json
import os

import pytest

from uav.experiment.runner import _log_path, _run_json_path, run_all

_EIL51 = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, "instances", "eil51.tsp")
)
requires_eil51 = pytest.mark.skipif(
    not os.path.exists(_EIL51), reason="eil51.tsp not downloaded"
)

# NSGA-II budget shrunk for a fast smoke run (pop divisible by 4 for selTournamentDCD).
_TINY = {"pop": 8, "gens": 2}


def _log_rows(results_dir):
    with open(_log_path(results_dir), newline="") as fh:
        return list(csv.DictReader(fh))


@requires_eil51
def test_single_cell_runs_once_then_resumes(tmp_path):
    rd = str(tmp_path)
    # First invocation: exactly one run, one JSON, one log row.
    s1 = run_all(instances=["eil51-k3"], algorithms=["nsga2"], seeds=[0],
                 results_dir=rd, hp_overrides=_TINY)
    assert s1 == {"completed": 1, "skipped": 0, "errors": []}

    assert os.path.exists(_run_json_path(rd, "eil51-k3", "nsga2", 0))
    rows = _log_rows(rd)
    assert len(rows) == 1
    assert (rows[0]["algorithm"], rows[0]["instance"], rows[0]["seed"]) == ("nsga2", "eil51-k3", "0")

    # Per-run JSON carries the variable-fleet metadata, parallel to the front.
    with open(_run_json_path(rd, "eil51-k3", "nsga2", 0)) as fh:
        payload = json.load(fh)
    assert len(payload["n_active_drones"]) == len(payload["front"])
    assert all(1 <= a <= 3 for a in payload["n_active_drones"])

    # Second invocation: matching hyperparameters -> skip, no new row.
    s2 = run_all(instances=["eil51-k3"], algorithms=["nsga2"], seeds=[0],
                 results_dir=rd, hp_overrides=_TINY)
    assert s2 == {"completed": 0, "skipped": 1, "errors": []}
    assert len(_log_rows(rd)) == 1


@requires_eil51
def test_mopso_refuses_without_nsga2_parity(tmp_path):
    # Co-equality guard: MOPSO's budget is NSGA-II's measured mean; with no
    # NSGA-II runs on disk the runner must refuse rather than guess a budget.
    s = run_all(instances=["eil51-k3"], algorithms=["mopso"], seeds=[0],
                results_dir=str(tmp_path), hp_overrides=_TINY)
    assert s["completed"] == 0
    assert s["errors"]
    assert not os.path.exists(_run_json_path(str(tmp_path), "eil51-k3", "mopso", 0))
