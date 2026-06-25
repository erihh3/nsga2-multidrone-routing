"""Aggregate saved fronts into metric tables for the non-eil51 instances.

The Phase-5 sweep persisted per-seed run JSONs for all four instances, but only
eil51-k3 was aggregated to a ``*_metrics.json`` (by ``scripts/eval_eil51.py``).
This script produces the three missing summaries: berlin52-k3, eil76-k3, rat99-k3.

It RE-RUNS NOTHING. It reads ``results/{instance}_{algo}_{seed}.json`` (the
fronts, routes, wall-clock and measured ``n_evals`` already on disk), rebuilds
``Run`` records, and calls the *same* ``evaluation.aggregate`` used for eil51 — so
every reported number is reproducible from disk and identical in pipeline to the
eil51 table. Figures and optimizer runs are untouched.

Usage:  .venv/bin/python scripts/eval_all.py              # all three
        .venv/bin/python scripts/eval_all.py rat99-k3     # one instance
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

from uav.evaluation.aggregate import Run, aggregate
from uav.experiment.config import SEEDS
from uav.solution import Solution

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
RESULTS = os.path.join(ROOT, "results")
INSTANCES = ("berlin52-k3", "eil76-k3", "rat99-k3")
ALGOS = ("nsga2", "mopso")

# Reporting direction per metric: True = higher is better (mirrors eval_eil51.py).
HIGHER_BETTER = {
    "nps": True, "dm": True, "hv": True,
    "spacing": False, "gd": False, "igd": False, "ct": False, "n_evals": None,
}


def _load_run(instance: str, algo: str, seed: int) -> Run:
    """Rebuild one seeded ``Run`` from its persisted JSON (reads disk only)."""
    path = os.path.join(RESULTS, f"{instance}_{algo}_{seed}.json")
    with open(path) as fh:
        data = json.load(fh)
    # front[i] (makespan, energy) is parallel to routes[i] (its K routes).
    front = [
        Solution(
            routes=tuple(tuple(int(p) for p in r) for r in routes_i),
            makespan=float(mk),
            energy=float(en),
        )
        for (mk, en), routes_i in zip(data["front"], data["routes"])
    ]
    return Run(
        seed=seed,
        front=front,
        wall_clock_s=float(data["wall_clock_s"]),
        n_evals=int(data["n_evals"]),
    )


def _load_arm(instance: str, algo: str) -> list[Run]:
    return [_load_run(instance, algo, s) for s in SEEDS]


def _mopso_iters(instance: str) -> int | None:
    """The MOPSO parity ``iters`` this instance ran at (read off any mopso run)."""
    path = os.path.join(RESULTS, f"{instance}_mopso_0.json")
    with open(path) as fh:
        return json.load(fh).get("hyperparams", {}).get("iters")


def _fmt(x: float) -> str:
    if x != x:                       # NaN
        return "  nan"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    if abs(x) >= 1:
        return f"{x:.3f}"
    return f"{x:.4f}"


def evaluate_instance(instance: str) -> None:
    runs = {algo: _load_arm(instance, algo) for algo in ALGOS}
    out = aggregate(runs)
    measured = {a: float(np.mean([r.n_evals for r in runs[a]])) for a in ALGOS}
    out["measured_evals"] = measured
    out["mopso_parity_iters"] = _mopso_iters(instance)

    with open(os.path.join(RESULTS, f"{instance}_metrics.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    print("\n" + "=" * 72)
    print(f"{instance} metric table  (median +- IQR, n={len(SEEDS)})")
    print(f"reference front: {out['reference_size']} points  "
          f"(union of both algos x all seeds, non-dominated)")
    print(f"measured n_evals/seed: NSGA-II {measured['nsga2']:.0f}  |  "
          f"MOPSO {measured['mopso']:.0f}  (parity iters={out['mopso_parity_iters']})")
    print("=" * 72)
    hdr = (f"{'metric':<10}{'dir':<5}{'NSGA-II (med+-IQR)':<26}"
           f"{'MOPSO (med+-IQR)':<26}{'MW p':>8}")
    print(hdr)
    print("-" * len(hdr))
    for key in ("nps", "spacing", "gd", "igd", "dm", "hv", "ct", "n_evals"):
        e = out["summary"][key]
        hb = HIGHER_BETTER[key]
        direction = "+" if hb else ("-" if hb is False else " ")
        n = f"{_fmt(e['nsga2']['median'])} +- {_fmt(e['nsga2']['iqr'])}"
        m = f"{_fmt(e['mopso']['median'])} +- {_fmt(e['mopso']['iqr'])}"
        p = e.get("p", float("nan"))
        print(f"{key:<10}{direction:<5}{n:<26}{m:<26}{_fmt(p):>8}")
    print("-" * len(hdr))
    print("dir: + higher-is-better, - lower-is-better. CT = wall-clock seconds.")
    print(f"saved: results/{instance}_metrics.json")


def main() -> None:
    targets = sys.argv[1:] or list(INSTANCES)
    for instance in targets:
        evaluate_instance(instance)


if __name__ == "__main__":
    main()
