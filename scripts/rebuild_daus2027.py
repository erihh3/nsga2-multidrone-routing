"""Single rebuild script for the DAUS 2027 revision (reviewer item R6).

Every number in the revised manuscript must come from a file this script writes.
Each step is one function registered in ``STEPS``; later chunks add theirs there.
Inputs are read-only: ``results_cobudget/`` is git-ignored and not recoverable.

    PYTHONPATH=src python scripts/rebuild_daus2027.py [--only STEP ...]
        [--results-dir results_cobudget] [--out docs/daus2027]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

from uav.experiment.config import SEEDS
from uav.problem.fitness import evaluate
from uav.problem.instance import load_instance

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
INSTANCES = ("eil51-k3", "berlin52-k3", "eil76-k3", "rat99-k3")
ALGORITHMS = ("nsga2", "mopso", "dmopso")


def _load_paper_instance(name: str):
    base, _, k_tag = name.partition("-k")
    return load_instance(os.path.join(ROOT, "instances", f"{base}.tsp"), k=int(k_tag))


def _run_path(results_dir: str, instance: str, algo: str, seed: int) -> str:
    return os.path.join(results_dir, f"{instance}_{algo}_{seed}.json")


def _solution_ok(obj, routes, inst, poi_ids: list[int]) -> bool:
    if len(routes) != inst.k:
        return False
    if any(len(r) < 2 or r[0] != inst.depot or r[-1] != inst.depot for r in routes):
        return False
    interior = [p for r in routes for p in r[1:-1]]
    if sorted(interior) != poi_ids:            # also rejects a depot inside a route
        return False
    f1, f2 = evaluate(routes, inst.dist)
    return math.isclose(f1, obj[0], rel_tol=1e-9) and math.isclose(f2, obj[1], rel_tol=1e-9)


def check_solutions(results_dir: str, out_dir: str) -> bool:
    """[R2, checklist] Recompute every saved solution of the 360 runs.

    Checks coverage (each POI once), depot bookends, K routes and both objectives
    against ``uav.problem.fitness.evaluate``. Writes ``solution_check.csv``.
    """
    rows = []
    for instance in INSTANCES:
        inst = _load_paper_instance(instance)
        poi_ids = [node for node in range(inst.n_pois + 1) if node != inst.depot]
        for algo in ALGORITHMS:
            runs = solutions = failures = 0
            for seed in SEEDS:
                with open(_run_path(results_dir, instance, algo, seed)) as fh:
                    run = json.load(fh)
                runs += 1
                front, routes = run["front"], run["routes"]
                if len(front) != len(routes):
                    failures += max(len(front), len(routes))
                    continue
                solutions += len(front)
                failures += sum(not _solution_ok(o, r, inst, poi_ids)
                                for o, r in zip(front, routes))
            rows.append({"instance": instance, "algorithm": algo, "runs": runs,
                         "solutions": solutions, "failures": failures})

    path = os.path.join(out_dir, "solution_check.csv")
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for r in rows:
        print(f"  {r['instance']:<12} {r['algorithm']:<7} runs={r['runs']:<3} "
              f"solutions={r['solutions']:<5} failures={r['failures']}")
    total = {key: sum(r[key] for r in rows) for key in ("runs", "solutions", "failures")}
    print(f"check_solutions: {total['runs']} runs, {total['solutions']} solutions, "
          f"{total['failures']} failures -> {os.path.relpath(path, ROOT)}")
    return total["failures"] == 0


STEPS = {"check_solutions": check_solutions}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", action="append", choices=list(STEPS), metavar="STEP",
                    help=f"run only this step (repeatable); steps: {', '.join(STEPS)}")
    ap.add_argument("--results-dir", default=os.path.join(ROOT, "results_cobudget"))
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "daus2027"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ok = True
    for name in args.only or list(STEPS):
        ok = STEPS[name](args.results_dir, args.out) and ok
    # A failed check can change a conclusion (global rule 5): make it loud.
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
