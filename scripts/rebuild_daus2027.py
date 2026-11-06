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

import numpy as np

from uav.algorithms.nsga2 import NSGA2, _genotype_to_routes
from uav.experiment.config import SEEDS, Budget, Hyperparams
from uav.problem.decode import decode_random_key
from uav.problem.fitness import evaluate
from uav.problem.instance import load_instance
from uav.seeds import set_all_seeds

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


def _initial_routes(algo: str, inst, hp: Hyperparams, seed: int) -> list[list[list[int]]]:
    """Routes of every initial individual, drawn exactly as ``run(seed)`` draws them.

    Only the first RNG use is replayed (no loop). Building the NSGA-II toolbox
    consumes no randomness, so seeding before or after it gives the same population.
    """
    n, k, depot = inst.n_pois, inst.k, inst.depot
    if algo == "nsga2":
        opt = NSGA2(inst, Budget(), hp)
        set_all_seeds(seed)
        pop = opt._build_toolbox().population(n=hp.pop)
        return [_genotype_to_routes(ind, opt._poi_ids, depot) for ind in pop]
    set_all_seeds(seed)
    pos = np.random.uniform(0.0, 1.0, size=(hp.swarm, n + k))
    return [decode_random_key(p, n, k, depot) for p in pos]


def _gen0_matches(objs: list, saved: dict) -> bool:
    """Replay check: the saved gen-0 best/mean/worst must equal the replayed ones."""
    vals = np.array(objs)
    replay = {"best": vals.min(axis=0), "mean": vals.mean(axis=0), "worst": vals.max(axis=0)}
    return all(math.isclose(a, b, rel_tol=1e-9)
               for key, got in replay.items() for a, b in zip(got, saved[key]))


def _exact_fleet_probs(n: int, algo: str) -> tuple[float, float, float]:
    """P(1, 2, 3 active routes) at K=3 for one initial individual.

    NSGA-II draws each of the two cuts uniformly on {0..N} (N+1 values). MOPSO's
    round(U*N) gives the endpoints 0 and N half the weight of an interior cut, so
    its cut density is 1/N inside and 1/(2N) at each end: the two do not match.
    """
    if algo == "nsga2":
        p1, p3 = 4 / (n + 1) ** 2, (n - 1) * (n - 2) / (n + 1) ** 2
    else:
        p1, p3 = 1 / n ** 2, (n - 1) * (n - 2) / n ** 2
    return p1, 1.0 - p1 - p3, p3


def initial_fleet(results_dir: str, out_dir: str) -> bool:
    """[R1] Share of initial individuals with 1, 2 and 3 active routes, per method.

    Replays each run's initial population (seeds 0-29) and verifies it against the
    run's saved gen-0 statistics. Writes ``initial_fleet_frequencies.csv`` with the
    per-instance mean over seeds beside the exact probabilities.
    """
    rows, max_z, ok = [], 0.0, True
    for instance in INSTANCES:
        inst = _load_paper_instance(instance)
        for algo in ("nsga2", "mopso"):
            shares, matched, individuals = [], 0, 0
            for seed in SEEDS:
                with open(_run_path(results_dir, instance, algo, seed)) as fh:
                    run = json.load(fh)
                routes = _initial_routes(algo, inst, Hyperparams(**run["hyperparams"]), seed)
                matched += _gen0_matches([evaluate(r, inst.dist) for r in routes],
                                         run["history"][0])
                active = [sum(len(r) > 2 for r in ind) for ind in routes]
                shares.append([active.count(m) / len(active) for m in (1, 2, 3)])
                individuals += len(active)
            obs = np.mean(shares, axis=0)
            exact = _exact_fleet_probs(inst.n_pois, algo)
            for o, p in zip(obs, exact):
                max_z = max(max_z, abs(o - p) / math.sqrt(p * (1 - p) / individuals))
            ok = ok and matched == len(SEEDS)
            rows.append({"instance": instance, "N": inst.n_pois, "method": algo,
                         "seeds": len(SEEDS), "individuals": individuals,
                         "gen0_match": f"{matched}/{len(SEEDS)}",
                         **{f"obs_{m}": f"{100 * o:.3f}" for m, o in zip((1, 2, 3), obs)},
                         **{f"exact_{m}": f"{100 * p:.3f}" for m, p in zip((1, 2, 3), exact)}})

    path = os.path.join(out_dir, "initial_fleet_frequencies.csv")
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for r in rows:
        print(f"  {r['instance']:<12} {r['method']:<6} gen0={r['gen0_match']:<6} "
              f"obs % {r['obs_1']:>6} {r['obs_2']:>7} {r['obs_3']:>7}   "
              f"exact % {r['exact_1']:>6} {r['exact_2']:>7} {r['exact_3']:>7}")
    print(f"initial_fleet: max |z| vs exact = {max_z:.2f}; "
          f"replay {'exact' if ok else 'MISMATCH'} -> {os.path.relpath(path, ROOT)}")
    return ok


STEPS = {"check_solutions": check_solutions, "initial_fleet": initial_fleet}


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
