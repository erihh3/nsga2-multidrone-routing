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
import hashlib
import json
import math
import os
import sys

import numpy as np

from uav.algorithms.nsga2 import NSGA2, _genotype_to_routes
from uav.experiment.config import SEEDS, Budget, Hyperparams
from uav.problem.decode import decode_random_key
from uav.problem.fitness import ALPHA, BETA, MASS, V_CRUISE, evaluate
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


def _eval_slots(algo: str, hp: dict) -> int:
    """Individuals that pass through a run: the initial set plus one per step.

    NSGA-II has ``pop`` initial individuals and ``pop`` offspring per generation; the
    swarm arms evaluate the initial swarm and the whole swarm every iteration.
    """
    if algo == "nsga2":
        return hp["pop"] * (hp["gens"] + 1)
    return hp["swarm"] * (hp["iters"] + 1)


def _sha256(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _write_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def budget(results_dir: str, out_dir: str) -> bool:
    """[R3, package] Evaluations per run, reuse, repair cost and the paper's configuration.

    Writes ``budget_per_run.csv`` (360 runs), ``budget_summary.csv`` (min/mean/max
    per instance and arm) and ``config_paper.json`` (hyperparameters from the runs,
    parity, UAV parameters, seeds, instance files with SHA-256, N, K and depot).
    """
    rows, hps, parity, ok = [], {algo: set() for algo in ALGORITHMS}, {}, True
    for instance in INSTANCES:
        for algo in ALGORITHMS:
            for seed in SEEDS:
                with open(_run_path(results_dir, instance, algo, seed)) as fh:
                    run = json.load(fh)
                hp, n = run["hyperparams"], run["n_evals"]
                hps[algo].add(json.dumps(hp, sort_keys=True))
                slots = _eval_slots(algo, hp)
                # Repair (_repair_counts, key clipping) and decoding never call the
                # evaluator, and neither method has a local search: CountingEvaluator
                # is reached only from run() (nsga2.py:273,293; mopso.py:224,261;
                # dmopso.py:203,259). So both costs are 0 by construction; the slot
                # checks catch extra calls in the swarm arms and bound them in NSGA-II.
                ok = ok and (n <= slots if algo == "nsga2" else n == slots)
                front = np.array(run["front"], dtype=float).reshape(-1, 2)
                rows.append({"instance": instance, "algorithm": algo, "seed": seed,
                             "n_evals": n, "reused_evals": slots - n,
                             "repair_evaluator_calls": 0, "local_search_calls": 0,
                             "wall_clock_s": run["wall_clock_s"],
                             "front_points": len(front),
                             # Same collapse as aggregate._front_objs (metrics use it).
                             "front_distinct": len(np.unique(front, axis=0))})
        with open(os.path.join(results_dir, f"{instance}_parity.json")) as fh:
            p = json.load(fh)
        parity[instance] = {"measured_nsga2_mean": p["measured_nsga2_mean"], "iters": p["iters"]}

    # One configuration per arm across all instances and seeds, else config_paper.json lies.
    ok = ok and all(len(s) == 1 for s in hps.values())
    config_hp = {algo: json.loads(min(hps[algo])) for algo in ALGORITHMS}
    ok = ok and all(v["iters"] == config_hp["mopso"]["iters"] for v in parity.values())
    swarm_budget = _eval_slots("mopso", config_hp["mopso"])

    summary = []
    for instance in INSTANCES:
        for algo in ALGORITHMS:
            arm = [r for r in rows if r["instance"] == instance and r["algorithm"] == algo]
            evals = np.array([r["n_evals"] for r in arm])
            reused = np.array([r["reused_evals"] for r in arm])
            wall = np.array([r["wall_clock_s"] for r in arm])
            summary.append({
                "instance": instance, "algorithm": algo, "runs": len(arm),
                "n_evals_min": int(evals.min()), "n_evals_mean": f"{evals.mean():.1f}",
                "n_evals_max": int(evals.max()),
                "n_evals_max_rel_dev_pct":
                    f"{100 * np.abs(evals - swarm_budget).max() / swarm_budget:.3f}",
                "reused_min": int(reused.min()), "reused_mean": f"{reused.mean():.1f}",
                "reused_max": int(reused.max()),
                "repair_evaluator_calls": sum(r["repair_evaluator_calls"] for r in arm),
                "local_search_calls": sum(r["local_search_calls"] for r in arm),
                "wall_clock_s_min": f"{wall.min():.2f}", "wall_clock_s_mean": f"{wall.mean():.2f}",
                "wall_clock_s_max": f"{wall.max():.2f}",
                "front_points": sum(r["front_points"] for r in arm),
                "front_distinct": sum(r["front_distinct"] for r in arm)})

    instances = {}
    for instance in INSTANCES:
        inst = _load_paper_instance(instance)
        rel = os.path.join("instances", f"{instance.partition('-k')[0]}.tsp")
        instances[instance] = {
            "file": rel, "sha256": _sha256(os.path.join(ROOT, rel)),
            "N": inst.n_pois, "K": inst.k,
            "depot": {"file_node": 1, "index": inst.depot,
                      "coords": [float(c) for c in inst.coords[inst.depot]]}}
    config = {
        "source": "scripts/eval_cobudget.py (POP, GENS, SWARM); swarm-arm iters from "
                  "results_cobudget/<instance>_parity.json",
        "seeds": list(SEEDS),
        "hyperparams": config_hp,
        "dmopso": "diagnostic arm only, not part of the main comparison",
        "evaluation_slots": {algo: _eval_slots(algo, config_hp[algo]) for algo in ALGORITHMS},
        "parity": parity,
        "uav": {"alpha_W_per_kg": ALPHA, "beta_W": BETA, "mass_kg": MASS,
                "v_cruise_m_per_s": V_CRUISE},
        "instances": instances,
    }

    _write_csv(os.path.join(out_dir, "budget_per_run.csv"), rows)
    _write_csv(os.path.join(out_dir, "budget_summary.csv"), summary)
    with open(os.path.join(out_dir, "config_paper.json"), "w") as fh:
        json.dump(config, fh, indent=2)
        fh.write("\n")

    for s in summary:
        print(f"  {s['instance']:<12} {s['algorithm']:<7} n_evals {s['n_evals_min']}-"
              f"{s['n_evals_max']} mean {s['n_evals_mean']:>8}  reused mean "
              f"{s['reused_mean']:>7}  distinct {s['front_distinct']}/{s['front_points']}")
    print(f"budget: {len(rows)} runs, swarm budget {swarm_budget}, checks "
          f"{'OK' if ok else 'FAILED'} -> {os.path.relpath(out_dir, ROOT)}/"
          "{budget_per_run,budget_summary}.csv, config_paper.json")
    return ok


STEPS = {"check_solutions": check_solutions, "initial_fleet": initial_fleet, "budget": budget}


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
