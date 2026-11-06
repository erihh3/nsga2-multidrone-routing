"""Single rebuild script for the DAUS 2027 revision (reviewer item R6).

Every number in the revised manuscript must come from a file this script writes.
Each step is one function registered in ``STEPS``; later chunks add theirs there.
Inputs are read-only: ``results_cobudget/`` is git-ignored and not recoverable.

    PYTHONPATH=src python scripts/rebuild_daus2027.py [--only STEP ...]
        [--results-dir results_cobudget] [--out docs/daus2027]
        [--manuscript-dir ../peer_review/daus2027/generated]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys

import numpy as np
from scipy.stats import mannwhitneyu

from uav.algorithms.nsga2 import NSGA2, _genotype_to_routes
from uav.evaluation import metrics as M
from uav.evaluation.aggregate import Run, _front_objs, aggregate
from uav.evaluation.reference_front import build_reference_front, normalize, reference_bounds
# Renamed: fitness.ALPHA (the UAV power coefficient) is imported below under the same name.
from uav.evaluation.stats import ALPHA as TEST_ALPHA, holm, mann_whitney, median_iqr
from uav.experiment.config import SEEDS, Budget, Hyperparams
from uav.problem.decode import decode_random_key
from uav.problem.fitness import ALPHA, BETA, MASS, V_CRUISE, evaluate
from uav.problem.instance import load_instance
from uav.seeds import set_all_seeds
from uav.solution import Solution

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


# Fixed in prompt 09_C1 before any audited HV was computed, so the reference point is
# not chosen after seeing which one looks best. The others are the sensitivity check.
HV_FACTOR = 1.1
HV_FACTORS = (1.05, 1.1, 1.2, 1.5)
OLD_REF = np.array([1.1, 1.1])     # the paper's reference point (aggregate.py:98)
MAIN = ("nsga2", "mopso")
OBJECTIVES = ("f1_makespan_s", "f2_energy_J")


def _load_runs(results_dir: str, instance: str, algo: str) -> list[Run]:
    """Every seed's saved run as the ``Run`` record ``aggregate`` consumes (one read per JSON)."""
    runs = []
    for seed in SEEDS:
        with open(_run_path(results_dir, instance, algo, seed)) as fh:
            run = json.load(fh)
        front = [Solution(routes=tuple(tuple(int(p) for p in r) for r in routes),
                          makespan=float(f1), energy=float(f2))
                 for (f1, f2), routes in zip(run["front"], run["routes"])]
        runs.append(Run(seed=seed, front=front, wall_clock_s=run["wall_clock_s"],
                        n_evals=run["n_evals"]))
    return runs


def _load_fronts(results_dir: str, instance: str, algo: str) -> list[list[Solution]]:
    """The saved final front of every seed, as ``Solution``s (the type metrics consume)."""
    return [r.front for r in _load_runs(results_dir, instance, algo)]


def _a12(x, y) -> float:
    """Vargha-Delaney A12 = P(X > Y) + 0.5 P(X = Y); 0.5 is no effect, 1 is X always larger."""
    x, y = np.asarray(x, dtype=float)[:, None], np.asarray(y, dtype=float)[None, :]
    return float(((x > y).sum() + 0.5 * (x == y).sum()) / (x.size * y.size))


def _hv_frac(front_norm: np.ndarray, r: np.ndarray) -> float:
    # The reference front holds the pooled minima (normalized 0), so this lies in [0, 1].
    return M.hypervolume(front_norm, r) / float(r[0] * r[1])


def _inside(front_norm: np.ndarray, r: np.ndarray) -> np.ndarray:
    # Same strict test metrics.hypervolume applies before it sums anything.
    return (front_norm[:, 0] < r[0]) & (front_norm[:, 1] < r[1])


def _no_contribution(front_norm: np.ndarray, r: np.ndarray) -> int:
    """Points whose removal leaves the run's HV unchanged (zero exclusive contribution)."""
    total = M.hypervolume(front_norm, r)
    return sum(M.hypervolume(np.delete(front_norm, i, axis=0), r) == total
               for i in range(len(front_norm)))


def hv_audit(results_dir: str, out_dir: str) -> bool:
    """[R4] The reviewer's five-step HV audit, one common reference point, sensitivity.

    Keeps the paper's min-max transformation (so IGD, GD, SM and DM are untouched) and
    moves only the reference point: ``r = HV_FACTOR x`` the per-objective worst
    normalized value pooled over NSGA-II and MOPSO, all seeds, common to both. HV is
    the fraction of the box ``[0, r1] x [0, r2]``. Writes ``hv_bounds.csv``,
    ``hv_points.csv``, ``hv_per_run.csv`` (the only HV carried forward) and
    ``hv_sensitivity.csv``. Fails on a broken check, or if the NSGA-II vs MOPSO order
    flips under any reference point (the prompt's stop rule).
    """
    bounds_rows, point_rows, per_run_rows, sens_rows = [], [], [], []
    failed, flips = [], []
    for instance in INSTANCES:
        fronts = {algo: _load_fronts(results_dir, instance, algo) for algo in ALGORITHMS}
        if any(len(f) != len(SEEDS) for f in fronts.values()):
            failed.append(f"{instance}: 30 fronts per arm")

        # Steps 1-2. The paper's transformation: min-max over the NSGA-II + MOPSO
        # reference front. The old three-arm table pooled DMOPSO too; an identical
        # front means identical bounds, so both old tables used one transformation.
        ref = build_reference_front(fronts["nsga2"] + fronts["mopso"])
        ref3 = build_reference_front([f for algo in ALGORITHMS for f in fronts[algo]])
        if [s.objectives for s in ref] != [s.objectives for s in ref3]:
            failed.append(f"{instance}: reference front unchanged by DMOPSO")
        lo, hi = reference_bounds(ref)
        raw = {algo: [_front_objs(f) for f in fronts[algo]] for algo in ALGORITHMS}
        norm = {algo: [normalize(o, lo, hi) for o in raw[algo]] for algo in ALGORITHMS}
        pooled = {algo: np.vstack(norm[algo]) for algo in ALGORITHMS}
        everything = np.vstack(list(pooled.values()))

        # Step 3. One (lo, hi) for every arm and no clipping: values above 1 survive.
        if not all(np.array_equal(n, (o - lo) / (hi - lo))
                   for algo in ALGORITHMS for o, n in zip(raw[algo], norm[algo])):
            failed.append(f"{instance}: unclipped, same transformation for every arm")
        if not everything.max() > 1:
            failed.append(f"{instance}: some normalized value above 1")
        if everything.min() < 0:
            failed.append(f"{instance}: normalized values >= 0")

        # The audit must start from the paper's numbers: same bounds, same per-run HV.
        with open(os.path.join(results_dir, f"{instance}_metrics.json")) as fh:
            old = json.load(fh)
        if not (np.array_equal(old["ref_min"], lo) and np.array_equal(old["ref_max"], hi)):
            failed.append(f"{instance}: old bounds reproduced")
        old_hv = {algo: [M.hypervolume(n, OLD_REF) for n in norm[algo]] for algo in ALGORITHMS}
        for algo in MAIN:
            saved = {row["seed"]: row["hv"] for row in old["per_run"][algo]}
            if not all(math.isclose(h, saved[s], rel_tol=0.0, abs_tol=1e-12)
                       for s, h in zip(SEEDS, old_hv[algo])):
                failed.append(f"{instance}: old HV reproduced ({algo})")

        for j, objective in enumerate(OBJECTIVES):
            for algo in ALGORITHMS:
                r_vals = np.concatenate([o[:, j] for o in raw[algo]])
                n_vals = pooled[algo][:, j]
                bounds_rows.append({
                    "instance": instance, "objective": objective, "method": algo,
                    "lower_bound": float(lo[j]), "upper_bound": float(hi[j]),
                    "best_raw": float(r_vals.min()), "worst_raw": float(r_vals.max()),
                    "best_norm": f"{n_vals.min():.4f}", "worst_norm": f"{n_vals.max():.4f}",
                    "n_values": len(n_vals), "above_1": int((n_vals > 1).sum()),
                    "above_1_1": int((n_vals > 1.1).sum())})

        # Step 5. Reference points. DMOPSO stays out of the main box: its points reach
        # further (rat99 f1), which would move the reference of the main comparison.
        worst2 = np.vstack([pooled["nsga2"], pooled["mopso"]]).max(axis=0)
        worst3 = everything.max(axis=0)
        main_r = HV_FACTOR * worst2
        refs = [("nsga2+mopso", f, f * worst2, MAIN) for f in HV_FACTORS]
        refs += [("nsga2+mopso+dmopso", HV_FACTOR, HV_FACTOR * worst3, ALGORITHMS),
                 ("old_1.1x1.1", "", OLD_REF, MAIN)]
        a12_main = []
        for box, factor, r, arms in refs:
            hv = {algo: np.array([_hv_frac(n, r) for n in norm[algo]]) for algo in arms}
            for algo in arms:
                med, iqr = median_iqr(hv[algo])
                sens_rows.append({
                    "instance": instance, "box": box, "factor": factor,
                    "r1": f"{r[0]:.4f}", "r2": f"{r[1]:.4f}", "method": algo,
                    "runs": len(hv[algo]), "hv_median": f"{med:.4f}", "hv_iqr": f"{iqr:.4f}",
                    "hv_zero_runs": int((hv[algo] == 0).sum()),
                    "a12_nsga2_vs": "" if algo == "nsga2" else f"{_a12(hv['nsga2'], hv[algo]):.3f}"})
            a12 = _a12(hv["nsga2"], hv["mopso"])
            if box == "nsga2+mopso":
                a12_main.append(a12)
            if not (np.median(hv["nsga2"]) > np.median(hv["mopso"]) and a12 > 0.5):
                flips.append(f"{instance} {box} {factor}")

        # Step 4 at the old and the audited reference, plus each arm's reference-front share.
        hv_main = {algo: [_hv_frac(n, main_r) for n in norm[algo]] for algo in ALGORITHMS}
        ref_set = {s.objectives for s in ref}
        for algo in ALGORITHMS:
            pooled_raw = {tuple(p) for o in raw[algo] for p in o.tolist()}
            point_rows.append({
                "instance": instance, "method": algo, "runs": len(norm[algo]),
                "distinct_points": sum(len(n) for n in norm[algo]),
                "pooled_distinct": len(pooled_raw),
                "outside_old_box": sum(int((~_inside(n, OLD_REF)).sum()) for n in norm[algo]),
                "no_contrib_old": sum(_no_contribution(n, OLD_REF) for n in norm[algo]),
                "outside_new_box": sum(int((~_inside(n, main_r)).sum()) for n in norm[algo]),
                "no_contrib_new": sum(_no_contribution(n, main_r) for n in norm[algo]),
                "old_hv_zero_runs": sum(h == 0 for h in old_hv[algo]),
                "new_hv_zero_runs": sum(h == 0 for h in hv_main[algo]),
                "ref_front_size": len(ref),
                "ref_share_pct": f"{100 * len(ref_set & pooled_raw) / len(ref):.1f}"})
        for algo in MAIN:
            for seed, n, h in zip(SEEDS, norm[algo], hv_main[algo]):
                per_run_rows.append({"instance": instance, "algorithm": algo, "seed": seed,
                                     "front_distinct": len(n), "ref_f1": float(main_r[0]),
                                     "ref_f2": float(main_r[1]), "hv": h})

        print(f"  {instance:<12} worst ({worst2[0]:.2f}, {worst2[1]:.2f}) "
              f"r ({main_r[0]:.2f}, {main_r[1]:.2f})  HV median N "
              f"{np.median(hv_main['nsga2']):.4f} M {np.median(hv_main['mopso']):.4f}  "
              f"A12 {min(a12_main):.3f}-{max(a12_main):.3f}  old HV=0 N "
              f"{point_rows[-3]['old_hv_zero_runs']}/30 M {point_rows[-2]['old_hv_zero_runs']}/30")

    for name, rows in (("hv_bounds", bounds_rows), ("hv_points", point_rows),
                       ("hv_per_run", per_run_rows), ("hv_sensitivity", sens_rows)):
        _write_csv(os.path.join(out_dir, f"{name}.csv"), rows)

    for f in failed:
        print(f"  CHECK FAILED: {f}")
    if flips:
        print(f"  ORDER FLIP (stop rule, report to the user): {'; '.join(flips)}")
    print(f"hv_audit: {len(per_run_rows)} runs, checks {'FAILED' if failed else 'OK'}, "
          f"order {'FLIPPED' if flips else 'held'} under {len(refs)} references per instance "
          f"-> {os.path.relpath(out_dir, ROOT)}/hv_{{bounds,points,per_run,sensitivity}}.csv")
    return not failed and not flips


# The paper's Holm family, fixed before any audited p-value was seen: the 7 metrics of
# its Table III on the 4 instances (28 tests). n_evals is a budget check, not an outcome.
FAMILY = ("nps", "spacing", "gd", "igd", "dm", "hv", "ct")
# Fixed before any CI was computed and written into hv_effects.csv, so the CI can be redrawn.
BOOTSTRAP_SEED = 2027
BOOTSTRAP_RESAMPLES = 10_000
LABEL = {"nsga2": "NSGA-II", "mopso": "MOPSO", "dmopso": "DMOPSO"}


def _read_csv(path: str) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _audited_hv(out_dir: str) -> dict[tuple[str, str], dict[int, float]]:
    """``{(instance, algorithm): {seed: hv}}`` from ``hv_per_run.csv``, the only HV carried forward."""
    hv: dict[tuple[str, str], dict[int, float]] = {}
    for r in _read_csv(os.path.join(out_dir, "hv_per_run.csv")):
        hv.setdefault((r["instance"], r["algorithm"]), {})[int(r["seed"])] = float(r["hv"])
    return hv


def stats(results_dir: str, out_dir: str) -> bool:
    """[R6, R5] The 28 tests with Holm, the HV effect size and CI, fleet, NPS, best energy.

    Per instance, ``aggregate`` recomputes the paper's metrics exactly as before (same
    reference front, normalization and code); only HV is replaced by the audited value
    from ``hv_per_run.csv`` (run ``hv_audit`` first). Holm runs once over all 28 raw
    p-values. Writes ``stats_28tests.csv``, ``hv_effects.csv``, ``fleet_coverage.csv``,
    ``nps_by_instance.csv`` and ``best_energy.csv``. Fails if a non-HV raw p-value or U
    differs from the saved ``<instance>_metrics.json`` (the pipeline must be unchanged).
    """
    audited = _audited_hv(out_dir)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    tests, effects, fleet, nps_rows, energy = [], [], [], [], []
    failed, mismatches = [], []
    sizes_all = {algo: [] for algo in ALGORITHMS}
    nps_all = {algo: [] for algo in MAIN}
    for instance in INSTANCES:
        runs = {algo: _load_runs(results_dir, instance, algo) for algo in ALGORITHMS}
        per_run = aggregate({algo: runs[algo] for algo in MAIN})["per_run"]
        for algo in MAIN:
            hv = audited.get((instance, algo), {})
            if sorted(hv) != list(SEEDS) or [row["seed"] for row in per_run[algo]] != list(SEEDS):
                failed.append(f"{instance}: audited HV for seeds 0-29 ({algo})")
                continue
            for row in per_run[algo]:
                row["hv"] = hv[row["seed"]]

        with open(os.path.join(results_dir, f"{instance}_metrics.json")) as fh:
            saved = json.load(fh)["summary"]
        for metric in FAMILY:
            x = np.array([row[metric] for row in per_run["nsga2"]], dtype=float)
            y = np.array([row[metric] for row in per_run["mopso"]], dtype=float)
            # NSGA-II first, so U counts NSGA-II wins (+ half the ties).
            u, p = mann_whitney(x, y)
            x, y = x[~np.isnan(x)], y[~np.isnan(y)]
            # Which branch SciPy's "auto" took, checked rather than assumed.
            p_asym = float(mannwhitneyu(x, y, alternative="two-sided", method="asymptotic").pvalue)
            if metric != "hv" and not (math.isclose(p, saved[metric]["p"], rel_tol=1e-12)
                                       and math.isclose(u, saved[metric]["U"], rel_tol=1e-12)):
                mismatches.append(f"{instance} {metric}")
            (mx, ix), (my, iy) = median_iqr(x), median_iqr(y)
            tests.append({"instance": instance, "metric": metric,
                          "n_nsga2": x.size, "n_mopso": y.size,
                          "median_nsga2": mx, "iqr_nsga2": ix, "median_mopso": my, "iqr_mopso": iy,
                          "U_nsga2": u, "p_raw": p, "p_holm": None, "reject_holm": None,
                          "method": "asymptotic" if p == p_asym else "exact"})
            if metric != "hv":
                continue
            a12 = u / (x.size * y.size)
            if not math.isclose(a12, _a12(x, y), rel_tol=0.0, abs_tol=1e-12):
                failed.append(f"{instance}: A12 = U/(n1 n2)")
            if not (mx > my and a12 > 0.5):
                failed.append(f"{instance}: HV order NSGA-II > MOPSO")
            # Each arm resampled on its own: the seeds create no pairing.
            bx = np.median(x[rng.integers(0, x.size, size=(BOOTSTRAP_RESAMPLES, x.size))], axis=1)
            by = np.median(y[rng.integers(0, y.size, size=(BOOTSTRAP_RESAMPLES, y.size))], axis=1)
            ci_low, ci_high = np.percentile(bx - by, [2.5, 97.5])
            effects.append({"instance": instance, "n_nsga2": x.size, "n_mopso": y.size,
                            "hv_median_nsga2": mx, "hv_median_mopso": my, "median_diff": mx - my,
                            "U_nsga2": u, "A12": a12, "ci_low": float(ci_low),
                            "ci_high": float(ci_high), "ci_level": 0.95,
                            "resamples": BOOTSTRAP_RESAMPLES, "bootstrap_seed": BOOTSTRAP_SEED,
                            "ci_method": "percentile bootstrap of median(NSGA-II) - median(MOPSO); "
                                         "each arm resampled independently with replacement"})

        for algo in ALGORITHMS:
            # A drone is active iff its route visits a POI (len > 2): Solution.n_active_drones.
            sizes = [{s.n_active_drones for s in r.front} for r in runs[algo]]
            sizes_all[algo] += sizes
            fleet.append({"instance": instance, "algorithm": algo, "runs": len(sizes),
                          **{f"reach_{m}": sum(m in z for z in sizes) for m in (1, 2, 3)},
                          "only_3": sum(z == {3} for z in sizes)})
            best = np.array([min(s.energy for s in r.front) for r in runs[algo]])
            med, iqr = median_iqr(best)
            energy.append({"instance": instance, "algorithm": algo, "runs": len(best),
                           "best_f2_median_J": med, "best_f2_iqr_J": iqr})
        for algo in MAIN:
            nps = np.array([row["nps"] for row in per_run[algo]], dtype=int)
            nps_all[algo].append(nps)
            med, iqr = median_iqr(nps)
            nps_rows.append({"instance": instance, "algorithm": algo, "runs": nps.size,
                             "nps_median": med, "nps_iqr": iqr,
                             "nps_min": int(nps.min()), "nps_max": int(nps.max()),
                             "runs_1_point": int((nps == 1).sum()),
                             "runs_lt_3_points": int((nps < 3).sum())})

    # Totals over the four instances, for counts such as "119 of 120 runs".
    for algo in ALGORITHMS:
        z = sizes_all[algo]
        fleet.append({"instance": "all", "algorithm": algo, "runs": len(z),
                      **{f"reach_{m}": sum(m in s for s in z) for m in (1, 2, 3)},
                      "only_3": sum(s == {3} for s in z)})
    for algo in MAIN:
        nps = np.concatenate(nps_all[algo])
        # No pooled median: the instances differ in size, so only counts are summed.
        nps_rows.append({"instance": "all", "algorithm": algo, "runs": nps.size,
                         "nps_median": "", "nps_iqr": "",
                         "nps_min": int(nps.min()), "nps_max": int(nps.max()),
                         "runs_1_point": int((nps == 1).sum()),
                         "runs_lt_3_points": int((nps < 3).sum())})

    p_raw = [t["p_raw"] for t in tests]
    if len(p_raw) != len(INSTANCES) * len(FAMILY) or not all(math.isfinite(p) for p in p_raw):
        failed.append("28 finite raw p-values")
    else:
        reject, adjusted = holm(p_raw, TEST_ALPHA)
        for t, rej, adj in zip(tests, reject, adjusted):
            t["p_holm"], t["reject_holm"] = adj, rej
            t["method"] = (f"Mann-Whitney U two-sided, SciPy auto -> {t['method']}; "
                           f"Holm over m={len(tests)}, alpha={TEST_ALPHA}")

    for name, rows in (("stats_28tests", tests), ("hv_effects", effects),
                       ("fleet_coverage", fleet), ("nps_by_instance", nps_rows),
                       ("best_energy", energy)):
        _write_csv(os.path.join(out_dir, f"{name}.csv"), rows)

    for e in effects:
        print(f"  {e['instance']:<12} HV A12 {e['A12']:.3f}  median diff {e['median_diff']:.4f} "
              f"95% CI [{e['ci_low']:.4f}, {e['ci_high']:.4f}]")
    for f in fleet[-len(ALGORITHMS):]:
        print(f"  fleet all {f['algorithm']:<7} runs {f['runs']}  reach 1/2/3 "
              f"{f['reach_1']}/{f['reach_2']}/{f['reach_3']}  only 3: {f['only_3']}")
    print(f"  Holm m={len(tests)}: {sum(bool(t['reject_holm']) for t in tests)} rejected; "
          f"methods {sorted({t['method'].split('-> ')[1].split(';')[0] for t in tests})}")
    for f in failed:
        print(f"  CHECK FAILED: {f}")
    if mismatches:
        print(f"  STOP: non-HV raw p-value or U differs from the saved one: {'; '.join(mismatches)}")
    print(f"stats: non-HV p-values {'MISMATCH' if mismatches else 'match the saved ones'}, "
          f"checks {'FAILED' if failed else 'OK'} -> {os.path.relpath(out_dir, ROOT)}/"
          "{stats_28tests,hv_effects,fleet_coverage,nps_by_instance,best_energy}.csv")
    return not failed and not mismatches


def _sig3(x: float) -> str:
    # Three significant figures with trailing zeros kept (0.300, 4.50), as in Table III.
    return f"{x:#.3g}".rstrip(".")


# Table 2 columns: metric, header, whether higher is better, cell format.
TABLE2 = (("hv", r"HV\,$\uparrow$", True, lambda v: f"{v:.3f}"),
          ("igd", r"IGD\,$\downarrow$", False, _sig3),
          ("ct", r"CT\,(s)\,$\downarrow$", False, lambda v: f"{v:.2f}"))
THIN, TO = r"\,", r"$\to$"
GENERATED = "% Generated by scripts/rebuild_daus2027.py (step tables); do not edit by hand."


def _table2(out_dir: str) -> tuple[list[dict], list[str], list[str]]:
    tests = {(r["instance"], r["metric"]): r for r in _read_csv(os.path.join(out_dir, "stats_28tests.csv"))}
    fleet = {(r["instance"], r["algorithm"]): r for r in _read_csv(os.path.join(out_dir, "fleet_coverage.csv"))}
    rows, failed = [], []
    tex = [GENERATED, r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
           "Instance & Alg. & " + " & ".join(h for _, h, _, _ in TABLE2) + r" & Fleet 1/2/3 \\",
           r"\midrule"]
    for i, instance in enumerate(INSTANCES):
        csv_rows = {algo: {"instance": instance, "algorithm": LABEL[algo]} for algo in MAIN}
        cells = {algo: [] for algo in MAIN}
        for metric, _, higher, fmt in TABLE2:
            t = tests[(instance, metric)]
            med = {algo: float(t[f"median_{algo}"]) for algo in MAIN}
            iqr = {algo: float(t[f"iqr_{algo}"]) for algo in MAIN}
            better = max(MAIN, key=lambda a: med[a] if higher else -med[a])
            tie = med["nsga2"] == med["mopso"]
            significant = t["reject_holm"] == "True"
            if significant and tie:
                failed.append(f"{instance} {metric}: significant with equal medians")
            # The marker sits on the MOPSO row: its outcome against NSGA-II after Holm.
            marker = ("+" if better == "mopso" else "-") if significant and not tie else "≈"
            for algo in MAIN:
                m, q = fmt(med[algo]), fmt(iqr[algo])
                cell = rf"\textbf{{{m}}}" if algo == better and not tie else m
                if algo == "mopso":
                    cell += {"+": "$^{+}$", "-": "$^{-}$", "≈": r"$^{\approx}$"}[marker]
                cells[algo].append(f"{cell} [{q}]")
                csv_rows[algo].update({f"{metric}_median": m, f"{metric}_iqr": q,
                                       f"{metric}_better_median": algo == better and not tie,
                                       f"{metric}_marker": marker if algo == "mopso" else "",
                                       f"{metric}_p_holm": t["p_holm"]})
        for algo in MAIN:
            f = fleet[(instance, algo)]
            reach = "/".join(f[f"reach_{m}"] for m in (1, 2, 3))
            csv_rows[algo].update({"fleet_1_2_3": reach, "runs": f["runs"]})
            cells[algo].append(reach)
            rows.append(csv_rows[algo])
        if i:
            tex.append(r"\midrule")
        tex.append(rf"\multirow{{2}}{{*}}{{{instance}}}")
        tex += [f"  & {LABEL[algo]} & " + " & ".join(cells[algo]) + r" \\" for algo in MAIN]
    tex += [r"\bottomrule", r"\end{tabular}"]
    return rows, tex, failed


def _thousands(n: int, tex: bool) -> str:
    return f"{n:,}".replace(",", THIN if tex else ",")


def _table1(out_dir: str) -> tuple[list[dict], list[str]]:
    with open(os.path.join(out_dir, "config_paper.json")) as fh:
        config = json.load(fh)
    budget_rows = _read_csv(os.path.join(out_dir, "budget_summary.csv"))
    hp_n, hp_m = config["hyperparams"]["nsga2"], config["hyperparams"]["mopso"]
    slots, uav, seeds = config["evaluation_slots"], config["uav"], config["seeds"]

    def evals(algo: str, tex: bool) -> str:
        arm = [r for r in budget_rows if r["algorithm"] == algo]
        lo = min(int(r["n_evals_min"]) for r in arm)
        hi = max(int(r["n_evals_max"]) for r in arm)
        dash = "--" if tex else "–"
        return _thousands(lo, tex) if lo == hi else f"{_thousands(lo, tex)}{dash}{_thousands(hi, tex)}"

    def max_dev(algo: str, tex: bool) -> str:
        arm = [r["n_evals_max_rel_dev_pct"] for r in budget_rows if r["algorithm"] == algo]
        return max(arm, key=float) + (THIN + r"\%" if tex else " %")

    ks = {inst["K"] for inst in config["instances"].values()}
    if len(ks) != 1:
        raise ValueError("Table 1 assumes one K for every instance")
    k = ks.pop()

    def unit(value, symbol: str, tex: bool) -> str:
        return f"{value}{THIN if tex else ' '}{symbol}"

    power = uav["alpha_W_per_kg"] * uav["mass_kg"] + uav["beta_W"]
    instances = [f"{name.partition('-k')[0]} ({inst['N']})" for name, inst in config["instances"].items()]
    na = "--"
    # (tex label, plain label, NSGA-II, MOPSO); a callable value gets tex=True/False.
    per_algo = [
        ("Population / swarm size", "Population / swarm size", hp_n["pop"], hp_m["swarm"]),
        ("Generations / iterations", "Generations / iterations", hp_n["gens"], hp_m["iters"]),
        (r"Crossover prob.\ $p_c$", "Crossover prob. p_c", hp_n["pcx"], na),
        # pmut_counts applies only inside a mutation (effective 0.15), so it gets its own
        # conditional row rather than reading as a second unconditional rate.
        (r"Mutation prob.\ $p_m$", "Mutation prob. p_m", hp_n["pmut"], na),
        ("Count move, given mutation", "Count move, given mutation", hp_n["pmut_counts"], na),
        ("Archive size / grid divisions", "Archive size / grid divisions", na,
         f"{hp_m['archive_size']} / {hp_m['grid_divisions']}"),
        (r"Inertia $w$ / $c_1$ / $c_2$", "Inertia w / c1 / c2", na,
         f"{hp_m['w_inertia']} / {hp_m['c1']} / {hp_m['c2']}"),
        (r"Velocity clamp $v_{\max}$", "Velocity clamp v_max", na, hp_m["vmax_frac"]),
        # The decaying quantity is a per-particle probability, not a share of keys.
        ("Turbulence prob.", "Turbulence prob.", na,
         lambda tex: f"{hp_m['mut_rate']} {TO if tex else '->'} {hp_m['mut_floor']}"),
        ("Evaluation slots per run", "Evaluation slots per run",
         lambda tex: _thousands(slots["nsga2"], tex), lambda tex: _thousands(slots["mopso"], tex)),
        # Short enough that the table fits one column at \footnotesize.
        ("Evaluations per run", "Evaluations per run (min-max)",
         lambda tex: evals("nsga2", tex), lambda tex: evals("mopso", tex)),
        (rf"Max.\ deviation from {_thousands(slots['mopso'], True)}",
         f"Max. deviation from {_thousands(slots['mopso'], False)}",
         lambda tex: max_dev("nsga2", tex), lambda tex: max_dev("mopso", tex)),
    ]
    # (tex label, plain label, value) shared by both methods.
    shared = [
        (r"UAV mass $m$, cruise speed $v$", "UAV mass m, cruise speed v",
         lambda tex: f"{unit(uav['mass_kg'], 'kg', tex)}, "
                     f"{unit(uav['v_cruise_m_per_s'], 'm/s', tex)}"),
        (r"Power $\alpha$, $\beta$; $\alpha m+\beta$", "Power alpha, beta; alpha*m + beta",
         lambda tex: f"{unit(uav['alpha_W_per_kg'], 'W/kg', tex)}, {unit(uav['beta_W'], 'W', tex)}; "
                     f"{unit(f'{power:.1f}', 'W', tex)}"),
        (r"Fleet size $K$; seeds", "Fleet size K; seeds",
         lambda tex: f"{k}; {seeds[0]}{'--' if tex else '-'}{seeds[-1]}"),
        (r"Instances ($N$ POIs)", "Instances (N POIs)", lambda tex: ", ".join(instances)),
    ]

    def val(v, tex: bool) -> str:
        return v(tex) if callable(v) else str(v)

    rows = [{"parameter": plain, "nsga2": val(n, False), "mopso": val(m, False), "shared": ""}
            for _, plain, n, m in per_algo]
    rows += [{"parameter": plain, "nsga2": "", "mopso": "", "shared": val(v, False)}
             for _, plain, v in shared]
    tex = [GENERATED, r"\begin{tabular}{@{}lll@{}}", r"\toprule",
           r"Parameter & NSGA-II & MOPSO \\", r"\midrule"]
    tex += [f"{label} & {val(n, True)} & {val(m, True)} " + r"\\" for label, _, n, m in per_algo]
    tex.append(r"\midrule")
    for label, _, v in shared[:-1]:
        tex.append(f"{label} & \\multicolumn{{2}}{{l}}{{{val(v, True)}}} " + r"\\")
    # The instance list is too wide for one line of a column: two lines.
    label = shared[-1][0]
    tex.append(f"{label} & \\multicolumn{{2}}{{l}}{{{', '.join(instances[:2])},}} " + r"\\")
    tex.append(f" & \\multicolumn{{2}}{{l}}{{{', '.join(instances[2:])}}} " + r"\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    return rows, tex


TABLE_FILES = ("table1.csv", "table1.tex", "table2.csv", "table2.tex")


def tables(results_dir: str, out_dir: str, manuscript_dir: str | None = None) -> bool:
    """[R5, R6] Table 1 (configuration) and Table 2 (HV, IGD, CT, fleet) as CSV and bare tabular.

    Reads only files this script wrote into ``out_dir`` (``stats_28tests.csv``,
    ``fleet_coverage.csv``, ``config_paper.json``, ``budget_summary.csv``), so every
    cell traces to one of them. Copies the four files to ``manuscript_dir`` when given.
    """
    rows2, tex2, failed = _table2(out_dir)
    rows1, tex1 = _table1(out_dir)
    _write_csv(os.path.join(out_dir, "table1.csv"), rows1)
    _write_csv(os.path.join(out_dir, "table2.csv"), rows2)
    for name, lines in (("table1.tex", tex1), ("table2.tex", tex2)):
        with open(os.path.join(out_dir, name), "w") as fh:
            fh.write("\n".join(lines) + "\n")

    if manuscript_dir is None:
        print("  manuscript copy skipped: --out is not the default")
    elif not os.path.isdir(manuscript_dir):
        print(f"  manuscript copy skipped: {manuscript_dir} does not exist")
    else:
        for name in TABLE_FILES:
            shutil.copyfile(os.path.join(out_dir, name), os.path.join(manuscript_dir, name))
        print(f"  copied {len(TABLE_FILES)} files -> {os.path.relpath(manuscript_dir, ROOT)}")
    for f in failed:
        print(f"  CHECK FAILED: {f}")
    print(f"tables: Table 1 {len(rows1)} rows, Table 2 {len(rows2)} rows, checks "
          f"{'FAILED' if failed else 'OK'} -> {os.path.relpath(out_dir, ROOT)}/table{{1,2}}.{{csv,tex}}")
    return not failed


FIG_FRONT_INSTANCES = ("eil51-k3", "rat99-k3")
FIG_CONV_INSTANCE = "eil51-k3"
FIG_LABELS = {"nsga2": "NSGA-II", "mopso": "MOPSO"}
CM = 1 / 2.54
# Printed sizes: the figures are drawn at the size they print at, so font sizes are
# the printed sizes. Text width 15.93 cm and column width 7.665 cm come from daus.sty.
FIG1_SIZE = (15.9 * CM, 6.0 * CM)
FIG2_SIZE = (7.6 * CM, 6.5 * CM)
# Okabe-Ito vermillion/blue/green: pareto.py's red/green pair for 1 vs 3 drones fails
# the deutan check (dE 3.9), and fleet size is encoded by colour alone.
FIG_FLEET_COLORS = {1: "#D55E00", 2: "#0072B2", 3: "#009E73"}
FIG_RC = {"font.size": 7.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
          "legend.fontsize": 7.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
          "ytick.major.width": 0.6, "savefig.dpi": 300, "pdf.fonttype": 42}
FIG_FILES = ("fig1_fronts.pdf", "fig1_fronts.png", "fig2_convergence.pdf", "fig2_convergence.png")


def _n_active(routes) -> int:
    return sum(len(r) > 2 for r in routes)   # an empty route is [depot, depot]


def _save_fig(fig, out_dir: str, stem: str) -> None:
    # CreationDate None keeps the PDF byte-identical across rebuilds.
    fig.savefig(os.path.join(out_dir, f"{stem}.pdf"), metadata={"CreationDate": None})
    fig.savefig(os.path.join(out_dir, f"{stem}.png"), dpi=300)


def _eval_axis(algo: str, n_gen: int, n_evals: int) -> np.ndarray:
    """Evaluations spent when ``history[g]`` was recorded.

    MOPSO evaluates the whole swarm every iteration, so it is exact. NSGA-II saves
    only the run total (reused offspring are not re-evaluated, rate fixed by the
    operators), so its calls are spread linearly over generations.
    """
    g = np.arange(n_gen, dtype=float)
    if algo == "mopso":
        return 100.0 * (g + 1)
    return 100.0 + g * (n_evals - 100) / (n_gen - 1)


def _fig1(results_dir: str):
    import matplotlib.pyplot as plt
    from matplotlib.legend import Legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from unittest import mock
    from uav.viz import pareto

    fig, axes = plt.subplots(1, 2, figsize=FIG1_SIZE, layout="constrained")
    seen = set()
    for ax, panel, inst in zip(axes, "ab", FIG_FRONT_INSTANCES):
        records = {}
        for algo in MAIN:
            recs = []
            for seed in SEEDS:
                with open(_run_path(results_dir, inst, algo, seed)) as fh:
                    run = json.load(fh)
                recs += [(f1, f2, _n_active(r)) for (f1, f2), r in zip(run["front"], run["routes"])]
            records[FIG_LABELS[algo]] = recs
        with mock.patch.dict(pareto._FLEET_COLORS, FIG_FLEET_COLORS):
            pareto.plot_pareto(records, ax=ax)
        # One shared legend replaces plot_pareto's two per-axes legends.
        for leg in [c for c in ax.get_children() if isinstance(c, Legend)]:
            leg.remove()
        # plot_pareto's marker sizes suit a 6-inch axes; shrink them to the panel.
        clouds = [c for c in ax.collections if c.get_zorder() == 1]   # one per arm, MAIN order
        clouds[0].set_sizes([3])
        # plot_pareto greys both arms' clouds alike: rings keep MOPSO's apart (caption).
        clouds[1].set_sizes([5])
        clouds[1].set_facecolor("none")
        clouds[1].set_edgecolor("0.6")
        clouds[1].set_linewidths(0.3)
        for coll in ax.collections:
            if coll.get_zorder() != 1:
                coll.set_sizes([16])
                coll.set_linewidths(0.35)
                seen.update(k for k, c in FIG_FLEET_COLORS.items()
                            if np.allclose(coll.get_facecolor()[0][:3],
                                           plt.matplotlib.colors.to_rgb(c)))
        ax.text(0.97, 0.95, f"({panel}) {inst}", transform=ax.transAxes, ha="right", va="top")
    handles = [Line2D([], [], marker=pareto._ALGO_MARKERS[i], color="0.3", linestyle="none",
                      markersize=4.5, label=FIG_LABELS[a]) for i, a in enumerate(MAIN)]
    handles += [Patch(facecolor=FIG_FLEET_COLORS[k], edgecolor="black", linewidth=0.35,
                      label=f"{k} active drone{'s' if k > 1 else ''}") for k in sorted(seen)]
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), frameon=False,
               handlelength=1.2, columnspacing=1.4)
    return fig


def _fig2(results_dir: str) -> tuple[object, list[str]]:
    """Seed-median running best and IQR against evaluations; also the final-value check."""
    import matplotlib.pyplot as plt
    from uav.viz.convergence import _ALGO_COLORS, _OBJECTIVES, _stack

    failed = []
    grid = 100.0 * np.arange(1, 2792)   # MOPSO's exact evaluation counts
    fig, axes = plt.subplots(2, 1, figsize=FIG2_SIZE, sharex=True, layout="constrained")
    for i, algo in enumerate(MAIN):
        runs, curves = [], []
        for seed in SEEDS:
            with open(_run_path(results_dir, FIG_CONV_INSTANCE, algo, seed)) as fh:
                runs.append(json.load(fh))
        for seed, run in zip(SEEDS, runs):
            b = np.column_stack([_stack([run["history"]], "best", obj)[0] for obj in (0, 1)])
            # history's best is over the population/swarm, not the output set; the
            # running best is what the method has found so far.
            rb = np.minimum.accumulate(b, axis=0)
            front_min = np.min(np.asarray(run["front"], dtype=float), axis=0)
            if not np.allclose(rb[-1], front_min, rtol=1e-12, atol=0):
                failed.append(f"{FIG_CONV_INSTANCE} {algo} seed {seed}: running best "
                              f"{rb[-1].tolist()} != front extreme {front_min.tolist()}")
            x = _eval_axis(algo, len(b), run["n_evals"])
            # np.interp holds the end value for seeds that stop short of the grid's end.
            curves.append(np.column_stack([np.interp(grid, x, rb[:, k]) for k in (0, 1)]))
        curves = np.stack(curves)   # (seeds, grid, objective)
        for k, ax in enumerate(axes):
            q1, med, q3 = np.percentile(curves[:, :, k], [25, 50, 75], axis=0)
            style = "-" if i == 0 else "--"
            ax.fill_between(grid, q1, q3, color=_ALGO_COLORS[i], alpha=0.25, linewidth=0)
            ax.plot(grid, med, style, color=_ALGO_COLORS[i], linewidth=1.0, label=FIG_LABELS[algo])
    for ax, label in zip(axes, _OBJECTIVES):
        ax.set_ylabel(label)
        ax.set_xscale("log")
        ax.grid(True, which="major", linewidth=0.3, color="0.85")
    axes[-1].set_xlabel("Function evaluations")
    axes[0].legend(loc="upper right", frameon=False)
    return fig, failed


def figures(results_dir: str, out_dir: str, manuscript_dir: str | None = None) -> bool:
    """[T8, Fmt] Fig. 1 (pooled fronts, two instances) and Fig. 2 (convergence vs evaluations).

    Copies the four files to ``figures/`` beside ``manuscript_dir`` when given.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    with plt.rc_context(FIG_RC):
        fig = _fig1(results_dir)
        _save_fig(fig, fig_dir, "fig1_fronts")
        plt.close(fig)
        fig, failed = _fig2(results_dir)
        _save_fig(fig, fig_dir, "fig2_convergence")
        plt.close(fig)

    if manuscript_dir is None:
        print("  manuscript copy skipped: --out is not the default")
    else:
        dest = os.path.join(os.path.dirname(manuscript_dir), "figures")
        if not os.path.isdir(dest):
            print(f"  manuscript copy skipped: {dest} does not exist")
        else:
            for name in FIG_FILES:
                shutil.copyfile(os.path.join(fig_dir, name), os.path.join(dest, name))
            print(f"  copied {len(FIG_FILES)} files -> {os.path.relpath(dest, ROOT)}")
    for f in failed:
        print(f"  CHECK FAILED: {f}")
    print(f"figures: Fig. 1 {FIG1_SIZE[0] / CM:.1f} x {FIG1_SIZE[1] / CM:.1f} cm, Fig. 2 "
          f"{FIG2_SIZE[0] / CM:.1f} x {FIG2_SIZE[1] / CM:.1f} cm; running best = front extreme in "
          f"{2 * len(SEEDS) - len(failed)}/{2 * len(SEEDS)} runs -> {os.path.relpath(fig_dir, ROOT)}")
    return not failed


STEPS = {"check_solutions": check_solutions, "initial_fleet": initial_fleet, "budget": budget,
         "hv_audit": hv_audit, "stats": stats, "tables": tables, "figures": figures}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", action="append", choices=list(STEPS), metavar="STEP",
                    help=f"run only this step (repeatable); steps: {', '.join(STEPS)}")
    ap.add_argument("--results-dir", default=os.path.join(ROOT, "results_cobudget"))
    default_out = os.path.join(ROOT, "docs", "daus2027")
    ap.add_argument("--out", default=default_out)
    ap.add_argument("--manuscript-dir",
                    default=os.path.normpath(os.path.join(ROOT, os.pardir, "peer_review",
                                                          "daus2027", "generated")),
                    help="where step tables copies its files; figures go to its sibling figures/ "
                         "(only when --out is the default)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    # A trial --out must never overwrite the manuscript's copies.
    same_out = os.path.abspath(args.out) == os.path.abspath(default_out)
    copy_to = {"manuscript_dir": args.manuscript_dir if same_out else None}
    extra = {"tables": copy_to, "figures": copy_to}
    ok = True
    for name in args.only or list(STEPS):
        ok = STEPS[name](args.results_dir, args.out, **extra.get(name, {})) and ok
    # A failed check can change a conclusion (global rule 5): make it loud.
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
