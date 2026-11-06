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
from uav.evaluation import metrics as M
from uav.evaluation.aggregate import _front_objs
from uav.evaluation.reference_front import build_reference_front, normalize, reference_bounds
from uav.evaluation.stats import median_iqr
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


def _load_fronts(results_dir: str, instance: str, algo: str) -> list[list[Solution]]:
    """The saved final front of every seed, as ``Solution``s (the type metrics consume)."""
    fronts = []
    for seed in SEEDS:
        with open(_run_path(results_dir, instance, algo, seed)) as fh:
            run = json.load(fh)
        fronts.append([Solution(routes=tuple(tuple(int(p) for p in r) for r in routes),
                                makespan=float(f1), energy=float(f2))
                       for (f1, f2), routes in zip(run["front"], run["routes"])])
    return fronts


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


STEPS = {"check_solutions": check_solutions, "initial_fleet": initial_fleet, "budget": budget,
         "hv_audit": hv_audit}


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
