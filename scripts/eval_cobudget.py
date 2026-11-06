"""Phase 2: the co-equal overnight run (NSGA-II vs MOPSO vs DMOPSO, n=30, all instances).

Runs the chosen co-equal budget across the four instances, writing to a SEPARATE,
promotable ``results_cobudget/`` (never the canonical ``results/``):

  * NSGA-II  pop=100, gens=3000  x 30 seeds         -> measured mean evals (the master knob)
  * MOPSO / DMOPSO  swarm=100, iters=PARITY x 30 seeds

PARITY is derived PER INSTANCE from NSGA-II's MEASURED mean at the chosen swarm
(``config.parity_iters``: iters = round(mean/swarm) - 1), so both swarm methods sit at
NSGA-II's measured budget — co-equality on MEASURED evals, NSGA-II never inflated. Swarm
size was fixed at 100 by the Phase-1 swarm-size probe (DMOPSO clearly best there; MOPSO
indifferent; swarm <= archive_size=100 avoids archive-overflow pruning; matches canonical).

WHY NOT ``experiment.runner.run_all``: its MOPSO/DMOPSO branch hardcodes
``_resolve_parity(..., Hyperparams().swarm)`` = swarm 100 AND applies overrides without
re-deriving iters, so a non-default swarm would mis-budget. Here parity is derived
explicitly at the requested swarm. Persistence reuses the runner's helpers, so the JSON
schema/filenames are byte-compatible with the canonical pipeline (promotable later).

RESUMABLE: a run whose JSON already exists is skipped (``--force`` to redo). PARITY is
recomputed from the NSGA-II JSONs on every invocation (cheap; avoids the sticky-cache
gotcha) and needs ALL requested seeds' NSGA-II runs present before MOPSO/DMOPSO start.

Usage:
    python scripts/eval_cobudget.py                                  # all 4 instances, n=30, all 3 algos
    python scripts/eval_cobudget.py --instances eil51-k3            # one instance
    python scripts/eval_cobudget.py --algorithms nsga2             # NSGA-II only (then resume for the swarm arms)
    python scripts/eval_cobudget.py --seeds 0,1 --results-dir results_cobudget_smoke   # smoke test (separate dir!)
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from uav.algorithms.dmopso import DiscreteMOPSO
from uav.algorithms.mopso import MOPSO
from uav.algorithms.nsga2 import NSGA2
from uav.experiment.config import SEEDS, Budget, Hyperparams, parity_iters
from uav.experiment.runner import _run_json_path, _save_run, _serialize_run
from uav.problem.instance import load_instance

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DEFAULT_RESULTS = os.path.join(ROOT, "results_cobudget")   # separate; promotable later

INSTANCES = ("eil51-k3", "berlin52-k3", "eil76-k3", "rat99-k3")
ALGORITHMS = ("nsga2", "mopso", "dmopso")
_OPTIMIZERS = {"nsga2": NSGA2, "mopso": MOPSO, "dmopso": DiscreteMOPSO}

# The co-equal budget (Phase-1 decisions). NSGA-II is the master knob; swarm is locked.
POP, GENS, SWARM = 100, 3000, 100


def _instance_spec(instance: str) -> tuple[str, int]:
    base, _, k_tag = instance.partition("-k")
    k = int(k_tag) if k_tag else 3
    return os.path.join(ROOT, "instances", f"{base}.tsp"), k


def _parity_path(results_dir: str, instance: str) -> str:
    return os.path.join(results_dir, f"{instance}_parity.json")


def _resolve_parity(results_dir: str, instance: str, swarm: int,
                    seeds) -> tuple[int, float] | None:
    """(iters, measured_mean) for MOPSO/DMOPSO from NSGA-II's measured mean over
    ``seeds``. Recomputed from disk each call (cheap; no stale-cache risk); returns
    None until every requested seed's NSGA-II run is present. Caches for the record."""
    evals = []
    for s in seeds:
        p = _run_json_path(results_dir, instance, "nsga2", s)
        if not os.path.exists(p):
            return None
        with open(p) as fh:
            evals.append(json.load(fh)["n_evals"])
    mean = float(np.mean(evals))
    iters = parity_iters(mean, swarm)
    os.makedirs(results_dir, exist_ok=True)
    with open(_parity_path(results_dir, instance), "w") as fh:
        json.dump({"measured_nsga2_mean": mean, "iters": iters, "swarm": swarm,
                   "n_seeds": len(list(seeds)), "seeds": list(seeds)}, fh, indent=2)
    return iters, mean


def _split(s: str) -> list[str]:
    return [x for x in s.split(",") if x]


def _int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _run_arm(algo: str, instance: str, inst, hp: Hyperparams, seeds,
             results_dir: str, force: bool, expected_evals: int | None) -> None:
    cls = _OPTIMIZERS[algo]
    for seed in seeds:
        out = _run_json_path(results_dir, instance, algo, seed)
        if os.path.exists(out) and not force:
            print(f"  [skip] {algo:<6} {instance:<12} seed {seed} (exists)")
            continue
        res = cls(inst, Budget(), hp).run(seed=seed)
        if expected_evals is not None:        # swarm arms: deterministic budget
            assert res.n_evals == expected_evals, (
                f"{algo} {instance} seed {seed}: n_evals={res.n_evals} != {expected_evals}")
        _save_run(_serialize_run(res, algo, instance, seed, hp), results_dir, instance, algo, seed)
        print(f"  [{algo:<6}] {instance:<12} seed {seed}: n_evals={res.n_evals} "
              f"wall={res.wall_clock_s:.1f}s front={len(res.final_front)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Co-equal overnight run (separate dir).")
    ap.add_argument("--instances", type=_split, default=list(INSTANCES),
                    help="comma-separated instance names (default: all 4)")
    ap.add_argument("--algorithms", type=_split, default=list(ALGORITHMS),
                    help="comma-separated: nsga2,mopso,dmopso (default: all 3)")
    ap.add_argument("--seeds", type=_int_list, default=list(SEEDS),
                    help="comma-separated seeds (default: the 30 fixed seeds)")
    ap.add_argument("--pop", type=int, default=POP)
    ap.add_argument("--gens", type=int, default=GENS)
    ap.add_argument("--swarm", type=int, default=SWARM)
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS,
                    help="output dir (default: results_cobudget; use a SEPARATE dir for smoke tests)")
    ap.add_argument("--force", action="store_true",
                    help="re-run optimizers even if the output JSON already exists")
    args = ap.parse_args()

    bad_i = [i for i in args.instances if i not in INSTANCES]
    bad_a = [a for a in args.algorithms if a not in _OPTIMIZERS]
    if bad_i:
        raise SystemExit(f"unknown instance(s): {bad_i}; valid: {list(INSTANCES)}")
    if bad_a:
        raise SystemExit(f"unknown algorithm(s): {bad_a}; valid: {list(_OPTIMIZERS)}")

    results_dir = os.path.abspath(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)
    print(f"co-equal run -> {os.path.relpath(results_dir, ROOT)}/  "
          f"(NSGA-II pop={args.pop}/gens={args.gens}; swarm={args.swarm}; n={len(args.seeds)} seeds)")
    print(f"instances={args.instances}  algorithms={args.algorithms}\n")

    for instance in args.instances:
        path, k = _instance_spec(instance)
        inst = load_instance(path, k=k)
        print(f"== {instance}  (N={inst.n_pois}, K={inst.k}) ==")

        # NSGA-II first — it sets the parity budget the swarm methods must match.
        if "nsga2" in args.algorithms:
            hp = Hyperparams(pop=args.pop, gens=args.gens)
            _run_arm("nsga2", instance, inst, hp, args.seeds, results_dir, args.force, None)

        # MOPSO / DMOPSO at measured-eval parity (needs all NSGA-II seeds on disk).
        swarm_algos = [a for a in ("mopso", "dmopso") if a in args.algorithms]
        if swarm_algos:
            parity = _resolve_parity(results_dir, instance, args.swarm, args.seeds)
            if parity is None:
                print(f"  [error] {instance}: NSGA-II runs incomplete for the requested seeds "
                      f"— run nsga2 first (parity needs the measured mean). Skipping swarm arms.\n")
                continue
            iters, mean = parity
            expected = args.swarm * (iters + 1)
            print(f"  parity: NSGA-II measured mean={mean:.0f} -> swarm={args.swarm}, "
                  f"iters={iters} (n_evals={expected})")
            for algo in swarm_algos:
                hp = Hyperparams(swarm=args.swarm, iters=iters)
                _run_arm(algo, instance, inst, hp, args.seeds, results_dir, args.force, expected)
        print()

    print(f"done. results in {os.path.relpath(results_dir, ROOT)}/. "
          f"Report with: python scripts/cobudget_report.py"
          + (f" --results-dir {args.results_dir}" if results_dir != DEFAULT_RESULTS else ""))


if __name__ == "__main__":
    main()
