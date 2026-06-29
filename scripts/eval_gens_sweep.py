"""Generation-scaling investigation: NSGA-II at pop=100, sweeping gens.

NSGA-II counterpart to ``scripts/eval_swarm_iters.py``. Holds pop=100 fixed and
sweeps gens across {1000, 2000, 3000} over 2 seeds on eil51-k3; all other
hyperparameters stay at their defaults. Off-parity by construction => EXPLORATORY,
not for the paper's co-equal table.

Bypasses ``experiment.runner`` and constructs ``Hyperparams`` directly. Outputs land
in ``tmp_results/`` with a ``genssweep_`` prefix and the gens encoded in each stem, so
nothing collides with the swarm-iters / budget-probe files or the canonical
``results/``. JSON schema matches the runner's so ``scripts/gens_sweep_report.py``
reads it back through the shared aggregator/viz.

NSGA-II UNDER-EVALUATES (untouched offspring keep the parent's fitness), so its
measured ``n_evals`` is NOT the nominal ``pop*gens`` and is NOT deterministic across
seeds — it is reported, never asserted (unlike the swarm methods).

Grid: nsga2 x {1000, 2000, 3000} x {seed 0, 1} = 6 runs.

Note: NSGA-II's per-generation operators do not depend on ``gens`` (fixed pcx/pmut),
so for a given seed the gens=3000 trajectory is the gens=1000 run extended — the
overlay is a clean prefix-extension. The FINAL fronts still differ per horizon
(history stores only per-gen summaries, not the population), so each level is a real
separate run.

Usage:  python scripts/eval_gens_sweep.py                     # full 6-run grid
        python scripts/eval_gens_sweep.py --gens 1000,2000    # subset of budgets
        python scripts/eval_gens_sweep.py --seeds 0,1,2       # more seeds
        python scripts/eval_gens_sweep.py --force             # ignore existing JSONs
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os

from uav.algorithms.base import RunResult
from uav.algorithms.nsga2 import NSGA2
from uav.experiment.config import Budget, Hyperparams
from uav.problem.instance import load_instance

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
TMP = os.path.join(ROOT, "tmp_results")          # disposable outputs (separate dir)
PREFIX = "genssweep_"                            # marks this investigation's files

DEFAULT_INSTANCE = "eil51-k3"
POP = 100                                         # held fixed (the investigation's premise)
DEFAULT_GENS = (1000, 2000, 3000)
DEFAULT_SEEDS = (0, 1)

_HP_FIELDS = ("pop", "gens", "pcx", "pmut", "pmut_counts")


def _hp_subset(hp: Hyperparams) -> dict:
    d = dataclasses.asdict(hp)
    sub = {k: d[k] for k in _HP_FIELDS}
    if d.get("extra"):
        sub["extra"] = d["extra"]
    return sub


def _serialize_run(res: RunResult, algo: str, instance: str, seed: int, hp: Hyperparams) -> dict:
    """Identical schema to the canonical runner (incl. n_active_drones + history)."""
    return {
        "algorithm": algo,
        "instance": instance,
        "seed": seed,
        "hyperparams": _hp_subset(hp),
        "n_evals": res.n_evals,
        "wall_clock_s": res.wall_clock_s,
        "front": [[float(s.makespan), float(s.energy)] for s in res.final_front],
        "routes": [[[int(p) for p in r] for r in s.routes] for s in res.final_front],
        "n_active_drones": [s.n_active_drones for s in res.final_front],
        "history": [
            {"gen": int(g.gen),
             "best": [float(x) for x in g.best],
             "mean": [float(x) for x in g.mean],
             "worst": [float(x) for x in g.worst]}
            for g in res.history
        ],
    }


def _instance_spec(instance: str) -> tuple[str, int]:
    base, _, k_tag = instance.partition("-k")
    k = int(k_tag) if k_tag else 3
    return os.path.join(ROOT, "instances", f"{base}.tsp"), k


def _out_path(instance: str, gens: int, seed: int) -> str:
    return os.path.join(TMP, f"{PREFIX}{instance}_nsga2_g{gens}_{seed}.json")


def _int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="NSGA-II generation-scaling sweep (pop fixed).")
    ap.add_argument("--instance", default=DEFAULT_INSTANCE,
                    help=f"instance name (default: {DEFAULT_INSTANCE})")
    ap.add_argument("--pop", type=int, default=POP,
                    help=f"fixed population size (default: {POP})")
    ap.add_argument("--gens", type=_int_list, default=list(DEFAULT_GENS),
                    help="comma-separated generation budgets (default: 1000,2000,3000)")
    ap.add_argument("--seeds", type=_int_list, default=list(DEFAULT_SEEDS),
                    help="comma-separated seeds (default: 0,1)")
    ap.add_argument("--force", action="store_true",
                    help="re-run optimizers even if the output JSON already exists")
    args = ap.parse_args()

    instance = args.instance
    path, k = _instance_spec(instance)
    inst = load_instance(path, k=k)
    os.makedirs(TMP, exist_ok=True)

    print(f"instance: {inst.name}  N={inst.n_pois} K={inst.k}  pop={args.pop} (fixed)")
    print(f"gens={args.gens}  seeds={args.seeds}")
    print(f"out={os.path.relpath(TMP, ROOT)}/  prefix='{PREFIX}'  "
          f"(off-parity sweep — exploratory, NOT for the paper)\n")

    for gens in args.gens:
        hp = Hyperparams(pop=args.pop, gens=gens)
        print(f"nsga2  pop={args.pop}, gens={gens}  (nominal {args.pop * gens}; "
              f"measured < nominal — NSGA-II under-evaluates)")
        for seed in args.seeds:
            out = _out_path(instance, gens, seed)
            if os.path.exists(out) and not args.force:
                print(f"  [skip] nsga2 g{gens} seed {seed} (exists: {os.path.basename(out)})")
                continue
            res = NSGA2(inst, Budget(), hp).run(seed=seed)
            with open(out, "w") as fh:
                json.dump(_serialize_run(res, "nsga2", instance, seed, hp), fh)
            print(f"  [nsga2] g{gens} seed {seed}: n_evals={res.n_evals} (measured) "
                  f"wall={res.wall_clock_s:.1f}s front_pts={len(res.final_front)} "
                  f"-> {os.path.basename(out)}")
        print()

    print(f"done. runs in {os.path.relpath(TMP, ROOT)}/ (prefix '{PREFIX}'). "
          f"Report with: python scripts/gens_sweep_report.py"
          + (f" --instance {instance}" if instance != DEFAULT_INSTANCE else ""))


if __name__ == "__main__":
    main()
