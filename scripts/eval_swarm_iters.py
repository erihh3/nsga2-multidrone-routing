"""Iteration-scaling investigation: MOPSO & DMOPSO at swarm=150, sweeping iters.

Focused follow-up to the 4-arm budget probe, now ONLY on the two swarm methods.
Holds swarm=150 fixed and sweeps iters across {1000, 2000, 3000} for BOTH mopso and
dmopso over 2 seeds on eil51-k3; all other hyperparameters stay at their defaults.
Off-parity by construction (fixed iters, not a NSGA-II-derived parity) => the numbers
are EXPLORATORY and must not enter the paper's co-equal table — same disclaimer the
``*_double_iters`` side-experiment carries.

Bypasses ``experiment.runner`` (whose swarm branch forces parity and needs the 30
NSGA-II runs on disk) and constructs ``Hyperparams`` directly, exactly as
``scripts/eval_double_iters.py`` does. Outputs land in ``tmp_results/`` with a
``swarmiters_`` prefix and the iters encoded in each stem, so nothing collides with
the budget-probe files or the canonical ``results/``. JSON schema matches the runner's
(incl. ``n_active_drones`` + ``history``) so ``scripts/swarm_iters_report.py`` reads it
back through the shared aggregator/viz.

Grid: {mopso, dmopso} x {1000, 2000, 3000} x {seed 0, 1} = 12 runs.
Deterministic budget per run = swarm*(iters+1): i1000->150150, i2000->300150,
i3000->450150 (asserted; both methods re-evaluate the whole swarm each iteration).

Usage:  python scripts/eval_swarm_iters.py                       # full 12-run grid
        python scripts/eval_swarm_iters.py --iters 1000,2000     # subset of budgets
        python scripts/eval_swarm_iters.py --seeds 0,1,2         # more seeds
        python scripts/eval_swarm_iters.py --algos dmopso        # one method
        python scripts/eval_swarm_iters.py --force               # ignore existing JSONs
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os

from uav.algorithms.base import RunResult
from uav.algorithms.dmopso import DiscreteMOPSO
from uav.algorithms.mopso import MOPSO
from uav.experiment.config import Budget, Hyperparams
from uav.problem.instance import load_instance

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
TMP = os.path.join(ROOT, "tmp_results")          # disposable outputs (separate dir)
PREFIX = "swarmiters_"                            # marks this investigation's files

DEFAULT_INSTANCE = "eil51-k3"
SWARM = 150                                       # held fixed (the investigation's premise)
DEFAULT_ITERS = (1000, 2000, 3000)
DEFAULT_SEEDS = (0, 1)
_OPTIMIZERS = {"mopso": MOPSO, "dmopso": DiscreteMOPSO}

# Both methods persist the same MOPSO hyperparameter subset (dmopso reuses the fields).
_HP_FIELDS = ("swarm", "iters", "archive_size", "grid_divisions", "w_inertia",
              "c1", "c2", "vmax_frac", "mut_rate", "mut_floor")


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


def _out_path(instance: str, algo: str, iters: int, seed: int) -> str:
    return os.path.join(TMP, f"{PREFIX}{instance}_{algo}_i{iters}_{seed}.json")


def _int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="MOPSO/DMOPSO iteration-scaling sweep (swarm fixed).")
    ap.add_argument("--instance", default=DEFAULT_INSTANCE,
                    help=f"instance name (default: {DEFAULT_INSTANCE})")
    ap.add_argument("--swarm", type=int, default=SWARM,
                    help=f"fixed swarm size (default: {SWARM})")
    ap.add_argument("--iters", type=_int_list, default=list(DEFAULT_ITERS),
                    help="comma-separated iteration budgets (default: 1000,2000,3000)")
    ap.add_argument("--seeds", type=_int_list, default=list(DEFAULT_SEEDS),
                    help="comma-separated seeds (default: 0,1)")
    ap.add_argument("--algos", type=lambda s: [a for a in s.split(",") if a],
                    default=list(_OPTIMIZERS),
                    help="comma-separated: mopso,dmopso (default: both)")
    ap.add_argument("--force", action="store_true",
                    help="re-run optimizers even if the output JSON already exists")
    args = ap.parse_args()

    bad = [a for a in args.algos if a not in _OPTIMIZERS]
    if bad:
        raise SystemExit(f"unknown algo(s): {bad}; valid: {list(_OPTIMIZERS)}")

    instance = args.instance
    path, k = _instance_spec(instance)
    inst = load_instance(path, k=k)
    os.makedirs(TMP, exist_ok=True)

    print(f"instance: {inst.name}  N={inst.n_pois} K={inst.k}  swarm={args.swarm} (fixed)")
    print(f"iters={args.iters}  seeds={args.seeds}  algos={args.algos}")
    print(f"out={os.path.relpath(TMP, ROOT)}/  prefix='{PREFIX}'  "
          f"(off-parity sweep — exploratory, NOT for the paper)\n")

    for algo in args.algos:
        cls = _OPTIMIZERS[algo]
        for iters in args.iters:
            hp = Hyperparams(swarm=args.swarm, iters=iters)
            expected = args.swarm * (iters + 1)
            print(f"{algo}  swarm={args.swarm}, iters={iters}  (n_evals/run = {expected})")
            for seed in args.seeds:
                out = _out_path(instance, algo, iters, seed)
                if os.path.exists(out) and not args.force:
                    print(f"  [skip] {algo} i{iters} seed {seed} (exists: {os.path.basename(out)})")
                    continue
                res = cls(inst, Budget(), hp).run(seed=seed)
                assert res.n_evals == expected, (
                    f"{algo} i{iters} seed {seed}: n_evals={res.n_evals} != {expected}")
                with open(out, "w") as fh:
                    json.dump(_serialize_run(res, algo, instance, seed, hp), fh)
                print(f"  [{algo}] i{iters} seed {seed}: n_evals={res.n_evals} "
                      f"wall={res.wall_clock_s:.1f}s front_pts={len(res.final_front)} "
                      f"-> {os.path.basename(out)}")
            print()

    print(f"done. runs in {os.path.relpath(TMP, ROOT)}/ (prefix '{PREFIX}'). "
          f"Report with: python scripts/swarm_iters_report.py"
          + (f" --instance {instance}" if instance != DEFAULT_INSTANCE else ""))


if __name__ == "__main__":
    main()
