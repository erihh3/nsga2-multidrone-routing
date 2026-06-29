"""Swarm-size probe at a FIXED budget: MOPSO & DMOPSO over swarm in {100,150,200}.

Phase 1 of choosing the co-equal overnight budget. The evaluation budget is held
FIXED at B (the NSGA-II pop=100/gens=3000 measured mean on eil51-k3, ~279,300 evals);
for each swarm size the iters are DERIVED so swarm*(iters+1) ~ B. This isolates the
swarm-vs-iters trade-off at constant budget:

  swarm=100 -> iters~2792   (more iters: MOPSO's degradation zone — it tanked at 3000)
  swarm=150 -> iters~1861   (near MOPSO's observed peak ~2000)
  swarm=200 -> iters~1396   (fewer iters)

Holding swarm fixed and sweeping iters was ``eval_swarm_iters.py``; this is the dual
(fix the budget, sweep swarm). Off-parity-with-NSGA-II only in that NSGA-II is not
re-run here — the swarm arms all sit at the SAME measured budget B, which is the point.
EXPLORATORY: this picks the swarm size for the Phase-2 overnight run; it is not the
paper table.

Bypasses ``experiment.runner`` and constructs ``Hyperparams`` directly, like
``scripts/eval_swarm_iters.py``. Outputs to ``tmp_results/`` with a ``swarmsize_``
prefix (the swarm encoded in each stem) so nothing collides with the other probes or
the canonical ``results/``.

Grid: {mopso, dmopso} x swarm{100,150,200} x {seed 0,1} = 12 runs (~8 min total).

Usage:  python scripts/eval_swarm_size.py                          # full 12-run grid
        python scripts/eval_swarm_size.py --swarms 100,150,200,250 # more swarm sizes
        python scripts/eval_swarm_size.py --budget 279300          # set the fixed budget
        python scripts/eval_swarm_size.py --algos mopso --force
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os

from uav.algorithms.base import RunResult
from uav.algorithms.dmopso import DiscreteMOPSO
from uav.algorithms.mopso import MOPSO
from uav.experiment.config import Budget, Hyperparams, parity_iters
from uav.problem.instance import load_instance

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
TMP = os.path.join(ROOT, "tmp_results")          # disposable outputs (separate dir)
PREFIX = "swarmsize_"                            # marks this investigation's files

DEFAULT_INSTANCE = "eil51-k3"
# Fixed budget = NSGA-II pop=100/gens=3000 measured mean on eil51-k3 (the co-equal
# budget the Phase-2 overnight run will use). Round number; --budget to override.
DEFAULT_BUDGET = 279300
DEFAULT_SWARMS = (100, 150, 200)
DEFAULT_SEEDS = (0, 1)
_OPTIMIZERS = {"mopso": MOPSO, "dmopso": DiscreteMOPSO}

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


def _out_path(instance: str, algo: str, swarm: int, seed: int) -> str:
    return os.path.join(TMP, f"{PREFIX}{instance}_{algo}_s{swarm}_{seed}.json")


def _int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Swarm-size probe at a fixed budget (eil51-k3).")
    ap.add_argument("--instance", default=DEFAULT_INSTANCE,
                    help=f"instance name (default: {DEFAULT_INSTANCE})")
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET,
                    help=f"fixed evaluation budget B (default: {DEFAULT_BUDGET})")
    ap.add_argument("--swarms", type=_int_list, default=list(DEFAULT_SWARMS),
                    help="comma-separated swarm sizes (default: 100,150,200)")
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

    # Derive iters per swarm so every arm sits at ~B evals (parity_iters is the same
    # rule the co-equal study uses: iters = round(B/swarm) - 1).
    derived = {s: parity_iters(args.budget, s) for s in args.swarms}

    print(f"instance: {inst.name}  N={inst.n_pois} K={inst.k}  fixed budget B={args.budget}")
    print("derived iters per swarm (swarm*(iters+1) ~ B):")
    for s in args.swarms:
        it = derived[s]
        print(f"  swarm={s:<4} -> iters={it:<5} (n_evals={s * (it + 1)})")
    print(f"seeds={args.seeds}  algos={args.algos}  out={os.path.relpath(TMP, ROOT)}/  "
          f"prefix='{PREFIX}'  (EXPLORATORY — swarm selection for Phase 2)\n")

    for algo in args.algos:
        cls = _OPTIMIZERS[algo]
        for s in args.swarms:
            it = derived[s]
            hp = Hyperparams(swarm=s, iters=it)
            expected = s * (it + 1)
            print(f"{algo}  swarm={s}, iters={it}  (n_evals/run = {expected})")
            for seed in args.seeds:
                out = _out_path(instance, algo, s, seed)
                if os.path.exists(out) and not args.force:
                    print(f"  [skip] {algo} s{s} seed {seed} (exists: {os.path.basename(out)})")
                    continue
                res = cls(inst, Budget(), hp).run(seed=seed)
                assert res.n_evals == expected, (
                    f"{algo} s{s} seed {seed}: n_evals={res.n_evals} != {expected}")
                with open(out, "w") as fh:
                    json.dump(_serialize_run(res, algo, instance, seed, hp), fh)
                print(f"  [{algo}] s{s} seed {seed}: n_evals={res.n_evals} "
                      f"wall={res.wall_clock_s:.1f}s front_pts={len(res.final_front)} "
                      f"-> {os.path.basename(out)}")
            print()

    print(f"done. runs in {os.path.relpath(TMP, ROOT)}/ (prefix '{PREFIX}'). "
          f"Report with: python scripts/swarm_size_report.py"
          + (f" --instance {instance}" if instance != DEFAULT_INSTANCE else ""))


if __name__ == "__main__":
    main()
