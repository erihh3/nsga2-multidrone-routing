"""Throwaway single-seed BUDGET PROBE on eil51-k3 (writes to tmp_results/).

Before committing to a full n=30 re-run at a larger budget, this runs ONE seed of
four configurations on eil51-k3 to eyeball what raising the budget does to the
fronts:

  NSGA-II   pop=100, gens=1200
  NSGA-II   pop=150, gens=1200
  MOPSO     swarm=150, iters=1200
  DMOPSO    swarm=150, iters=1200      (the opt-in encoding-diagnostic variant)

This DELIBERATELY ignores measured-budget parity (the swarm arms get a fixed
iters=1200, not a NSGA-II-derived parity), so its numbers are EXPLORATORY and must
not enter the paper's co-equal table — same disclaimer the ``*_double_iters``
side-experiment carries. It bypasses ``experiment.runner`` (whose MOPSO/DMOPSO
branch forces parity and needs all 30 NSGA-II runs on disk) and constructs
``Hyperparams`` directly, exactly as ``scripts/eval_double_iters.py`` does.

Outputs land in a SEPARATE ``tmp_results/`` directory with the budget encoded in
each filename (``<instance>_<tag>_<seed>.json``) so the two NSGA-II configs do not
collide with each other and nothing touches the canonical ``results/``. The JSON
schema matches the runner's (incl. ``n_active_drones`` + ``history``) so
``scripts/budget_probe_report.py`` can read it back with the shared aggregator/viz.

Usage:  python scripts/eval_budget_probe.py                 # all 4 arms, seed 0
        python scripts/eval_budget_probe.py --seed 7        # a different single seed
        python scripts/eval_budget_probe.py --force         # ignore existing JSONs
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os

from uav.algorithms.base import RunResult
from uav.algorithms.dmopso import DiscreteMOPSO
from uav.algorithms.mopso import MOPSO
from uav.algorithms.nsga2 import NSGA2
from uav.experiment.config import Budget, Hyperparams
from uav.problem.instance import load_instance

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
TMP = os.path.join(ROOT, "tmp_results")          # disposable probe outputs (separate)

DEFAULT_INSTANCE = "eil51-k3"
DEFAULT_SEED = 4

# Each arm: label, optimizer class, the explicit Hyperparams, and a filename tag
# that encodes the budget (so the two NSGA-II configs get distinct stems). "All
# other hyperparameters remain the same" => only pop/gens (NSGA-II) and swarm/iters
# (swarm arms) move; archive_size/grid_divisions/inertia/c1/c2/turbulence stay at
# their defaults even though swarm grows to 150.
ARMS = (
    ("NSGA-II p100g1200", NSGA2,         Hyperparams(pop=100, gens=1000),      "nsga2-p100g1200"),
    ("NSGA-II p150g1200", NSGA2,         Hyperparams(pop=100, gens=3000),      "nsga2-p150g1200"),
    ("MOPSO s150i1200",   MOPSO,         Hyperparams(swarm=150, iters=2000),   "mopso-s150i1200"),
    ("DMOPSO s150i1200",  DiscreteMOPSO, Hyperparams(swarm=100, iters=3000),   "dmopso-s150i1200"),
)

# Per-arm hyperparameter subset to persist (mirrors the runner/eval_double_iters
# convention: an NSGA-II file must not carry MOPSO's swarm/iters/c1/... and vice
# versa). dmopso reuses MOPSO's fields verbatim.
_HP_FIELDS = {
    NSGA2: ("pop", "gens", "pcx", "pmut", "pmut_counts"),
    MOPSO: ("swarm", "iters", "archive_size", "grid_divisions", "w_inertia",
            "c1", "c2", "vmax_frac", "mut_rate", "mut_floor"),
}
_HP_FIELDS[DiscreteMOPSO] = _HP_FIELDS[MOPSO]


def _hp_subset(hp: Hyperparams, cls) -> dict:
    d = dataclasses.asdict(hp)
    sub = {k: d[k] for k in _HP_FIELDS[cls]}
    if d.get("extra"):
        sub["extra"] = d["extra"]
    return sub


def _serialize_run(res: RunResult, algo: str, instance: str, seed: int,
                   hp: Hyperparams, cls) -> dict:
    """Identical schema to the canonical runner (incl. n_active_drones + history)."""
    return {
        "algorithm": algo,
        "instance": instance,
        "seed": seed,
        "hyperparams": _hp_subset(hp, cls),
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
    """Map ``eil51-k3`` -> (path/to/eil51.tsp, K)."""
    base, _, k_tag = instance.partition("-k")
    k = int(k_tag) if k_tag else 3
    return os.path.join(ROOT, "instances", f"{base}.tsp"), k


def _out_path(instance: str, tag: str, seed: int) -> str:
    return os.path.join(TMP, f"{instance}_{tag}_{seed}.json")


def main() -> None:
    ap = argparse.ArgumentParser(description="Single-seed budget probe (eil51-k3).")
    ap.add_argument("--instance", default=DEFAULT_INSTANCE,
                    help=f"instance name (default: {DEFAULT_INSTANCE})")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help=f"single seed to run (default: {DEFAULT_SEED})")
    ap.add_argument("--force", action="store_true",
                    help="re-run optimizers even if the output JSON already exists")
    args = ap.parse_args()

    instance, seed = args.instance, args.seed
    path, k = _instance_spec(instance)
    inst = load_instance(path, k=k)
    os.makedirs(TMP, exist_ok=True)

    print(f"instance: {inst.name}  N={inst.n_pois} K={inst.k}  seed={seed}")
    print(f"out={os.path.relpath(TMP, ROOT)}  (off-parity probe — exploratory, NOT for the paper)\n")

    for label, cls, hp, tag in ARMS:
        algo = {NSGA2: "nsga2", MOPSO: "mopso", DiscreteMOPSO: "dmopso"}[cls]
        out = _out_path(instance, tag, seed)
        if os.path.exists(out) and not args.force:
            print(f"  [skip] {label} (exists: {os.path.basename(out)})")
            continue

        res = cls(inst, Budget(), hp).run(seed=seed)

        # Swarm arms re-evaluate the whole swarm each iter => deterministic budget.
        # NSGA-II under-evaluates, so its measured count is reported, not asserted.
        if cls in (MOPSO, DiscreteMOPSO):
            expected = hp.swarm * (hp.iters + 1)
            assert res.n_evals == expected, (
                f"{label}: n_evals={res.n_evals} != {expected}")

        with open(out, "w") as fh:
            json.dump(_serialize_run(res, algo, instance, seed, hp, cls), fh)
        print(f"  [{label}] n_evals={res.n_evals} wall={res.wall_clock_s:.1f}s "
              f"front_pts={len(res.final_front)} -> {os.path.basename(out)}")

    print(f"\ndone. probe runs in {os.path.relpath(TMP, ROOT)}/. "
          f"Report with: python scripts/budget_probe_report.py"
          + (f" --instance {instance}" if instance != DEFAULT_INSTANCE else "")
          + (f" --seed {seed}" if seed != DEFAULT_SEED else ""))


if __name__ == "__main__":
    main()
