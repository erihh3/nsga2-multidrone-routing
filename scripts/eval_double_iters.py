"""Side-experiment: rerun MOPSO and DMOPSO on eil51-k3 with DOUBLE the iterations.

In the main study both swarm algorithms run at measured-budget parity with NSGA-II
(eil51-k3: iters=465, n_evals=swarm*(iters+1)=46_600, cached in
``results/eil51-k3_parity.json``). The convergence curves suggest they are still
improving at that cutoff. This script tests that hypothesis on a small sample:
MOPSO and DMOPSO x 5 seeds (0-4) at ``iters = 2 x parity`` (930), persisted to a
SEPARATE directory with a distinct ``2x_`` filename prefix so the canonical n=30
results are never touched.

This DELIBERATELY breaks budget parity — its numbers are exploratory and must not be
folded into the paper's equal-budget comparison. Use ``scripts/compare_double_iters.py``
to tabulate 1x vs 2x afterwards. Re-runs are cheap: an existing 2x JSON is skipped
unless ``--force`` is given.

Usage:  python scripts/eval_double_iters.py                                   # 2x, MOPSO+DMOPSO
        python scripts/eval_double_iters.py --algorithm dmopso --multiplier 4 # 4x dmopso only
        python scripts/eval_double_iters.py --force                           # ignore existing JSONs
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
RESULTS = os.path.join(ROOT, "results")          # canonical (read parity from here)
RESULTS_2X = os.path.join(ROOT, "results_2x")    # 2x outputs (kept separate)

INSTANCE = "eil51-k3"
SEEDS = tuple(range(5))                            # small sample: seeds 0-4
DEFAULT_MULTIPLIER = 2                             # 2x by default; --multiplier to override
_OPTIMIZERS = {"mopso": MOPSO, "dmopso": DiscreteMOPSO}

# MOPSO/DMOPSO persist the same hyperparameter subset (dmopso reuses MOPSO's fields).
_HP_FIELDS = ("swarm", "iters", "archive_size", "grid_divisions", "w_inertia",
              "c1", "c2", "vmax_frac", "mut_rate", "mut_floor")


def _hp_subset(hp: Hyperparams) -> dict:
    d = dataclasses.asdict(hp)
    sub = {k: d[k] for k in _HP_FIELDS}
    if d.get("extra"):
        sub["extra"] = d["extra"]
    return sub


def _serialize_run(res: RunResult, algo: str, seed: int, hp: Hyperparams) -> dict:
    """Identical schema to the canonical runner (incl. n_active_drones + history)."""
    return {
        "algorithm": algo,
        "instance": INSTANCE,
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


def _parity_iters() -> int:
    """The current parity iters for eil51-k3, read from its cached parity file."""
    with open(os.path.join(RESULTS, f"{INSTANCE}_parity.json")) as fh:
        return int(json.load(fh)["iters"])


def _out_path(algo: str, seed: int, mult: int) -> str:
    return os.path.join(RESULTS_2X, f"{mult}x_{INSTANCE}_{algo}_{seed}.json")


def main() -> None:
    ap = argparse.ArgumentParser(description="Iteration-budget side-experiment (eil51-k3).")
    ap.add_argument("--multiplier", type=int, default=DEFAULT_MULTIPLIER,
                    help="iters = multiplier x parity (default 2; e.g. 4 for 4x)")
    ap.add_argument("--algorithm", action="append", choices=list(_OPTIMIZERS),
                    help="mopso and/or dmopso (repeatable; default both)")
    ap.add_argument("--force", action="store_true",
                    help="re-run optimizers even if the output JSON already exists")
    args = ap.parse_args()

    mult = args.multiplier
    if mult < 1:
        raise SystemExit("--multiplier must be >= 1")
    algos = args.algorithm or list(_OPTIMIZERS)

    parity = _parity_iters()
    iters = mult * parity
    os.makedirs(RESULTS_2X, exist_ok=True)
    inst = load_instance(os.path.join(ROOT, "instances", "eil51.tsp"), k=3)
    expected_evals = Hyperparams().swarm * (iters + 1)

    print(f"instance: {inst.name}  N={inst.n_pois} K={inst.k}")
    print(f"parity iters={parity}  ->  {mult}x iters={iters}  "
          f"(expected n_evals/run = {expected_evals})")
    print(f"seeds={list(SEEDS)}  algos={algos}  out={RESULTS_2X}\n")

    for algo in algos:
        cls = _OPTIMIZERS[algo]
        hp = Hyperparams(iters=iters)
        print(f"{algo}  (swarm={hp.swarm}, iters={hp.iters}) ...")
        for seed in SEEDS:
            path = _out_path(algo, seed, mult)
            if os.path.exists(path) and not args.force:
                print(f"  [skip] {algo} seed {seed} (exists: {os.path.basename(path)})")
                continue
            res = cls(inst, Budget(), hp).run(seed=seed)
            assert res.n_evals == expected_evals, (
                f"{algo} seed {seed}: n_evals={res.n_evals} != {expected_evals}")
            with open(path, "w") as fh:
                json.dump(_serialize_run(res, algo, seed, hp), fh)
            print(f"  [{algo}] seed {seed}: n_evals={res.n_evals} "
                  f"wall={res.wall_clock_s:.1f}s front_pts={len(res.final_front)} "
                  f"-> {os.path.basename(path)}")
        print()

    print(f"done. {mult}x runs in {RESULTS_2X} (prefix '{mult}x_'). "
          f"Compare with: python scripts/compare_double_iters.py")


if __name__ == "__main__":
    main()
