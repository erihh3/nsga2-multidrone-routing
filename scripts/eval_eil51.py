"""Phase 4 evaluation on eil51-k3: run both arms at parity, persist, tabulate.

This is the Phase-4 deliverable run (scope: eil51-k3 only). It:
  1. Runs NSGA-II x 10 seeds at full budget (pop=100, gens=500) and measures its
     mean evaluation count through the shared CountingEvaluator.
  2. Derives MOPSO's iters from that *measured* mean (config.parity_iters), which
     brings MOPSO's measured budget DOWN to NSGA-II's — never inflating NSGA-II.
  3. Runs MOPSO x 10 seeds at that parity budget.
  4. Persists every run to results/eil51-k3_<algo>_<seed>.json (hyperparams,
     n_evals, wall-clock, front objectives + routes, per-gen history) so the data
     is reusable and the Phase-5 runner inherits the format.
  5. Builds the union reference front, normalizes, and prints the metric table
     (median +- IQR per metric per algo + Mann-Whitney p), saving it to
     results/eil51-k3_metrics.json.

This MEASURES; it does not tune. MOPSO is configured (Phase 3) to trade
convergence for diversity and is dominated by NSGA-II — reported honestly.

Usage:  python scripts/eval_eil51.py
"""

from __future__ import annotations

import dataclasses
import json
import os

import numpy as np

from uav.algorithms.base import RunResult
from uav.algorithms.mopso import MOPSO
from uav.algorithms.nsga2 import NSGA2
from uav.evaluation.aggregate import Run, aggregate
from uav.experiment.config import SEEDS, Budget, Hyperparams, parity_iters
from uav.problem.instance import load_instance
from uav.solution import Solution

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
RESULTS = os.path.join(ROOT, "results")
INSTANCE = "eil51-k3"

# Reporting direction per metric: True = higher is better.
HIGHER_BETTER = {
    "nps": True, "dm": True, "hv": True,
    "spacing": False, "gd": False, "igd": False, "ct": False, "n_evals": None,
}

# Hyperparams is one combined dataclass holding BOTH arms' fields. Each run only
# uses its own subset, so we persist only those (otherwise an NSGA-II file would
# carry MOPSO's defaults — swarm/iters/c1/... — that NSGA-II never reads, and
# vice versa, which is misleading).
_HP_FIELDS = {
    "nsga2": ("pop", "gens", "pcx", "pmut", "pmut_counts"),
    "mopso": ("swarm", "iters", "archive_size", "grid_divisions", "w_inertia",
              "c1", "c2", "vmax_frac", "mut_rate", "mut_floor"),
}


def _hp_subset(hp: Hyperparams, algo: str) -> dict:
    d = dataclasses.asdict(hp)
    sub = {k: d[k] for k in _HP_FIELDS[algo]}
    if d.get("extra"):                       # only when non-empty
        sub["extra"] = d["extra"]
    return sub


def _serialize_run(res: RunResult, algo: str, seed: int, hp: Hyperparams) -> dict:
    return {
        "algorithm": algo,
        "instance": INSTANCE,
        "seed": seed,
        "hyperparams": _hp_subset(hp, algo),
        "n_evals": res.n_evals,
        "wall_clock_s": res.wall_clock_s,
        "front": [[float(s.makespan), float(s.energy)] for s in res.final_front],
        "routes": [[[int(p) for p in r] for r in s.routes] for s in res.final_front],
        # GenStats fields are numpy float64 (from min/mean/max over the pop) — cast.
        "history": [
            {"gen": int(g.gen),
             "best": [float(x) for x in g.best],
             "mean": [float(x) for x in g.mean],
             "worst": [float(x) for x in g.worst]}
            for g in res.history
        ],
    }


def _save(payload: dict, algo: str, seed: int) -> None:
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"{INSTANCE}_{algo}_{seed}.json")
    with open(path, "w") as fh:
        json.dump(payload, fh)


def _run_arm(optimizer_cls, algo: str, inst, hp: Hyperparams) -> list[Run]:
    runs: list[Run] = []
    for seed in SEEDS:
        res = optimizer_cls(inst, Budget(), hp).run(seed=seed)
        _save(_serialize_run(res, algo, seed, hp), algo, seed)
        runs.append(Run(seed=seed, front=res.final_front,
                        wall_clock_s=res.wall_clock_s, n_evals=res.n_evals))
        print(f"  [{algo}] seed {seed}: n_evals={res.n_evals} "
            f"wall={res.wall_clock_s:.1f}s front_pts={len(res.final_front)}")
    return runs


def _fmt(x: float) -> str:
    if x != x:                       # NaN
        return "  nan"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    if abs(x) >= 1:
        return f"{x:.3f}"
    return f"{x:.4f}"


def main() -> None:
    inst = load_instance(os.path.join(ROOT, "instances", "eil51.tsp"), k=3)
    print(f"instance: {inst.name}  N={inst.n_pois} K={inst.k}  seeds={len(SEEDS)}\n")

    # --- NSGA-II at full budget; measure its evaluation count -------------------
    nsga_hp = Hyperparams()
    print(f"NSGA-II  (pop={nsga_hp.pop}, gens={nsga_hp.gens}) ...")
    nsga_runs = _run_arm(NSGA2, "nsga2", inst, nsga_hp)
    nsga_mean_evals = float(np.mean([r.n_evals for r in nsga_runs]))

    # --- MOPSO at measured-eval parity (brought DOWN to NSGA-II's mean) ---------
    iters = parity_iters(nsga_mean_evals, Hyperparams().swarm)
    mopso_hp = Hyperparams(iters=iters)
    print(f"\nNSGA-II measured mean n_evals = {nsga_mean_evals:.0f}")
    print(f"=> MOPSO parity: swarm={mopso_hp.swarm}, iters={iters} "
        f"(nominal {Hyperparams().iters}) -> target {mopso_hp.swarm * (iters + 1)}")
    print(f"\nMOPSO  (swarm={mopso_hp.swarm}, iters={iters}) ...")
    mopso_runs = _run_arm(MOPSO, "mopso", inst, mopso_hp)
    mopso_mean_evals = float(np.mean([r.n_evals for r in mopso_runs]))

    # --- aggregate + report -----------------------------------------------------
    out = aggregate({"nsga2": nsga_runs, "mopso": mopso_runs})
    out["measured_evals"] = {"nsga2": nsga_mean_evals, "mopso": mopso_mean_evals}
    out["mopso_parity_iters"] = iters
    with open(os.path.join(RESULTS, f"{INSTANCE}_metrics.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    print("\n" + "=" * 72)
    print(f"eil51-k3 metric table  (median +- IQR, n={len(SEEDS)})")
    print(f"reference front: {out['reference_size']} points  "
        f"(union of both algos x all seeds, non-dominated)")
    print(f"measured n_evals/seed: NSGA-II {nsga_mean_evals:.0f}  |  "
        f"MOPSO {mopso_mean_evals:.0f}  (MOPSO brought down to parity)")
    print("=" * 72)
    hdr = f"{'metric':<10}{'dir':<5}{'NSGA-II (med+-IQR)':<26}{'MOPSO (med+-IQR)':<26}{'MW p':>8}"
    print(hdr)
    print("-" * len(hdr))
    for key in ("nps", "spacing", "gd", "igd", "dm", "hv", "ct", "n_evals"):
        e = out["summary"][key]
        hb = HIGHER_BETTER[key]
        direction = "+" if hb else ("-" if hb is False else " ")
        n = f"{_fmt(e['nsga2']['median'])} +- {_fmt(e['nsga2']['iqr'])}"
        m = f"{_fmt(e['mopso']['median'])} +- {_fmt(e['mopso']['iqr'])}"
        p = e.get("p", float("nan"))
        print(f"{key:<10}{direction:<5}{n:<26}{m:<26}{_fmt(p):>8}")
    print("-" * len(hdr))
    print("dir: + higher-is-better, - lower-is-better. CT = wall-clock seconds.")
    print(f"\nsaved: results/{INSTANCE}_metrics.json + per-run JSON ({len(SEEDS)} x 2)")


if __name__ == "__main__":
    main()
