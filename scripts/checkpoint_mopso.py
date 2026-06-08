"""THROWAWAY Phase-3 checkpoint — not part of the deliverable.

Runs full-budget MOPSO on eil51-k3 over 10 seeds and reports the archive fronts,
overlaid against the NSGA-II union front, so the Phase-3 done-when — "archive
converges to a front comparable in spread to NSGA-II's" — can be eyeballed.
Per-seed fronts are thin (energy is near-constant among near-optimal tours; the
trade-off lives in makespan), so the meaningful object is the union front across
seeds. Saves an overlay scatter to figures/mopso_eil51_checkpoint.png. The real
Pareto visualization is Phase 6; delete this file (and checkpoint_nsga2.py) once
viz/ exists.

NOTE: budget here is the *nominal* swarm*iters. Measured budget parity vs NSGA-II
(~46.5k evals) is Phase-4 work; this script prints MOPSO's measured n_evals so the
gap is visible.

Usage:  python scripts/checkpoint_mopso.py
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from uav.algorithms.mopso import MOPSO
from uav.algorithms.nsga2 import NSGA2
from uav.experiment.config import Budget, Hyperparams
from uav.problem.instance import load_instance

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SEEDS = range(10)


def _nondominated(points):
    return [
        a for a in points
        if not any(
            b[0] <= a[0] and b[1] <= a[1] and (b[0] < a[0] or b[1] < a[1])
            for b in points if b is not a
        )
    ]


def _union_front(optimizer_cls, inst, hp):
    all_objs: list[tuple[float, float]] = []
    per_seed_sizes: list[int] = []
    total_wall = 0.0
    total_evals = 0
    for seed in SEEDS:
        res = optimizer_cls(inst, Budget(), hp).run(seed=seed)
        objs = sorted({s.objectives for s in res.final_front})
        all_objs.extend(objs)
        per_seed_sizes.append(len(objs))
        total_wall += res.wall_clock_s
        total_evals += res.n_evals
    union = sorted(_nondominated(sorted(set(all_objs))))
    n = len(list(SEEDS))
    return {
        "all": all_objs,
        "union": union,
        "sizes": per_seed_sizes,
        "wall": total_wall,
        "evals_mean": total_evals / n,
    }


def main() -> None:
    inst = load_instance(os.path.join(ROOT, "instances", "eil51.tsp"), k=3)
    hp = Hyperparams()  # nominal full budget for both arms

    mopso = _union_front(MOPSO, inst, hp)
    nsga = _union_front(NSGA2, inst, hp)

    print(f"instance        : {inst.name}  (N={inst.n_pois}, K={inst.k})")
    print(f"seeds           : {len(list(SEEDS))}")
    print("--- MOPSO ---")
    print(f"budget/seed     : swarm={hp.swarm} iters={hp.iters}")
    print(f"measured n_evals: {mopso['evals_mean']:.0f}  (mean/seed)")
    print(f"wall-clock total: {mopso['wall']:.2f} s")
    print(f"front pts/seed  : {mopso['sizes']}")
    print(f"UNION nondom pts: {len(mopso['union'])}")
    umk = [o[0] for o in mopso["union"]]
    uen = [o[1] for o in mopso["union"]]
    print(f"makespan (s)    : {min(umk):.3f}  ->  {max(umk):.3f}")
    print(f"energy   (J)    : {min(uen):.1f}  ->  {max(uen):.1f}")
    for o in mopso["union"]:
        print(f"    ({o[0]:8.3f}, {o[1]:11.1f})")
    print("--- NSGA-II (reference) ---")
    print(f"measured n_evals: {nsga['evals_mean']:.0f}  (mean/seed)")
    print(f"UNION nondom pts: {len(nsga['union'])}")
    nmk = [o[0] for o in nsga["union"]]
    nen = [o[1] for o in nsga["union"]]
    print(f"makespan (s)    : {min(nmk):.3f}  ->  {max(nmk):.3f}")
    print(f"energy   (J)    : {min(nen):.1f}  ->  {max(nen):.1f}")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    amk = [o[0] for o in mopso["all"]]
    aen = [o[1] for o in mopso["all"]]
    ax.scatter(amk, aen, s=10, color="0.8", label="MOPSO all seeds")
    ax.plot(umk, uen, "-o", color="C1", ms=6, label="MOPSO union")
    ax.plot(nmk, nen, "-s", color="C0", ms=5, label="NSGA-II union")
    ax.set_xlabel("makespan (s)")
    ax.set_ylabel("total energy (J)")
    ax.set_title(f"MOPSO vs NSGA-II Pareto front — {inst.name} ({len(list(SEEDS))} seeds)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(ROOT, "figures", "mopso_eil51_checkpoint.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"saved figure    : {out}")


if __name__ == "__main__":
    main()
