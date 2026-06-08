"""THROWAWAY Phase-2 checkpoint — not part of the deliverable.

Runs full-budget NSGA-II on eil51-k3 over a few seeds and reports the fronts so
the "non-degenerate Pareto front" done-when can be eyeballed. Per-seed fronts are
thin (energy is near-constant among near-optimal tours; the trade-off lives in
makespan), so the meaningful object is the union front across seeds — that is
what feeds the Phase-4 reference front. Saves a scatter to
figures/nsga2_eil51_checkpoint.png. The real Pareto visualization is Phase 6;
delete this file once viz/ exists.

Usage:  python scripts/checkpoint_nsga2.py
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

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


def main() -> None:
    inst = load_instance(os.path.join(ROOT, "instances", "eil51.tsp"), k=3)
    hp = Hyperparams(pop=100, gens=500)

    all_objs: list[tuple[float, float]] = []
    per_seed_sizes: list[int] = []
    total_wall = 0.0
    for seed in SEEDS:
        res = NSGA2(inst, Budget(), hp).run(seed=seed)
        objs = sorted({s.objectives for s in res.final_front})
        all_objs.extend(objs)
        per_seed_sizes.append(len(objs))
        total_wall += res.wall_clock_s

    union = _nondominated(sorted(set(all_objs)))
    union.sort()
    umk = [o[0] for o in union]
    uen = [o[1] for o in union]

    print(f"instance        : {inst.name}  (N={inst.n_pois}, K={inst.k})")
    print(f"budget/seed     : pop={hp.pop} gens={hp.gens}   seeds={len(list(SEEDS))}")
    print(f"wall-clock total: {total_wall:.2f} s")
    print(f"front pts/seed  : {per_seed_sizes}")
    print(f"UNION nondom pts: {len(union)}")
    print(f"makespan (s)    : {min(umk):.3f}  ->  {max(umk):.3f}")
    print(f"energy   (J)    : {min(uen):.1f}  ->  {max(uen):.1f}")
    print("union front (makespan, energy):")
    for o in union:
        print(f"    ({o[0]:8.3f}, {o[1]:11.1f})")

    fig, ax = plt.subplots(figsize=(5, 4))
    amk = [o[0] for o in all_objs]
    aen = [o[1] for o in all_objs]
    ax.scatter(amk, aen, s=10, color="0.7", label="all seeds' fronts")
    ax.plot(umk, uen, "-o", color="C0", ms=6, label="union nondominated")
    ax.set_xlabel("makespan (s)")
    ax.set_ylabel("total energy (J)")
    ax.set_title(f"NSGA-II Pareto front — {inst.name} ({len(per_seed_sizes)} seeds)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(ROOT, "figures", "nsga2_eil51_checkpoint.png")
    fig.savefig(out, dpi=120)
    print(f"saved figure    : {out}")


if __name__ == "__main__":
    main()
