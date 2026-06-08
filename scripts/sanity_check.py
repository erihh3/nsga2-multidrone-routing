"""THROWAWAY pre-Phase-3 sanity check — not part of the deliverable.

Two manual checks the user runs by hand before MOPSO is built:

  sweep : reduced-budget NSGA-II across all four instances, to confirm the SHARED
          CORE (parser, distance matrix, decode, fitness) generalizes across
          coordinate scales. berlin52's coords are in the thousands vs the eil
          instances' tens — a distance/scale/rounding bug hides on eil51 and only
          surfaces on berlin52. We anchor the min-makespan endpoint against the
          Necula-Breaban MinMax longest-tour optima.

  time  : full-budget single-seed wall-clock on the heaviest instance, to size the
          Phase-5 matrix (equal budget => MOPSO per-seed ~ NSGA-II per-seed).

Mirrors scripts/checkpoint_nsga2.py for instance loading, optimizer construction,
run(seed), and reading RunResult / Solution. Additive only; writes optional CSVs
to scratch/. Delete once Phase 5 has its own runner.

Usage:
    python scripts/sanity_check.py sweep
    python scripts/sanity_check.py time [--instance rat99-k3]
"""

from __future__ import annotations

import argparse
import csv
import os

from uav.algorithms.nsga2 import NSGA2
from uav.experiment.config import Budget, Hyperparams
from uav.problem.fitness import ALPHA, BETA, MASS, V_CRUISE
from uav.problem.instance import load_instance

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SCRATCH = os.path.join(ROOT, "scratch")

# Locked physics as the single source of truth (== spec's 8.02 J/m, 15.0 m/s).
ENERGY_PER_M = (ALPHA * MASS + BETA) / V_CRUISE

# Necula-Breaban mTSPLIB MinMax optima: the longest single-tour length at the
# exact CPLEX MinMax optimum, in RAW Euclidean distance (NOT nint-rounded).
# Used only as an order-of-magnitude floor for the min-makespan endpoint.
# Source: Necula & Breaban mTSPLIB benchmark (MinMaxMTSP); unverified transcription.
MINMAX_OPT = {
    "eil51-k3": 160,      # control
    "berlin52-k3": 3074,
    "eil76-k3": 197,
    "rat99-k3": 518,
}


def _nondominated(points):
    return [
        a for a in points
        if not any(
            b[0] <= a[0] and b[1] <= a[1] and (b[0] < a[0] or b[1] < a[1])
            for b in points if b is not a
        )
    ]


def _load(name: str):
    """Map an instance name like 'berlin52-k3' to its .tsp file + k."""
    stem, k = name.rsplit("-k", 1)
    return load_instance(os.path.join(ROOT, "instances", f"{stem}.tsp"), k=int(k))


def _union_front(name: str, hp: Hyperparams, seeds):
    """Run NSGA-II over `seeds`, return (union nondominated objectives, total wall)."""
    inst = _load(name)
    objs: list[tuple[float, float]] = []
    wall = 0.0
    for seed in seeds:
        res = NSGA2(inst, Budget(), hp).run(seed=seed)
        objs.extend(s.objectives for s in res.final_front)
        wall += res.wall_clock_s
    return _nondominated(sorted(set(objs))), wall


# --- sweep ----------------------------------------------------------------------

def cmd_sweep(args) -> None:
    hp = Hyperparams(pop=52, gens=100)
    seeds = (0, 1)
    instances = ("eil51-k3", "berlin52-k3", "eil76-k3", "rat99-k3")

    print(f"SWEEP  (reduced budget pop={hp.pop} gens={hp.gens}, seeds={seeds})")
    print("Cross-instance core sanity: min-makespan endpoint vs MinMax optimum.\n")
    header = f"{'instance':12} {'longest_tour':>13} {'minmax_opt':>11} {'ratio':>7}  flag"
    print(header)
    print("-" * len(header))

    rows = []
    for name in instances:
        front, wall = _union_front(name, hp, seeds)
        best_mk = min(o[0] for o in front)           # min-makespan endpoint
        # energy at that same endpoint (for the info-only total distance)
        best_en = min(o[1] for o in front if o[0] == best_mk)
        longest_tour = best_mk * V_CRUISE
        opt = MINMAX_OPT[name]
        ratio = longest_tour / opt
        if ratio < 1.0:
            flag = "RED: below proven optimum — distance/scale/decode bug"
        elif ratio > 3.0:
            flag = "CHECK: far above floor — under-converged or scale bug"
        else:
            flag = "OK (plausible for reduced budget)"
        total_distance = best_en / ENERGY_PER_M

        print(f"{name:12} {longest_tour:13.1f} {opt:11d} {ratio:7.2f}  {flag}")
        print(f"{'':12} (info: total_distance ~= {total_distance:.1f} m "
              f"from min-makespan endpoint; {wall:.2f}s for {len(seeds)} seeds)")
        rows.append({
            "instance": name, "longest_tour": round(longest_tour, 1),
            "minmax_opt": opt, "ratio": round(ratio, 3),
            "total_distance": round(total_distance, 1),
            "front_points": len(front), "wall_s": round(wall, 2), "flag": flag,
        })

    if args.csv:
        os.makedirs(SCRATCH, exist_ok=True)
        out = os.path.join(SCRATCH, "sanity_sweep.csv")
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {out}")


# --- time -----------------------------------------------------------------------

def cmd_time(args) -> None:
    name = args.instance
    hp = Hyperparams(pop=100, gens=500)
    inst = _load(name)

    res = NSGA2(inst, Budget(), hp).run(seed=0)
    wall = res.wall_clock_s

    print(f"TIME  (full budget pop={hp.pop} gens={hp.gens}, seed=0)")
    print(f"instance     : {name}  (N={inst.n_pois}, K={inst.k})")
    print(f"wall_clock_s : {wall:.2f}")
    print(f"n_evals      : {res.n_evals}")
    print(
        f"\nPhase-5 estimate for this instance: ~{wall * 10:.0f} s for 10 NSGA-II "
        f"seeds,\n  ~{wall * 10:.0f} s more for 10 MOPSO seeds (equal-budget parity) "
        f"=>\n  ~{wall * 20:.0f} s total for this instance's column of the 4x2x10 matrix."
    )

    if args.csv:
        os.makedirs(SCRATCH, exist_ok=True)
        out = os.path.join(SCRATCH, "sanity_time.csv")
        with open(out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["instance", "pop", "gens", "wall_s", "n_evals"])
            w.writerow([name, hp.pop, hp.gens, round(wall, 2), res.n_evals])
        print(f"\nwrote {out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pre-Phase-3 NSGA-II sanity checks.")
    sub = p.add_subparsers(dest="mode", required=True)

    s = sub.add_parser("sweep", help="cross-instance core sanity (reduced budget)")
    s.add_argument("--csv", action="store_true", help="also write scratch/sanity_sweep.csv")
    s.set_defaults(func=cmd_sweep)

    t = sub.add_parser("time", help="full-budget timing probe (one seed)")
    t.add_argument("--instance", default="rat99-k3", help="instance name (default rat99-k3)")
    t.add_argument("--csv", action="store_true", help="also write scratch/sanity_time.csv")
    t.set_defaults(func=cmd_time)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
