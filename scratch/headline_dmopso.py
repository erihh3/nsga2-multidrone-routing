"""Step 1 headline extractor for the discrete-MOPSO (Attempt C) verification.

Reads a dmopso result JSON and reports the load-bearing numbers against the
encoding-hypothesis bar (best single tour < ~750 on eil51-k3). Read-only; computes
nothing the runner didn't already evaluate — route distances are recomputed from
the shared distance matrix so the "best single tour" is exact.

Run from the repo root:
    .venv/bin/python scratch/headline_dmopso.py
    .venv/bin/python scratch/headline_dmopso.py --instance eil51-k3 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Make ``uav`` importable without installing (mirrors pyproject's pytest pythonpath).
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, os.path.join(ROOT, "src"))

from uav.problem.fitness import ALPHA, BETA, MASS, V_CRUISE, route_distance  # noqa: E402
from uav.problem.instance import load_instance  # noqa: E402

# Energy (Dorling linear) -> distance: E = (alpha*m + beta)/v * total_dist.
DIST_PER_J = V_CRUISE / (ALPHA * MASS + BETA)


def main() -> int:
    ap = argparse.ArgumentParser(description="dmopso Step 1 headline numbers")
    ap.add_argument("--instance", default="eil51-k3")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bar", type=float, default=750.0,
                    help="success bar on the best single tour distance")
    args = ap.parse_args()

    tsp = args.instance.split("-")[0] + ".tsp"          # eil51-k3 -> eil51.tsp
    inst = load_instance(os.path.join(ROOT, "instances", tsp), k=3)

    jpath = os.path.join(ROOT, "results", f"{args.instance}_dmopso_{args.seed}.json")
    if not os.path.exists(jpath):
        print(f"ERROR: result JSON not found: {jpath}\n"
              f"Run the verification first:\n"
              f"  .venv/bin/python -m uav.experiment.runner "
              f"--algorithm dmopso --instance {args.instance} --seed {args.seed}")
        return 1
    d = json.load(open(jpath))

    # Shortest *non-empty* individual route in the final archive (empty idle-drone
    # routes [depot, depot] have distance 0 and are excluded).
    best_route = min(route_distance(r, inst.dist)
                     for sol in d["routes"] for r in sol if len(r) > 2)
    energies = [e for _, e in d["front"]]
    makespans = [m for m, _ in d["front"]]
    n_active = sorted(set(d["n_active_drones"]))

    # The bar metric: a 1-drone solution's full tour = its total distance =
    # energy * DIST_PER_J (all POIs on one route). None present => fleet still collapsed.
    one_drone = [e for e, na in zip(energies, d["n_active_drones"]) if na == 1]
    best_1drone_tour = min(one_drone) * DIST_PER_J if one_drone else None

    print(f"=== dmopso Attempt C — {args.instance} seed {args.seed} ===")
    print(f"n_evals (parity)     : {d['n_evals']}")
    print(f"wall_clock_s         : {d['wall_clock_s']}")
    print(f"NPS (front size)     : {len(d['front'])}")
    print(f"fleet classes present: {n_active}   (n_active_drones set)")
    print(f"energy range (J)     : {min(energies):.1f} .. {max(energies):.1f}")
    print(f"makespan range (s)   : {min(makespans):.1f} .. {max(makespans):.1f}")
    print()
    if best_1drone_tour is not None:
        verdict = ("UNDER the bar — encoding hypothesis SUPPORTED"
                   if best_1drone_tour < args.bar else
                   "ABOVE the bar — encoding not the binding constraint")
        print(f"BEST 1-DRONE TOUR    : {best_1drone_tour:.1f}   (bar: < ~{args.bar:.0f})")
        print(f"  -> {verdict}")
    else:
        print(f"BEST 1-DRONE TOUR    : none in front (no 1-drone solution survived;"
              f" fleet did not reach the 1-drone axis)")
    print(f"shortest non-empty route in front: {best_route:.1f}  "
          f"(partial tour, not the bar metric)")
    print(f"best front total distance        : {min(energies) * DIST_PER_J:.1f}  "
          f"(energy {min(energies):.0f} J * DIST_PER_J={DIST_PER_J:.5f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
