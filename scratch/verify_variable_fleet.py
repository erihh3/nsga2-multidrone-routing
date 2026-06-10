"""Step-1 verification (eil51-k3 only): variable-fleet front shape + extremes.

Throwaway. Runs NSGA-II seed 42 (pop=100, gens=500) and MOPSO seed 42 at a
one-off budget derived from that single NSGA-II run's measured evals (NOT the
10-seed sweep parity). Inspects the relaxed front; the full sweep is the user's.
"""
from __future__ import annotations

import os

import numpy as np

from uav.algorithms.mopso import MOPSO
from uav.algorithms.nsga2 import NSGA2
from uav.experiment.config import Budget, Hyperparams, parity_iters
from uav.problem.instance import load_instance

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SEED = 42
TSP_OPT = 426                       # eil51 single-tour optimum (nint EUC_2D)
POWER_OVER_V = (46.7 * 2.0 + 26.9) / 15.0   # (alpha*m + beta)/v = energy per unit dist


def _distinct(front):
    return np.unique(np.array([[s.makespan, s.energy] for s in front]), axis=0)


def _summary(tag, res):
    objs = _distinct(res.final_front)
    active = sorted({s.n_active_drones for s in res.final_front})
    ms, en = objs[:, 0], objs[:, 1]
    print(f"\n[{tag}] n_evals={res.n_evals}  wall={res.wall_clock_s:.1f}s")
    print(f"  distinct front points: {len(objs)}")
    print(f"  active-drone counts present: {active}")
    print(f"  makespan range: {ms.min():.2f} .. {ms.max():.2f} s")
    print(f"  energy   range: {en.min():.1f} .. {en.max():.1f} J  (span {en.max()-en.min():.1f})")
    # extremes
    i_ms = int(np.argmin(ms))            # makespan-optimal end
    i_en = int(np.argmin(en))            # energy-optimal end
    print(f"  makespan end: {ms[i_ms]:.2f} s @ {en[i_ms]:.1f} J")
    print(f"  energy   end: {ms[i_en]:.2f} s @ {en[i_en]:.1f} J "
          f"(implied total dist {en[i_en]/POWER_OVER_V:.0f}, TSP opt {TSP_OPT})")
    return res.wall_clock_s


def main():
    inst = load_instance(os.path.join(ROOT, "instances", "eil51.tsp"), k=3)
    print(f"instance {inst.name}  N={inst.n_pois} K={inst.k}  seed={SEED}")
    print(f"energy/dist factor (alpha*m+beta)/v = {POWER_OVER_V:.4f} J per unit dist")
    print(f"=> 1-drone TSP-opt energy ~= {TSP_OPT * POWER_OVER_V:.0f} J")

    nsga = NSGA2(inst, Budget(), Hyperparams()).run(seed=SEED)
    w_nsga = _summary("NSGA-II pop=100 gens=500", nsga)

    iters = parity_iters(float(nsga.n_evals), Hyperparams().swarm)
    print(f"\nMOPSO one-off budget: iters={iters} (from this single NSGA-II run)")
    mopso = MOPSO(inst, Budget(), Hyperparams(iters=iters)).run(seed=SEED)
    w_mopso = _summary("MOPSO (parity one-off)", mopso)

    print(f"\nwall-clock: NSGA-II {w_nsga:.1f}s  MOPSO {w_mopso:.1f}s  "
          f"(per-run estimate for the 80-run sweep)")


if __name__ == "__main__":
    main()
