"""One place to change everything: instances, hyperparams, seeds, budget.

Equal-budget rule: pop*gens == swarm*iters. Wall-clock is a reported metric,
never balanced by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# 30 fixed seeds — set for random + numpy, logged per run. Single-seed = noise.
SEEDS: tuple[int, ...] = tuple(range(30))

# K=3 for every instance (locked).
INSTANCES: tuple[str, ...] = ("eil51-k3", "berlin52-k3", "eil76-k3", "rat99-k3")
K = 3


@dataclass
class Budget:
    """Shared evaluation budget. n_evals == pop*gens == swarm*iters."""

    n_evals: int = 50_000  # placeholder; set per-instance in Phase 5


@dataclass
class Hyperparams:
    """Algorithm hyperparameters. Defaults from the original papers; do not sweep
    heavily — we compare algorithms, not tune one configuration."""

    # NSGA-II
    pop: int = 100
    gens: int = 500
    pcx: float = 0.9
    pmut: float = 0.3          # per-individual prob of inversion mutation on Part 1
    pmut_counts: float = 0.5   # prob the Part-2 (counts) mutation also fires
    # MOPSO (Coello Coello 2004 style). NOTE: iters is the *nominal* budget for
    # Phase 3; measured budget parity vs NSGA-II (~46.5k evals, not the nominal
    # swarm*iters) is deferred to the start of Phase 4 along with the shared
    # CountingEvaluator.
    swarm: int = 100
    iters: int = 500
    archive_size: int = 100
    grid_divisions: int = 30
    # Diversity/convergence levers (calibrated on eil51-k3, variable fleet, 3 seeds
    # x parity budget). Under the variable-fleet relaxation MOPSO's tours barely
    # converged: vmax_frac=0.5 let each argsort key jump half the [0,1] range per
    # step, so the visit order never settled (energy ~2x NSGA-II; on rat99 the best
    # energy moved ~3% in 465 iters). Lowering vmax_frac to 0.1 is the dominant fix
    # (GD 3.05->1.91, best-energy ~8050->6200 on eil51); trimming turbulence to
    # 0.3/0.05 sharpens it further while the floor still keeps late diversity alive.
    # NOTE: a separate, structural issue is NOT fixed by these knobs. MOPSO stays
    # ~3-drone and clearly dominated. The decoder was switched to stars-and-bars
    # cut-points (decode._counts_from_keys) so c_k=0 IS reachable, but that did not
    # change the fronts: idle-drone solutions need a near-optimal single tour to be
    # non-dominated, which random-key MOPSO can't produce, so they stay pruned. The
    # collapse is downstream of weak tour convergence (see MOPSO_INVESTIGATION.md);
    # only a memetic local search (2-opt) or a combinatorial PSO would change it,
    # both out of scope. c2>c1 leans on the archive leader. All confined to the
    # optimizer loop / decoder (MOPSO's allowed point of difference) — NOT a 2-opt
    # route local search (user-gated fallback).
    w_inertia: float = 0.4      # inertia weight (Coello 2004: 0.4)
    c1: float = 1.5             # cognitive coefficient (pull to personal best)
    c2: float = 2.0             # social coefficient (pull to archive leader)
    vmax_frac: float = 0.1      # velocity clamp as a fraction of the [0,1] key range
    mut_rate: float = 0.3       # initial turbulence fraction
    mut_floor: float = 0.05     # turbulence never decays below this (keeps late
    #                             diversity alive; per Coello 2004 mutation persists)
    extra: dict = field(default_factory=dict)


def parity_iters(measured_nsga2_mean: float, swarm: int) -> int:
    """MOPSO ``iters`` that matches NSGA-II's *measured* mean evaluation count.

    Budget parity is on MEASURED evals, not the nominal ``pop*gens``. NSGA-II
    under-evaluates (untouched offspring keep their parent's fitness, so only
    invalid offspring are re-evaluated), measuring ~46.6k/run on eil51-k3. MOPSO
    re-evaluates the *whole* swarm every iteration plus the initial swarm, so its
    measured count is exactly ``swarm * (iters + 1)``. Solving that for ``iters``:

        iters = round(measured_nsga2_mean / swarm) - 1

    **Co-equality guard:** this can only bring MOPSO's nominal 50,100
    (swarm=100, iters=500) *down* to ~46.5k to meet NSGA-II — it must never be used
    to inflate NSGA-II's budget upward to flatter the comparison.
    """
    return max(1, round(measured_nsga2_mean / swarm) - 1)
