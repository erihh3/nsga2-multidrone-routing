"""One place to change everything: instances, hyperparams, seeds, budget.

Equal-budget rule: pop*gens == swarm*iters. Wall-clock is a reported metric,
never balanced by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# 10 fixed seeds — set for random + numpy, logged per run. Single-seed = noise.
SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

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
    # Diversity-remediation settings (eil51-k3, 10-seed union). An earlier session
    # tuned vmax_frac DOWN to 0.2 for point-quality, but that collapsed the swarm
    # onto one region: degenerate 1-2 point fronts (spacing() = NaN, Phase 4
    # blocked). Restored to 0.5 (the Coello-style value) — the necessary lever for
    # a measurable, non-degenerate union (~6 points). c2 > c1 leans on the archive
    # leader. All confined to the optimizer loop (the one allowed point of
    # difference) — NOT a 2-opt local search on routes (user-gated fallback).
    w_inertia: float = 0.4      # inertia weight (Coello 2004: 0.4)
    c1: float = 1.5             # cognitive coefficient (pull to personal best)
    c2: float = 2.0             # social coefficient (pull to archive leader)
    vmax_frac: float = 0.5      # velocity clamp as a fraction of the [0,1] key range
    mut_rate: float = 0.5       # initial turbulence fraction
    mut_floor: float = 0.1      # turbulence never decays below this (keeps late
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
