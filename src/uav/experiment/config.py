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
    # Tuned on eil51-k3 (10-seed union): a small velocity clamp curbs the erratic
    # jumps that wreck random-key tours, and c2 > c1 leans on the archive leader.
    # This cut Sigma d from ~1255 (vmax 0.5) to ~610. Standard PSO tuning, confined
    # to the optimizer loop (the one allowed point of difference) — not a 2-opt
    # local search on routes (that remains the sanctioned, user-gated fallback).
    w_inertia: float = 0.4      # inertia weight (Coello 2004: 0.4)
    c1: float = 1.5             # cognitive coefficient (pull to personal best)
    c2: float = 2.0             # social coefficient (pull to archive leader)
    vmax_frac: float = 0.2      # velocity clamp as a fraction of the [0,1] key range
    mut_rate: float = 0.5       # initial turbulence fraction; decays to 0 over iters
    extra: dict = field(default_factory=dict)
