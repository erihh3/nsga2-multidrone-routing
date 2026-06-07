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
    pmut: float = 0.1
    # MOPSO
    swarm: int = 100
    iters: int = 500
    archive_size: int = 100
    grid_divisions: int = 30
    extra: dict = field(default_factory=dict)
