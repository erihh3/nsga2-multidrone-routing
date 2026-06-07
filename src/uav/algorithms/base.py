"""The interface both optimizers obey.

``RunResult`` carries the final non-dominated set *and* the per-generation
convergence history, so ``viz/convergence.py`` works for both algorithms with no
special-casing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from uav.solution import Solution


@dataclass
class GenStats:
    """One generation's convergence snapshot (per objective: best/mean/worst)."""

    gen: int
    best: tuple[float, float]
    mean: tuple[float, float]
    worst: tuple[float, float]


@dataclass
class RunResult:
    final_front: list[Solution]
    history: list[GenStats] = field(default_factory=list)
    wall_clock_s: float = 0.0
    n_evals: int = 0


class Optimizer(ABC):
    """Common contract. The constructor takes the shared problem + budget + hp;
    ``run(seed)`` is the only algorithm-specific entry point.
    """

    def __init__(self, instance, budget, hp) -> None:
        self.instance = instance
        self.budget = budget
        self.hp = hp

    @abstractmethod
    def run(self, seed: int) -> RunResult:
        """Execute one seeded run and return its RunResult."""
        raise NotImplementedError
