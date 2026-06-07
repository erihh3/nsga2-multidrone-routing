"""NSGA-II via DEAP, two-part chromosome (perm of length N + counts of length K).

Operators (justified in CODE_IMPLEMENTATION_PLAN.md):
- selection: tools.selNSGA2 + tools.selTournamentDCD (canonical, Deb 2002)
- crossover P1: tools.cxOrdered (OX, order-preserving)
- crossover P2: count-segment swap + repair to (sum=N, each>=1)
- mutation: tools.mutShuffleIndexes on P1 + light +-1 reshuffle on P2
- weights = (-1.0, -1.0); both objectives minimized

Repair after every P2 operator is mandatory: a drone with 0 POIs silently
corrupts fitness.

Phase 2. Stub until the Phase-1 fitness gate is green.
"""

from __future__ import annotations

from uav.algorithms.base import Optimizer, RunResult


class NSGA2(Optimizer):
    def run(self, seed: int) -> RunResult:
        raise NotImplementedError("Phase 2: NSGA-II loop.")
