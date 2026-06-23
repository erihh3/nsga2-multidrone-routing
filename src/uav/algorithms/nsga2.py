"""NSGA-II via DEAP, two-part chromosome (perm of length N + counts of length K).

Operators (justified in CODE_IMPLEMENTATION_PLAN.md):
- selection: tools.selNSGA2 (environmental) + tools.selTournamentDCD (mating)
  (canonical NSGA-II, Deb 2002), with duplicate removal before environmental
  selection
- crossover P1: tools.cxOrdered (OX, order-preserving)
- crossover P2: one-point on counts + repair to (sum=N, each>=0)
- mutation P1: inversion (reverse a random sub-tour) + P2: shift/consolidate split
- weights = (-1.0, -1.0); both objectives minimized

**Two deviations from the literal plan table, forced by a measured premature-
convergence failure (the canonical OX + shuffle collapsed the population to a
single genotype by ~gen 100 and returned a dominated 2-point front):**
1. *Inversion* mutation replaces ``mutShuffleIndexes``. Inversion reverses a
   contiguous sub-tour, preserving most adjacencies — the standard, far stronger
   permutation operator for TSP-like problems. Shuffle scrambles adjacency and
   barely improves tours.
2. *Duplicate removal* (by objective vector) before ``selNSGA2``. Without it,
   identical individuals flood the population and crowding distance degenerates.
Both are ordinary GA-quality choices confined to the optimizer loop — not a
2-opt local-search bolt-on (that is reserved for MOPSO's sanctioned fallback) and
not a change to the shared core.

The genotype is the *only* thing that differs from MOPSO. Everything downstream
(decode_two_part, evaluate) is the shared core, called unchanged — there is no
GA-specific fitness or distance code here.

Repair after every P2 operator restores ``sum(counts) == N`` with each count
``>= 0`` (variable fleet: empty drones are legal). The decoder guardrail now only
rejects negative counts.
"""

from __future__ import annotations

import random
import time

import numpy as np
from deap import base, creator, tools

from uav.algorithms.base import GenStats, Optimizer, RunResult
from uav.problem.decode import decode_two_part
from uav.problem.fitness import CountingEvaluator
from uav.seeds import set_all_seeds
from uav.solution import Solution

# --- DEAP global types: create once at import (creator mutates module-global
# state, and run() is called once per seed). Guard so re-import is a no-op. -------
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
if not hasattr(creator, "Individual"):
    # The individual *is* the permutation (a list of POI ids); the per-drone
    # split lives in the .counts attribute. This lets DEAP's stock permutation
    # operators act on the list directly while we handle Part 2 separately.
    creator.create("Individual", list, fitness=creator.FitnessMin, counts=None)


# --- Part-2 (counts) helpers ----------------------------------------------------

def _repair_counts(counts: list[int], n_pois: int, k: int) -> list[int]:
    """Force counts to be a valid split: sum == n_pois and every entry >= 0.

    Variable fleet: zero counts are legal (a drone may stay at the depot), so the
    only clamp is non-negativity. Nudge single units at random indices until the
    total matches; surplus is only ever taken from a drone that still has > 0.
    """
    counts = [max(0, int(c)) for c in counts]
    total = sum(counts)
    while total > n_pois:                      # remove units from drones with >0
        i = random.randrange(k)
        if counts[i] > 0:
            counts[i] -= 1
            total -= 1
    while total < n_pois:                       # add units anywhere
        counts[random.randrange(k)] += 1
        total += 1
    return counts


def _cx_counts(c1: list[int], c2: list[int], n_pois: int, k: int) -> tuple[list[int], list[int]]:
    """One-point crossover on the counts vectors, each child repaired to validity."""
    if k < 2:
        return _repair_counts(c1, n_pois, k), _repair_counts(c2, n_pois, k)
    point = random.randint(1, k - 1)
    child1 = c1[:point] + c2[point:]
    child2 = c2[:point] + c1[point:]
    return _repair_counts(child1, n_pois, k), _repair_counts(child2, n_pois, k)


def _mut_counts(counts: list[int], n_pois: int, k: int) -> list[int]:
    """Mutate the per-drone split. Two moves, chosen with equal probability:

    - *shift*: move one POI from a random non-empty donor to another drone (the
      fine, local move — explores nearby balanced splits);
    - *consolidate*: dump a non-empty donor's entire count onto another drone
      (``c_dst += c_src; c_src = 0``), deactivating that drone in one step.

    The consolidation move exists for **reachability of the relaxed search space**,
    not as a local search. One-POI shifts would need ~N/K accepted steps in a row
    (~17 on eil51) to drive a balanced split down to a single active drone, so the
    energy end of the front — one drone flying the whole tour — is effectively
    unreachable by drift alone. A whole-count dump makes 1-/2-drone solutions
    reachable in a single mutation.
    """
    counts = list(counts)
    if k >= 2:
        donors = [i for i in range(k) if counts[i] > 0]
        if donors:
            src = random.choice(donors)
            dst = random.choice([i for i in range(k) if i != src])
            if random.random() < 0.5:
                # consolidate: empty the donor entirely onto the recipient
                counts[dst] += counts[src]
                counts[src] = 0
            else:
                # shift: move a single POI
                counts[src] -= 1
                counts[dst] += 1
    return _repair_counts(counts, n_pois, k)


def _inversion(ind) -> None:
    """In-place inversion mutation: reverse a random contiguous sub-tour.

    The adjacency-preserving permutation operator (a single 2-opt move's effect on
    ordering). Far stronger than index shuffling for tour-shaped genotypes, and
    the empirical fix for the population collapse seen with shuffle.
    """
    n = len(ind)
    if n < 2:
        return
    a, b = sorted(random.sample(range(n), 2))
    ind[a:b + 1] = ind[a:b + 1][::-1]


# --- individual construction + conversion --------------------------------------

def _random_composition(n: int, k: int) -> list[int]:
    """A uniform random composition of ``n`` into ``k`` non-negative parts.

    Stars-and-bars: place K-1 cut points (with replacement) in {0..N} and take the
    sorted adjacent gaps. Parts may be zero, so the initial population already
    spans 1-, 2-, and 3-active-drone individuals — the relaxed search space is
    represented from generation 0 rather than relying on mutation to discover it.
    """
    cuts = sorted(random.randint(0, n) for _ in range(k - 1))
    bounds = [0, *cuts, n]
    return [bounds[i + 1] - bounds[i] for i in range(k)]


def _init_individual(n_pois: int, k: int):
    """Random valid individual.

    The genotype is a permutation of **0-based positions** ``0..N-1`` (not POI
    ids) so DEAP's ``cxOrdered`` — which indexes a helper array by gene value —
    stays in range (POI ids are 1..N and would overflow). Positions map to POI ids
    only at decode time. Counts is a random feasible split allowing empty drones.
    """
    perm = list(range(n_pois))
    random.shuffle(perm)
    ind = creator.Individual(perm)
    ind.counts = _random_composition(n_pois, k)
    return ind


def _genotype_to_routes(ind, poi_ids: list[int], depot: int) -> list[list[int]]:
    """Map the 0-based position genotype to POI ids, then decode via the shared
    two-part decoder."""
    order = [poi_ids[g] for g in ind]
    return decode_two_part(order, ind.counts, depot)


def _make_solution(ind, poi_ids: list[int], depot: int) -> Solution:
    routes = _genotype_to_routes(ind, poi_ids, depot)
    makespan, energy = ind.fitness.values
    return Solution(
        routes=tuple(tuple(r) for r in routes),
        makespan=makespan,
        energy=energy,
        genotype=(list(ind), list(ind.counts)),
    )


class NSGA2(Optimizer):
    """Canonical NSGA-II on the two-part chromosome.

    Consumes the shared core only: decode_two_part -> evaluate. The result is a
    RunResult carrying the final non-dominated set and per-generation history, so
    the convergence plot (Phase 6) works with no special-casing.
    """

    def _build_toolbox(self) -> base.Toolbox:
        inst = self.instance
        hp = self.hp
        n_pois, k, depot, dist = inst.n_pois, inst.k, inst.depot, inst.dist
        # Position -> POI id map (every node except the depot).
        poi_ids = [node for node in range(n_pois + 1) if node != depot]
        self._poi_ids = poi_ids

        toolbox = base.Toolbox()
        toolbox.register("individual", _init_individual, n_pois, k)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # The single shared eval counter (CLAUDE.md invariant): every fitness call
        # — initial population and each generation's invalid offspring — flows
        # through this one wrapper, so run()'s n_evals is the *measured* count with
        # no second tally to drift out of sync. MOPSO routes through the same class.
        self._ev = CountingEvaluator(dist)

        def _evaluate(ind):
            # The single shared fitness — identical to the one MOPSO will call.
            routes = _genotype_to_routes(ind, poi_ids, depot)
            return self._ev(routes)

        def _mate(ind1, ind2):
            tools.cxOrdered(ind1, ind2)                      # OX on the perm
            ind1.counts, ind2.counts = _cx_counts(ind1.counts, ind2.counts, n_pois, k)
            return ind1, ind2

        def _mutate(ind):
            _inversion(ind)                                  # reverse a sub-tour
            if random.random() < hp.pmut_counts:
                ind.counts = _mut_counts(ind.counts, n_pois, k)
            return (ind,)

        toolbox.register("evaluate", _evaluate)
        toolbox.register("mate", _mate)
        toolbox.register("mutate", _mutate)
        toolbox.register("select", tools.selNSGA2)
        return toolbox

    @staticmethod
    def _dedup(individuals):
        """Reorder so one representative per objective vector comes first.

        selNSGA2's crowding distance degenerates when a front is full of identical
        objective vectors (zero spacing between clones). Surfacing the unique
        objectives first lets environmental selection keep a spread instead of a
        cloud of duplicates; the duplicates remain available as filler so the pool
        never drops below the population size.
        """
        seen: set = set()
        unique, dups = [], []
        for ind in individuals:
            key = ind.fitness.values
            if key in seen:
                dups.append(ind)
            else:
                seen.add(key)
                unique.append(ind)
        return unique + dups

    @staticmethod
    def _gen_stats(gen: int, pop) -> GenStats:
        vals = np.array([ind.fitness.values for ind in pop])   # (pop, 2)
        return GenStats(
            gen=gen,
            best=tuple(vals.min(axis=0)),     # both objectives minimized
            mean=tuple(vals.mean(axis=0)),
            worst=tuple(vals.max(axis=0)),
        )

    def run(self, seed: int) -> RunResult:
        set_all_seeds(seed)                      # seed BEFORE any RNG use
        toolbox = self._build_toolbox()
        hp = self.hp
        depot = self.instance.depot
        t0 = time.perf_counter()

        pop = toolbox.population(n=hp.pop)
        for ind in pop:
            ind.fitness.values = toolbox.evaluate(ind)
        # Assign crowding distance / rank for the first DCD tournament.
        pop = toolbox.select(pop, hp.pop)

        history = [self._gen_stats(0, pop)]
        for gen in range(1, hp.gens + 1):
            offspring = tools.selTournamentDCD(pop, hp.pop)
            offspring = [toolbox.clone(ind) for ind in offspring]

            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < hp.pcx:
                    toolbox.mate(c1, c2)
                    del c1.fitness.values, c2.fitness.values
            for mutant in offspring:
                if random.random() < hp.pmut:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            invalid = [ind for ind in offspring if not ind.fitness.valid]
            for ind in invalid:
                ind.fitness.values = toolbox.evaluate(ind)

            pop = toolbox.select(self._dedup(pop + offspring), hp.pop)
            history.append(self._gen_stats(gen, pop))

        wall = time.perf_counter() - t0
        front_inds = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
        final_front = [_make_solution(ind, self._poi_ids, depot) for ind in front_inds]
        return RunResult(
            final_front=final_front,
            history=history,
            wall_clock_s=wall,
            n_evals=self._ev.n_calls,    # the single shared counter (one source of truth)
        )
