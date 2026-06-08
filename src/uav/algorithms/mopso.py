"""MOPSO (Coello Coello 2004 style), continuous random-key particles.

- particle position = random-key vector (length N+K), decoded via decode_random_key
- external archive of non-dominated solutions (bounded, e.g. 100)
- adaptive grid / hypercube over objective space for leader selection + archive
  truncation (leaders drawn from sparse cells preserve diversity)
- velocity update: inertia + cognitive + social (leader from archive);
  turbulence/mutation operator decaying over iterations
- same evaluation budget as NSGA-II (swarm*iters == pop*gens)

The genotype + this loop are the *only* things that differ from NSGA-II.
Everything downstream (decode_random_key, evaluate) is the shared core, called
unchanged — there is no PSO-specific objective or distance code here. The only
PSO-local logic is selection bookkeeping (Pareto domination, the adaptive grid,
pbest/archive maintenance), which is the optimizer's allowed point of difference,
exactly as NSGA-II's selNSGA2 carries its own domination/crowding.

Design decisions (Coello Coello, Pulido & Lechuga 2004, "Handling multiple
objectives with PSO", IEEE TEVC — unverified DOI):
- Positions live in the [0,1]^(N+K) random-key box. Velocity is clamped to a
  fraction of that range; on a boundary violation the position is clamped and the
  offending velocity component flipped (reflective bound) so particles search
  inward instead of piling on the walls.
- Leader (global guide) per particle is drawn from the external archive via the
  adaptive grid: cells are scored 10/|cell| so sparse regions are favoured, which
  spreads the swarm along the front rather than collapsing onto one knee.
- The adaptive grid recomputes per-objective min/max from the *archive itself*
  each iteration. That is an implicit normalization: the energy axis spans only
  ~1% in absolute terms, but dividing the archive's own range keeps makespan from
  being the sole driver of the grid — diversity is measured on both axes.
- Turbulence (diversity kick) resets a random subset of a particle's keys; the
  fraction of the swarm it touches decays linearly to 0 over the run, so the
  search explores early and exploits late.

HIGHEST RISK module. Pre-committed fallback (the *only* sanctioned rescue): if the
front is poor after focused debugging, bolt 2-opt local search onto decoded routes
and report honestly as "MOPSO + 2-opt". That raises a co-equality question, so it
is *not* improvised here — it must be raised with the user first.

NOTE (scope): the shared CountingEvaluator wrapper and measured budget parity vs
NSGA-II (~46.5k evals, not the nominal swarm*iters) are deferred to the start of
Phase 4. Here MOPSO tallies n_evals locally, exactly as NSGA-II currently does.
"""

from __future__ import annotations

import random
import time

import numpy as np

from uav.algorithms.base import GenStats, Optimizer, RunResult
from uav.problem.decode import decode_random_key
from uav.problem.fitness import evaluate
from uav.seeds import set_all_seeds
from uav.solution import Solution

# An archive/swarm member is a (position, objectives, routes) triple.
#   position   : np.ndarray, length N+K
#   objectives : (makespan, energy)
#   routes     : list[list[int]] decoded phenotype


# --- Pareto domination (minimization) ------------------------------------------

def _dominates(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """True iff objective vector ``a`` dominates ``b`` (both objectives minimized)."""
    return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])


def _nondominated(members: list) -> list:
    """Keep only non-dominated members, deduplicated by objective vector.

    O(n^2), which is nothing at archive sizes ~100-200. Duplicate objective
    vectors are collapsed to one representative so identical points don't crowd
    the grid (the analogue of NSGA-II's _dedup before environmental selection).
    """
    keep: list = []
    seen: set = set()
    for i, m in enumerate(members):
        oi = m[1]
        if any(_dominates(n[1], oi) for j, n in enumerate(members) if j != i):
            continue
        if oi not in seen:
            seen.add(oi)
            keep.append(m)
    return keep


# --- Adaptive grid (hypercube) -------------------------------------------------

def _grid_coords(objs: np.ndarray, divisions: int) -> np.ndarray:
    """Map objective vectors to integer grid cells over the set's own min/max.

    Using the archive's own per-objective range is the implicit normalization the
    thin energy axis needs. A degenerate (zero-width) axis collapses to cell 0.
    """
    mins = objs.min(axis=0)
    maxs = objs.max(axis=0)
    ranges = maxs - mins
    coords = np.zeros_like(objs, dtype=int)
    for d in range(objs.shape[1]):
        if ranges[d] <= 0:
            coords[:, d] = 0
        else:
            c = np.floor((objs[:, d] - mins[d]) / ranges[d] * divisions).astype(int)
            coords[:, d] = np.clip(c, 0, divisions - 1)
    return coords


def _leader_sampler(archive: list, divisions: int) -> tuple[list, np.ndarray]:
    """Precompute (cell member-index groups, selection probabilities) once per
    iteration. Cell fitness = 10/|cell| favours sparse cells (Coello 2004)."""
    objs = np.array([m[1] for m in archive])
    coords = _grid_coords(objs, divisions)
    cells: dict = {}
    for idx, c in enumerate(coords):
        cells.setdefault(tuple(c), []).append(idx)
    groups = list(cells.values())
    fit = np.array([10.0 / len(g) for g in groups])
    return groups, fit / fit.sum()


def _draw_leader(archive: list, groups: list, probs: np.ndarray):
    """Roulette a grid cell (sparse-favoured), then a uniform member of it."""
    g = groups[int(np.random.choice(len(groups), p=probs))]
    return archive[random.choice(g)]


def _truncate(archive: list, max_size: int, divisions: int) -> list:
    """Drop members from the most crowded cell until |archive| <= max_size."""
    while len(archive) > max_size:
        objs = np.array([m[1] for m in archive])
        coords = _grid_coords(objs, divisions)
        cells: dict = {}
        for idx, c in enumerate(coords):
            cells.setdefault(tuple(c), []).append(idx)
        crowded = max(cells.values(), key=len)
        archive.pop(random.choice(crowded))
    return archive


def _turbulence(pos: np.ndarray) -> None:
    """In-place diversity kick: reset a random ~10% of the keys to U(0,1)."""
    d = pos.shape[0]
    n_mut = max(1, int(0.1 * d))
    dims = np.random.choice(d, size=n_mut, replace=False)
    pos[dims] = np.random.uniform(0.0, 1.0, size=n_mut)


class MOPSO(Optimizer):
    """Coello Coello 2004 MOPSO on the random-key genotype.

    Consumes the shared core only: decode_random_key -> evaluate. Returns the same
    RunResult shape as NSGA-II (final non-dominated set + per-iteration history),
    so the convergence plot (Phase 6) works with no special-casing.
    """

    def _rebuild_archive(self, members: list) -> list:
        """Non-dominated filter + grid truncation to archive_size."""
        archive = _nondominated(members)
        return _truncate(archive, self.hp.archive_size, self.hp.grid_divisions)

    @staticmethod
    def _gen_stats(gen: int, swarm_objs: list) -> GenStats:
        """Per-iteration best/mean/worst per objective over the *swarm* (not the
        archive) — mirrors NSGA-II's _gen_stats(pop) so the convergence plot is
        directly comparable."""
        vals = np.array(swarm_objs)               # (swarm, 2)
        return GenStats(
            gen=gen,
            best=tuple(vals.min(axis=0)),         # both objectives minimized
            mean=tuple(vals.mean(axis=0)),
            worst=tuple(vals.max(axis=0)),
        )

    def run(self, seed: int) -> RunResult:
        set_all_seeds(seed)                        # seed BEFORE any RNG use
        inst, hp = self.instance, self.hp
        n, k, depot, dist = inst.n_pois, inst.k, inst.depot, inst.dist
        dim = n + k
        swarm, iters = hp.swarm, hp.iters
        vmax = hp.vmax_frac                        # key range is 1.0, so vmax = frac
        t0 = time.perf_counter()
        n_evals = 0

        # --- init swarm: positions in the [0,1] random-key box, zero velocity ---
        pos = np.random.uniform(0.0, 1.0, size=(swarm, dim))
        vel = np.zeros((swarm, dim))

        swarm_objs: list = []
        members: list = []
        for i in range(swarm):
            routes = decode_random_key(pos[i], n, k, depot)
            objs = evaluate(routes, dist)
            n_evals += 1
            swarm_objs.append(objs)
            members.append((pos[i].copy(), objs, routes))

        pbest_pos = pos.copy()
        pbest_objs = list(swarm_objs)
        archive = self._rebuild_archive(members)
        history = [self._gen_stats(0, swarm_objs)]

        # --- main loop ---------------------------------------------------------
        for t in range(1, iters + 1):
            mut_frac = hp.mut_rate * (1.0 - t / iters)     # decays to 0
            groups, probs = _leader_sampler(archive, hp.grid_divisions)

            for i in range(swarm):
                leader = _draw_leader(archive, groups, probs)
                r1 = np.random.random(dim)
                r2 = np.random.random(dim)
                vel[i] = (
                    hp.w_inertia * vel[i]
                    + hp.c1 * r1 * (pbest_pos[i] - pos[i])
                    + hp.c2 * r2 * (leader[0] - pos[i])
                )
                np.clip(vel[i], -vmax, vmax, out=vel[i])
                pos[i] += vel[i]
                # Reflective bound: clamp to [0,1], flip velocity on violated dims.
                out = (pos[i] < 0.0) | (pos[i] > 1.0)
                np.clip(pos[i], 0.0, 1.0, out=pos[i])
                vel[i][out] *= -1.0
                if np.random.random() < mut_frac:
                    _turbulence(pos[i])

            # Evaluate the moved swarm, update pbests, grow the archive.
            swarm_objs = []
            new_members = []
            for i in range(swarm):
                routes = decode_random_key(pos[i], n, k, depot)
                objs = evaluate(routes, dist)
                n_evals += 1
                swarm_objs.append(objs)
                new_members.append((pos[i].copy(), objs, routes))
                if _dominates(objs, pbest_objs[i]):
                    pbest_objs[i], pbest_pos[i] = objs, pos[i].copy()
                elif not _dominates(pbest_objs[i], objs) and random.random() < 0.5:
                    # Mutually non-dominated: replace with probability 0.5.
                    pbest_objs[i], pbest_pos[i] = objs, pos[i].copy()

            archive = self._rebuild_archive(archive + new_members)
            history.append(self._gen_stats(t, swarm_objs))

        wall = time.perf_counter() - t0
        final_front = [
            Solution(
                routes=tuple(tuple(r) for r in routes),
                makespan=objs[0],
                energy=objs[1],
                genotype=position.tolist(),
            )
            for position, objs, routes in archive
        ]
        return RunResult(
            final_front=final_front,
            history=history,
            wall_clock_s=wall,
            n_evals=n_evals,
        )
