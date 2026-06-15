"""Discrete (swap-sequence) MOPSO — the encoding-diagnostic variant (Attempt C).

This is a deliberate, scoped *variant* of MOPSO, not a third baseline algorithm.
The MOPSO investigation isolated the random-key collapse twice (tuning can't fix
it, reachability can't fix it); the last suspect is the **random-key + argsort
order encoding**. This arm tests that directly by replacing *only* the order
representation with a permutation-native swap-sequence PSO (Clerc-style discrete
PSO — citation unverified), leaving every other axis byte-for-byte shared.

The one-axis contract (CLAUDE.md co-equality): this differs from NSGA-II and
random-key MOPSO in **exactly one axis — the order genotype and its velocity/
position update**. Everything else is the shared, untouched core:
- the count axis stays the existing continuous cut-keys, decoded by the SAME
  ``_counts_from_keys`` (so a positive/negative result is attributable to the
  order encoding alone, and the phenotype space is identical by construction);
- ``decode_discrete`` funnels into the SAME ``_partition_to_routes`` phenotype;
- the SAME shared ``CountingEvaluator`` (one eval per particle per iteration, so
  the measured count is the deterministic ``swarm * (iters + 1)`` — identical to
  random-key MOPSO, so the measured-eval parity budget is reused verbatim);
- the external archive, adaptive hypercube grid, sparse-cell leader selection,
  crowding truncation and turbulence *schedule* are imported from ``mopso`` and
  reused **unchanged** — they operate on objective vectors only, so they are
  algorithm-agnostic.

No route local search (no 2-opt) anywhere — that would turn this into a memetic
study and is out of scope.

Swap-sequence algebra (Clerc-style; convention documented per operator below):
a particle's order is a genuine permutation; its order-"velocity" is a swap
sequence (an ordered list of transpositions). Applying any swap sequence to a
permutation yields a permutation, so the order is valid by construction and never
needs repair.
"""

from __future__ import annotations

import random
import time

import numpy as np

from uav.algorithms.base import GenStats, Optimizer, RunResult
# Reused UNCHANGED — all operate on objective vectors / genotype-agnostic state.
from uav.algorithms.mopso import (
    _dominates,
    _draw_leader,
    _leader_sampler,
    _nondominated,
    _truncate,
    _turbulence,            # continuous turbulence on the cut-keys (count axis)
    _turbulence_fraction,   # the identical decaying-and-floored schedule
)
from uav.problem.decode import decode_discrete
from uav.problem.fitness import CountingEvaluator
from uav.seeds import set_all_seeds
from uav.solution import Solution

# A swap operator is a transposition (i, j); a swap sequence (SS) is an ordered
# list of them. An archive/swarm member is a (genotype, objectives, routes)
# triple, with genotype = (perm, kappa): perm a list[int] permutation of 0-based
# positions, kappa an np.ndarray of K continuous cut-keys.


# --- swap-sequence operators (pure functions) ----------------------------------

def swap_op(i: int, j: int) -> tuple[int, int]:
    """A single swap operator: exchange positions ``i`` and ``j``."""
    return (i, j)


def apply_ss(perm, ss) -> list[int]:
    """Fold-left apply a swap sequence to ``perm``; return a NEW permutation.

    Applying transpositions to a permutation always yields a permutation, so the
    result is valid by construction — the whole point of the swap encoding.
    """
    out = list(perm)
    for i, j in ss:
        out[i], out[j] = out[j], out[i]
    return out


def difference(a, b) -> list[tuple[int, int]]:
    """A basic swap sequence ``SS`` such that ``apply_ss(b, SS) == a``.

    Selection-sort construction: walk left to right, and whenever ``b`` disagrees
    with ``a`` at position ``i``, swap ``a[i]`` into place. Round-trip invariant
    (tested): applying this sequence to ``b`` reproduces ``a`` exactly.
    """
    work = list(b)
    pos = {v: idx for idx, v in enumerate(work)}   # value -> current index
    ss: list[tuple[int, int]] = []
    for i in range(len(a)):
        if work[i] == a[i]:
            continue
        j = pos[a[i]]
        ss.append((i, j))
        vi, vj = work[i], work[j]
        work[i], work[j] = vj, vi
        pos[vi], pos[vj] = j, i
    return ss


def scale_ss(alpha: float, ss) -> list[tuple[int, int]]:
    """Coefficient scaling ``alpha (x) SS``: keep each swap independently with
    probability ``min(alpha, 1)``.

    Chosen convention: a scalar coefficient is read as a per-swap survival
    probability. PSO coefficients exceed 1 (e.g. c1=1.5), so the probability is
    clamped at 1 — ``alpha >= 1`` keeps the whole sequence; ``alpha <= 0`` drops
    it. Draws use the seeded ``random`` RNG, consistent with the count axis.
    """
    p = min(alpha, 1.0)
    if p >= 1.0:
        return list(ss)
    return [s for s in ss if random.random() < p]


def concat_ss(ss1, ss2) -> list[tuple[int, int]]:
    """Concatenation ``SS_1 (+) SS_2``: apply ``ss1`` then ``ss2``."""
    return list(ss1) + list(ss2)


def clamp_ss(ss, max_len: int) -> list[tuple[int, int]]:
    """Velocity clamp: cap a swap sequence at ``max_len`` swaps (keep the first).

    The discrete analogue of MOPSO's per-dimension velocity clamp (``vmax`` in
    ``mopso.py``). Without it the composed order-velocity grows to ~2N swaps per
    step under the shared continuous coefficients (c1=1.5, c2=2.0 saturate the
    keep-probability at 1), which re-randomizes a length-N permutation every
    iteration and prevents convergence. Capping each pull to a small ``vmax``
    keeps movement bounded and informative, exactly as ``vmax`` does on the
    continuous count axis.
    """
    return list(ss[:max_len])


def _turbulence_perm(perm: list[int], n_swaps: int) -> None:
    """Discrete diversity kick: apply ``n_swaps`` random transpositions in place.

    The order analogue of MOPSO's continuous ``_turbulence`` (random key resets),
    fired on the SAME decaying-and-floored schedule so turbulence is not a
    confound between the two arms.
    """
    d = len(perm)
    for _ in range(n_swaps):
        i, j = random.randrange(d), random.randrange(d)
        perm[i], perm[j] = perm[j], perm[i]


class DiscreteMOPSO(Optimizer):
    """Swap-sequence MOPSO on a hybrid (permutation + continuous cut-keys) genotype.

    Order part: permutation evolved by swap-sequence velocity (the variable under
    test). Count part: the EXISTING continuous random-key update with reflective
    bounds, unchanged. Returns the same ``RunResult`` shape as the other arms.
    """

    def _rebuild_archive(self, members: list) -> list:
        """Non-dominated filter + grid truncation (reused MOPSO machinery)."""
        archive = _nondominated(members)
        return _truncate(archive, self.hp.archive_size, self.hp.grid_divisions)

    @staticmethod
    def _gen_stats(gen: int, swarm_objs: list) -> GenStats:
        """Per-iteration best/mean/worst per objective over the swarm — identical
        to MOPSO/NSGA-II so the convergence plot is directly comparable."""
        vals = np.array(swarm_objs)
        return GenStats(
            gen=gen,
            best=tuple(vals.min(axis=0)),
            mean=tuple(vals.mean(axis=0)),
            worst=tuple(vals.max(axis=0)),
        )

    def run(self, seed: int) -> RunResult:
        set_all_seeds(seed)                        # seed BEFORE any RNG use
        inst, hp = self.instance, self.hp
        n, k, depot, dist = inst.n_pois, inst.k, inst.depot, inst.dist
        swarm, iters = hp.swarm, hp.iters
        vmax = hp.vmax_frac                        # cut-key range is 1.0
        # Discrete velocity clamp: each order pull (inertia/cognitive/social) is
        # capped at this many swaps — the per-pull analogue of the continuous
        # per-dimension vmax. Total order movement <= 3*vmax_swaps per step.
        vmax_swaps = max(1, round(hp.vmax_frac * n))
        n_swaps = max(1, int(0.1 * n))             # turbulence kick size (order)
        t0 = time.perf_counter()

        # The single shared eval counter (CLAUDE.md invariant): same wrapper class
        # both other arms use; every decode->evaluate flows through it.
        ev = CountingEvaluator(dist)

        # --- init swarm: random permutations + cut-keys in [0,1], zero velocity --
        perms = [list(map(int, np.random.permutation(n))) for _ in range(swarm)]
        kappa = np.random.uniform(0.0, 1.0, size=(swarm, k))
        v_order: list[list[tuple[int, int]]] = [[] for _ in range(swarm)]
        v_count = np.zeros((swarm, k))

        swarm_objs: list = []
        members: list = []
        for i in range(swarm):
            routes = decode_discrete(perms[i], kappa[i], n, k, depot)
            objs = ev(routes)
            swarm_objs.append(objs)
            members.append(((perms[i], kappa[i].copy()), objs, routes))

        pbest_perm = [list(p) for p in perms]
        pbest_kappa = kappa.copy()
        pbest_objs = list(swarm_objs)
        archive = self._rebuild_archive(members)
        history = [self._gen_stats(0, swarm_objs)]

        # --- main loop ---------------------------------------------------------
        for t in range(1, iters + 1):
            mut_frac = _turbulence_fraction(t, iters, hp.mut_rate, hp.mut_floor)
            groups, probs = _leader_sampler(archive, hp.grid_divisions)

            for i in range(swarm):
                leader = _draw_leader(archive, groups, probs)
                leader_perm, leader_kappa = leader[0]

                # Order part (swap sequences). Scalar r per particle per term.
                # Each pull is clamped to vmax_swaps (the discrete velocity bound)
                # so movement stays bounded and all three pulls are represented.
                r1, r2 = random.random(), random.random()
                v_inertia = clamp_ss(scale_ss(hp.w_inertia, v_order[i]), vmax_swaps)
                v_cog = clamp_ss(scale_ss(hp.c1 * r1, difference(pbest_perm[i], perms[i])), vmax_swaps)
                v_soc = clamp_ss(scale_ss(hp.c2 * r2, difference(leader_perm, perms[i])), vmax_swaps)
                v_order[i] = concat_ss(concat_ss(v_inertia, v_cog), v_soc)
                new_perm = apply_ss(perms[i], v_order[i])

                # Count part (continuous — the EXISTING MOPSO update, unchanged).
                r1k = np.random.random(k)
                r2k = np.random.random(k)
                v_count[i] = (
                    hp.w_inertia * v_count[i]
                    + hp.c1 * r1k * (pbest_kappa[i] - kappa[i])
                    + hp.c2 * r2k * (leader_kappa - kappa[i])
                )
                np.clip(v_count[i], -vmax, vmax, out=v_count[i])
                kappa[i] += v_count[i]
                out = (kappa[i] < 0.0) | (kappa[i] > 1.0)
                np.clip(kappa[i], 0.0, 1.0, out=kappa[i])
                v_count[i][out] *= -1.0

                # Turbulence (single gate, same schedule): order = random swaps,
                # counts = the existing continuous key reset, both unchanged in role.
                if random.random() < mut_frac:
                    _turbulence_perm(new_perm, n_swaps)
                    _turbulence(kappa[i])

                perms[i] = new_perm

            # Evaluate the moved swarm, update pbests, grow the archive.
            swarm_objs = []
            new_members = []
            for i in range(swarm):
                routes = decode_discrete(perms[i], kappa[i], n, k, depot)
                objs = ev(routes)
                swarm_objs.append(objs)
                new_members.append(((perms[i], kappa[i].copy()), objs, routes))
                if _dominates(objs, pbest_objs[i]):
                    pbest_objs[i] = objs
                    pbest_perm[i] = list(perms[i])
                    pbest_kappa[i] = kappa[i].copy()
                elif not _dominates(pbest_objs[i], objs) and random.random() < 0.5:
                    # Mutually non-dominated: replace with probability 0.5.
                    pbest_objs[i] = objs
                    pbest_perm[i] = list(perms[i])
                    pbest_kappa[i] = kappa[i].copy()

            archive = self._rebuild_archive(archive + new_members)
            history.append(self._gen_stats(t, swarm_objs))

        wall = time.perf_counter() - t0
        final_front = [
            Solution(
                routes=tuple(tuple(r) for r in routes),
                makespan=objs[0],
                energy=objs[1],
                genotype=(list(geno[0]), geno[1].tolist()),
            )
            for geno, objs, routes in archive
        ]
        return RunResult(
            final_front=final_front,
            history=history,
            wall_clock_s=wall,
            n_evals=ev.n_calls,    # the single shared counter (one source of truth)
        )
