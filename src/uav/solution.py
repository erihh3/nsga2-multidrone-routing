"""The shared phenotype type.

Both optimizers return ``list[Solution]`` (their final non-dominated set). Every
consumer downstream — metrics, stats, plots — reads *only* this type and never
branches on which algorithm produced it. That single rule is what keeps the
comparison fair and the analysis code trivial.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Solution:
    """One decoded candidate.

    Attributes:
        routes: ``K`` routes, each ``[depot, p_i, ..., depot]`` (the identical
            phenotype both decoders emit).
        makespan: f1, seconds. Mission time bounded by the slowest drone.
        energy: f2, joules. Total fleet energy (Dorling linear model).
        genotype: optional raw genotype, carried only for debugging/repro. Never
            read by metrics/stats/plots.
    """

    routes: tuple[tuple[int, ...], ...]
    makespan: float
    energy: float
    genotype: object | None = field(default=None, compare=False)

    @property
    def objectives(self) -> tuple[float, float]:
        return (self.makespan, self.energy)
