"""Reference front + objective normalization.

Reference front: union of all non-dominated solutions across both algorithms x
all seeds for an instance, re-filtered to non-dominated. Standard proxy when no
true front exists — disclose it in the paper.

Normalization: makespan (tens-hundreds of s) and energy (large J) live on wildly
different scales; un-normalized hypervolume is dominated by energy. Normalize
each objective to [0,1] using the reference front's per-objective min/max before
any volume/spacing metric.

Phase 4. Stub.
"""

from __future__ import annotations

from uav.solution import Solution


def build_reference_front(fronts: list[list[Solution]]) -> list[Solution]:
    raise NotImplementedError("Phase 4: union + non-dominated re-filter.")
