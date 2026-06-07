"""The 80-run harness: instances x algorithms x seeds.

Each run writes results/<instance>_<algo>_<seed>.json (final front + history +
hyperparams + wall-clock) and appends a row to results/log.csv. If multiprocessing
is used, set seeds inside each worker and keep mapped functions top-level.

Phase 5. Stub.
"""

from __future__ import annotations


def run_all() -> None:
    raise NotImplementedError("Phase 5: 4x2x10 harness + logging.")
