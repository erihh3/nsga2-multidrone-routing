"""Best/mean/worst objective vs generation, seed-averaged with a shaded IQR band
(single-run curves are noise). Phase 6.

Two stacked panels (makespan on top, energy below). For each arm and objective we
stack the per-seed curves and plot the seed-**median** with a shaded 25-75%
IQR band — never a single seed (CLAUDE.md). Co-equality: both arms go through the
identical code; an arm contributes only a label/color. The two arms may have
different curve lengths (NSGA-II runs `gens`; MOPSO runs its measured-parity
`iters`), so each is plotted to its own length against the generation index.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: must precede pyplot
import matplotlib.pyplot as plt
import numpy as np

# Per-arm line color, assigned in call order (the only per-algorithm visual).
_ALGO_COLORS: tuple[str, ...] = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd")
_OBJECTIVES: tuple[str, ...] = ("makespan $f_1$ (s)", "total energy $f_2$ (J)")


def _stack(histories: list[list[dict]], stat: str, obj: int) -> np.ndarray:
    """Stack one statistic of one objective over seeds -> ``(n_seeds, n_gens)``.

    ``histories`` is a list over seeds of a run's ``history`` (per-gen dicts with
    ``best``/``mean``/``worst``, each a ``[makespan, energy]`` pair). Seeds of one
    arm share a generation count, so the rows align into a rectangular array.
    """
    return np.array([[g[stat][obj] for g in h] for h in histories], dtype=float)


def plot_convergence(histories_by_algo, axes=None):
    """Seed-median convergence with an IQR band, two stacked objective panels.

    Args:
        histories_by_algo: ``{label: list_over_seeds[ list_of_gen_dicts ]}`` where
            each gen dict has ``best``/``mean``/``worst`` as ``[makespan, energy]``.
        axes: optional length-2 array/list of Axes (makespan, energy); created if
            omitted.

    Returns:
        The length-2 array of Axes drawn into.
    """
    if axes is None:
        _, axes = plt.subplots(2, 1, figsize=(6.5, 7.0), sharex=False)
    axes = np.atleast_1d(axes)

    for obj, ax in enumerate(axes):
        for i, (label, histories) in enumerate(histories_by_algo.items()):
            if not histories:
                continue
            color = _ALGO_COLORS[i % len(_ALGO_COLORS)]
            mean = _stack(histories, "mean", obj)
            best = _stack(histories, "best", obj)
            gens = np.arange(mean.shape[1])

            # Seed-median central line + 25-75% IQR band on the population mean.
            med = np.median(mean, axis=0)
            lo, hi = np.percentile(mean, [25, 75], axis=0)
            ax.plot(gens, med, color=color, linewidth=1.6, label=f"{label} (mean)")
            ax.fill_between(gens, lo, hi, color=color, alpha=0.18, linewidth=0)
            # Seed-median best-so-far as a thinner dashed reference.
            ax.plot(np.arange(best.shape[1]), np.median(best, axis=0),
                    color=color, linewidth=1.0, linestyle="--", alpha=0.9,
                    label=f"{label} (best)")

        ax.set_ylabel(_OBJECTIVES[obj])
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("generation / iteration")
    axes[0].legend(fontsize=8, ncol=2)
    # Derive the seed count from the data so the label never goes stale on a
    # seed-count change (it read a hardcoded "10" through the n=30 rerun).
    n_seeds = max((len(h) for h in histories_by_algo.values()), default=0)
    axes[0].set_title(f"Convergence (seed-median, shaded IQR over {n_seeds} seeds)")
    return axes
