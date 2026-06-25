"""Compare the 2x-iteration MOPSO/DMOPSO runs against their 1x parity baseline.

READS DISK ONLY — re-runs no optimizer. Loads, for seeds 0-4 on eil51-k3:
  * NSGA-II 1x  (fixed reference)      results/eil51-k3_nsga2_<seed>.json
  * MOPSO  1x / DMOPSO 1x (parity)     results/eil51-k3_<algo>_<seed>.json
  * MOPSO  2x / DMOPSO 2x (doubled)    results_2x/2x_eil51-k3_<algo>_<seed>.json

DMOPSO is additionally loaded at 4x (results_2x/4x_...) when present. Builds ONE
union reference front over all arms (via evaluation.aggregate) so HV/IGD/GD/spacing
share a common reference, prints a median table with two headline columns (best
makespan, best energy = the min of each objective over a run's front), reports
Mann-Whitney p across the iteration-scaling pairs (MOPSO 1x->2x and DMOPSO
1x->2x->4x), and writes ONE convergence overlay (MOPSO 1x, NSGA-II 1x, DMOPSO 2x,
DMOPSO 4x), each arm drawn to its own horizon.

Requires the 4x dmopso runs on disk first:
  python scripts/eval_double_iters.py --algorithm dmopso --multiplier 4

Outputs:
  results_2x/2x_eil51-k3_compare.json
  figures/dmopso_iter_scaling_convergence.pdf

Usage:  python scripts/compare_double_iters.py
"""

from __future__ import annotations

import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")  # headless: must precede pyplot
import matplotlib.pyplot as plt
import numpy as np

from uav.evaluation.aggregate import Run, aggregate
from uav.evaluation.stats import mann_whitney, median_iqr
from uav.solution import Solution
from uav.viz.convergence import plot_convergence

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
RESULTS = os.path.join(ROOT, "results")
RESULTS_2X = os.path.join(ROOT, "results_2x")
FIGURES = os.path.join(ROOT, "figures")

INSTANCE = "eil51-k3"
SEEDS = tuple(range(5))

# Reported columns: (key, label, higher_is_better). best_* come straight off the
# front; hv/igd/gd/spacing come from the shared-reference aggregate.
COLUMNS = (
    ("nps", "NPS", True),
    ("best_mk", "best_mk", False),
    ("best_en", "best_en", False),
    ("hv", "HV", True),
    ("igd", "IGD", False),
    ("gd", "GD", False),
    ("spacing", "spacing", False),
)


def _raw_path(algo: str, seed: int, mult: int) -> str:
    if mult == 1:
        return os.path.join(RESULTS, f"{INSTANCE}_{algo}_{seed}.json")
    return os.path.join(RESULTS_2X, f"{mult}x_{INSTANCE}_{algo}_{seed}.json")


def _load_raw(algo: str, seed: int, mult: int) -> dict:
    with open(_raw_path(algo, seed, mult)) as fh:
        return json.load(fh)


def _run_from_raw(data: dict, seed: int) -> Run:
    front = [
        Solution(
            routes=tuple(tuple(int(p) for p in r) for r in routes_i),
            makespan=float(mk),
            energy=float(en),
        )
        for (mk, en), routes_i in zip(data["front"], data["routes"])
    ]
    return Run(seed=seed, front=front,
               wall_clock_s=float(data["wall_clock_s"]), n_evals=int(data["n_evals"]))


def _parity_iters() -> int:
    with open(os.path.join(RESULTS, f"{INSTANCE}_parity.json")) as fh:
        return int(json.load(fh)["iters"])


def _dmopso_mults() -> tuple[int, ...]:
    """DMOPSO iteration multipliers available on disk, sorted: 1x baseline plus every
    ``Nx`` with a full seed set in results_2x/. Picks up new doublings (2x, 4x, 8x,
    ...) with no code edits — just run the optimizer at that Nx beforehand."""
    mults = {1}
    for path in glob.glob(os.path.join(RESULTS_2X, f"*x_{INSTANCE}_dmopso_0.json")):
        m = re.fullmatch(rf"(\d+)x_{re.escape(INSTANCE)}_dmopso_0\.json", os.path.basename(path))
        if m and all(os.path.exists(
                os.path.join(RESULTS_2X, f"{m.group(1)}x_{INSTANCE}_dmopso_{s}.json"))
                for s in SEEDS):
            mults.add(int(m.group(1)))
    return tuple(sorted(mults))


def _fmt(x: float) -> str:
    if x != x:                       # NaN
        return "  nan"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    if abs(x) >= 1:
        return f"{x:.3f}"
    return f"{x:.4f}"


def main() -> None:
    parity = _parity_iters()
    dm = _dmopso_mults()             # discovered dmopso scaling (1x, 2x, 4x, ...)

    # Arm spec: label -> (algo, iters-multiplier). Order fixes table rows.
    arms = {
        "NSGA-II (ref)": ("nsga2", 1),
        f"MOPSO 1x({parity})": ("mopso", 1),
        f"MOPSO 2x({2 * parity})": ("mopso", 2),
    }
    for m in dm:
        arms[f"DMOPSO {m}x({m * parity})"] = ("dmopso", m)

    raws = {lab: [_load_raw(algo, s, m) for s in SEEDS] for lab, (algo, m) in arms.items()}
    runs_by_algo = {lab: [_run_from_raw(raws[lab][i], s) for i, s in enumerate(SEEDS)]
                    for lab in arms}

    # One shared reference front over ALL arms -> comparable hv/igd/gd/spacing.
    agg = aggregate(runs_by_algo)

    # Per-arm metric arrays (over seeds): best_* off the front + geometric from agg.
    per_arm: dict[str, dict[str, np.ndarray]] = {}
    for lab in arms:
        rows = agg["per_run"][lab]                       # aligned to SEEDS order
        fronts = [np.asarray(r["front"], dtype=float) for r in raws[lab]]
        per_arm[lab] = {
            "nps": np.array([r["nps"] for r in rows]),
            "best_mk": np.array([f[:, 0].min() for f in fronts]),
            "best_en": np.array([f[:, 1].min() for f in fronts]),
            "hv": np.array([r["hv"] for r in rows]),
            "igd": np.array([r["igd"] for r in rows]),
            "gd": np.array([r["gd"] for r in rows]),
            "spacing": np.array([r["spacing"] for r in rows]),
        }

    # --- table ------------------------------------------------------------------
    print("\n" + "=" * 102)
    print(f"{INSTANCE}: iteration-scaling side-experiment  (median +- IQR, n={len(SEEDS)} seeds 0-4)")
    scaling = ", ".join(f"{m}x={m * parity}" for m in dm if m > 1)
    print(f"parity iters={parity}  |  dmopso scaling: {scaling}  |  "
          f"reference front: {agg['reference_size']} pts (union of all arms)")
    print("NOTE: 2x/4x/... break budget parity with NSGA-II — exploratory, not for the paper table.")
    print("=" * 102)
    hdr = f"{'arm':<18}" + "".join(f"{lab:>12}" for _, lab, _ in COLUMNS)
    print(hdr)
    print("-" * len(hdr))
    for lab in arms:
        cells = []
        for key, _, _ in COLUMNS:
            med, _ = median_iqr(per_arm[lab][key])
            cells.append(f"{_fmt(med):>12}")
        print(f"{lab:<18}" + "".join(cells))
    print("-" * len(hdr))
    dirs = "  ".join(f"{lab}{'+' if hb else '-'}" for _, lab, hb in COLUMNS)
    print(f"dir (+ higher better / - lower better): {dirs}")

    # --- Mann-Whitney across the iteration-scaling pairs -------------------------
    # MOPSO 1x->2x, each consecutive dmopso doubling, and the overall 1x->max.
    comparisons = [("MOPSO  1x->2x", f"MOPSO 1x({parity})", f"MOPSO 2x({2 * parity})")]
    for prev, cur in zip(dm, dm[1:]):
        comparisons.append((f"DMOPSO {prev}x->{cur}x",
                            f"DMOPSO {prev}x({prev * parity})", f"DMOPSO {cur}x({cur * parity})"))
    if len(dm) > 2:
        comparisons.append((f"DMOPSO {dm[0]}x->{dm[-1]}x",
                            f"DMOPSO {dm[0]}x({dm[0] * parity})", f"DMOPSO {dm[-1]}x({dm[-1] * parity})"))
    pvals: dict[str, dict[str, float]] = {}
    print("\nMann-Whitney U p-values per metric (small p => significant change):")
    for name, la, lb in comparisons:
        pvals[name] = {}
        cells = []
        for key, lab, _ in COLUMNS:
            _, p = mann_whitney(per_arm[la][key], per_arm[lb][key])
            pvals[name][key] = p
            cells.append(f"{lab}={_fmt(p)}")
        print(f"  {name}: " + "  ".join(cells))

    # --- persist comparison JSON ------------------------------------------------
    os.makedirs(RESULTS_2X, exist_ok=True)
    table = {lab: {key: dict(zip(("median", "iqr"), median_iqr(per_arm[lab][key])))
                   for key, _, _ in COLUMNS}
             for lab in arms}
    out = {"instance": INSTANCE, "n_seeds": len(SEEDS), "seeds": list(SEEDS),
           "parity_iters": parity,
           "dmopso_scaling_iters": {f"{m}x": m * parity for m in dm},
           "reference_size": agg["reference_size"], "table": table, "pvalues": pvals}
    cmp_path = os.path.join(RESULTS_2X, f"2x_{INSTANCE}_compare.json")
    with open(cmp_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nsaved: {os.path.relpath(cmp_path, ROOT)}")

    # --- single convergence overlay: MOPSO 1x, NSGA-II 1x + the two deepest dmopso --
    # Shows dmopso's budget progression against the plateaued MOPSO baseline and the
    # NSGA-II target; each arm is drawn to its own horizon, so the deepest dmopso curve
    # extends furthest (whether it keeps descending or flattens). Capped at 4 arms to
    # respect the convergence palette (4 colors); we keep the two largest dmopso runs.
    os.makedirs(FIGURES, exist_ok=True)
    top_dm = [m for m in dm if m > 1][-2:]           # two deepest dmopso multipliers
    hist_by_algo = {
        f"MOPSO 1x({parity})": [r["history"] for r in raws[f"MOPSO 1x({parity})"]],
        "NSGA-II 1x": [r["history"] for r in raws["NSGA-II (ref)"]],
    }
    for m in top_dm:
        hist_by_algo[f"DMOPSO {m}x({m * parity})"] = [
            r["history"] for r in raws[f"DMOPSO {m}x({m * parity})"]]
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 7.0))
    plot_convergence(hist_by_algo, axes=axes)
    fig.tight_layout()
    fig_path = os.path.join(FIGURES, "dmopso_iter_scaling_convergence.pdf")
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"saved: {os.path.relpath(fig_path, ROOT)}")


if __name__ == "__main__":
    main()
