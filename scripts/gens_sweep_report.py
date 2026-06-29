"""Report the NSGA-II generation-scaling sweep: table + Pareto + convergence.

NSGA-II counterpart to ``scripts/swarm_iters_report.py``. READS ``tmp_results/`` ONLY.
Auto-discovers every ``genssweep_<instance>_nsga2_g<gens>_<seed>.json`` written by
``scripts/eval_gens_sweep.py`` (so a subset reports just that subset), builds ONE
shared union reference front over all gens levels (via ``evaluation.aggregate``) so
HV/IGD/GD/spacing are comparable, and emits:

  * a printed metric table (rows = gens levels; headline best-makespan/best-energy
    columns) + ``tmp_results/genssweep_<instance>_metrics.json``;
  * ONE Pareto overlay  ``tmp_results/genssweep_<instance>_pareto.pdf``;
  * ONE convergence overlay ``tmp_results/genssweep_<instance>_convergence.pdf``.

Single algorithm => 3 gens levels stay within the 4-color (convergence) / 4-marker
(pareto) palettes, so one figure each suffices.

CAVEATS. (1) Off-parity, n=2 seeds: read TRENDS, not significance; not for the paper.
(2) The convergence BOLD line is the population MEAN of each objective. For a Pareto
method the mean of one objective does NOT converge to the minimum — it rises as the
front spreads across the trade-off (the variable-fleet 1/2/3-drone span). Read the
thin DASHED best line for true convergence; the mean line "rising" is diversity, not
divergence.

Usage:  python scripts/gens_sweep_report.py
        python scripts/gens_sweep_report.py --instance eil51-k3
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")  # headless: must precede pyplot
import matplotlib.pyplot as plt
import numpy as np

from uav.evaluation.aggregate import Run, aggregate
from uav.evaluation.stats import median_iqr
from uav.solution import Solution
from uav.viz.convergence import plot_convergence
from uav.viz.pareto import plot_pareto

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
TMP = os.path.join(ROOT, "tmp_results")
PREFIX = "genssweep_"
DEFAULT_INSTANCE = "eil51-k3"

# Reported columns: (key, label, higher_is_better). best_* off the raw front; the
# rest from the shared-reference aggregate (mirrors swarm_iters_report).
COLUMNS = (
    ("nps", "NPS", True),
    ("best_mk", "best_mk", False),
    ("best_en", "best_en", False),
    ("hv", "HV", True),
    ("igd", "IGD", False),
    ("gd", "GD", False),
    ("spacing", "spacing", False),
)


def _discover(instance: str) -> dict[int, list[int]]:
    """Map gens -> sorted seed list present on disk for this instance."""
    pat = re.compile(rf"{re.escape(PREFIX)}{re.escape(instance)}_nsga2_g(\d+)_(\d+)\.json$")
    found: dict[int, list[int]] = {}
    for p in glob.glob(os.path.join(TMP, f"{PREFIX}{instance}_nsga2_g*_*.json")):
        m = pat.search(os.path.basename(p))
        if m:
            found.setdefault(int(m.group(1)), []).append(int(m.group(2)))
    return {g: sorted(v) for g, v in found.items()}


def _raw_path(instance: str, gens: int, seed: int) -> str:
    return os.path.join(TMP, f"{PREFIX}{instance}_nsga2_g{gens}_{seed}.json")


def _load_raw(instance: str, gens: int, seed: int) -> dict:
    with open(_raw_path(instance, gens, seed)) as fh:
        return json.load(fh)


def _run_from_raw(data: dict, seed: int) -> Run:
    """Rebuild a Run (Solution fronts) from a persisted JSON (n_active recomputed
    inside Solution from routes, never read off a JSON key)."""
    front = [
        Solution(routes=tuple(tuple(int(p) for p in r) for r in routes_i),
                 makespan=float(mk), energy=float(en))
        for (mk, en), routes_i in zip(data["front"], data["routes"])
    ]
    return Run(seed=seed, front=front,
               wall_clock_s=float(data["wall_clock_s"]), n_evals=int(data["n_evals"]))


def _records(data: dict) -> list[tuple]:
    """(makespan, energy, n_active) points for the Pareto plot (fleet from routes)."""
    return [(obj[0], obj[1], sum(1 for r in routes if len(r) > 2))
            for obj, routes in zip(data["front"], data["routes"])]


def _fmt(x: float) -> str:
    if x != x:                       # NaN
        return "  nan"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    if abs(x) >= 1:
        return f"{x:.3f}"
    return f"{x:.4f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Report the NSGA-II generation sweep.")
    ap.add_argument("--instance", default=DEFAULT_INSTANCE,
                    help=f"instance name (default: {DEFAULT_INSTANCE})")
    args = ap.parse_args()
    instance = args.instance

    found = _discover(instance)
    if not found:
        raise SystemExit(
            f"no {PREFIX}{instance}_* files in {os.path.relpath(TMP, ROOT)}/ — "
            f"run: python scripts/eval_gens_sweep.py --instance {instance}")

    # Ordered arms: gens ascending. Label is the aggregate key.
    arms: list[tuple[int, str, list[int]]] = [
        (gens, f"gens={gens}", found[gens]) for gens in sorted(found)
    ]

    raws = {label: [_load_raw(instance, gens, s) for s in seeds]
            for gens, label, seeds in arms}
    runs_by_arm = {label: [_run_from_raw(raws[label][i], s) for i, s in enumerate(seeds)]
                   for gens, label, seeds in arms}

    # One shared reference front over ALL gens levels -> comparable hv/igd/gd/spacing.
    agg = aggregate(runs_by_arm)

    # Per-arm metric arrays over seeds: best_* off the raw fronts, geometric from agg.
    per_arm: dict[str, dict[str, np.ndarray]] = {}
    for gens, label, seeds in arms:
        rows = agg["per_run"][label]                 # aligned to this arm's seed order
        fronts = [np.asarray(r["front"], dtype=float) for r in raws[label]]
        per_arm[label] = {
            "nps": np.array([r["nps"] for r in rows]),
            "best_mk": np.array([f[:, 0].min() for f in fronts]),
            "best_en": np.array([f[:, 1].min() for f in fronts]),
            "hv": np.array([r["hv"] for r in rows]),
            "igd": np.array([r["igd"] for r in rows]),
            "gd": np.array([r["gd"] for r in rows]),
            "spacing": np.array([r["spacing"] for r in rows]),
        }

    # --- table ------------------------------------------------------------------
    n_seeds = max(len(s) for *_, s in arms)
    print("\n" + "=" * 100)
    print(f"{instance}: NSGA-II GENERATION SWEEP  (pop fixed; median over seeds, n<={n_seeds})")
    print("NOTE: off-parity, small-n -> read TRENDS not significance; NOT for the paper table.")
    print("n_evals is MEASURED (NSGA-II under-evaluates), shown as the seed median.")
    print(f"reference front: {agg['reference_size']} pts (union of all gens levels, non-dominated)")
    print("=" * 100)
    hdr = f"{'arm':<12}{'n_evals':>10}" + "".join(f"{lab:>10}" for _, lab, _ in COLUMNS)
    print(hdr)
    print("-" * len(hdr))
    for gens, label, seeds in arms:
        n_evals = int(median_iqr(np.array([d["n_evals"] for d in raws[label]], float))[0])
        cells = "".join(f"{_fmt(median_iqr(per_arm[label][key])[0]):>10}"
                        for key, _, _ in COLUMNS)
        print(f"{label:<12}{n_evals:>10}{cells}")
    print("-" * len(hdr))
    dirs = "  ".join(f"{lab}{'+' if hb else '-'}" for _, lab, hb in COLUMNS)
    print(f"dir (+ higher better / - lower better): {dirs}")

    # --- persist metrics JSON ---------------------------------------------------
    os.makedirs(TMP, exist_ok=True)
    table = {label: {key: dict(zip(("median", "iqr"), median_iqr(per_arm[label][key])))
                     for key, _, _ in COLUMNS}
             for _, label, _ in arms}
    out = {"instance": instance, "algorithm": "nsga2", "pop_fixed": True,
           "arms": {label: {"gens": gens, "seeds": seeds,
                            "n_evals_measured": [int(d["n_evals"]) for d in raws[label]]}
                    for gens, label, seeds in arms},
           "reference_size": agg["reference_size"], "table": table}
    metrics_path = os.path.join(TMP, f"{PREFIX}{instance}_metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nsaved: {os.path.relpath(metrics_path, ROOT)}")

    # --- Pareto overlay (gens levels, fleet-colored) ----------------------------
    recs = {label: [pt for d in raws[label] for pt in _records(d)]
            for _, label, _ in arms}
    ax = plot_pareto(recs, title=f"NSGA-II generation scaling — {instance} "
                                 f"(pop fixed, pooled seeds)")
    pareto_path = os.path.join(TMP, f"{PREFIX}{instance}_pareto.pdf")
    ax.figure.savefig(pareto_path)
    plt.close(ax.figure)
    print(f"saved: {os.path.relpath(pareto_path, ROOT)}")

    # --- convergence overlay (each gens level to its own horizon) ---------------
    hists = {label: [d["history"] for d in raws[label]] for _, label, _ in arms}
    axes = plot_convergence(hists)
    axes[0].set_title(f"NSGA-II convergence — {instance} (seed-median, pop fixed)")
    conv_path = os.path.join(TMP, f"{PREFIX}{instance}_convergence.pdf")
    axes[0].figure.savefig(conv_path)
    plt.close(axes[0].figure)
    print(f"saved: {os.path.relpath(conv_path, ROOT)}")


if __name__ == "__main__":
    main()
