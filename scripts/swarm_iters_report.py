"""Report the MOPSO/DMOPSO iteration-scaling sweep: table + per-method figures.

READS ``tmp_results/`` ONLY — re-runs no optimizer. Auto-discovers every
``swarmiters_<instance>_<algo>_i<iters>_<seed>.json`` written by
``scripts/eval_swarm_iters.py`` (so running a subset just reports that subset),
builds ONE shared union reference front over all arms (via ``evaluation.aggregate``)
so HV/IGD/GD/spacing are comparable, and emits:

  * a printed metric table (rows = algo x iters; headline best-makespan/best-energy
    columns) + ``tmp_results/swarmiters_<instance>_metrics.json``;
  * PER METHOD, an iteration overlay (keeps each figure within the 4-color/4-marker
    palettes and answers the scaling question directly):
      tmp_results/swarmiters_<instance>_<algo>_pareto.pdf
      tmp_results/swarmiters_<instance>_<algo>_convergence.pdf

The headline for an iteration study is best_en (does more budget lower energy?) and
whether the front keeps spreading. These numbers are off-parity and EXPLORATORY —
not for the paper's co-equal table. With only 2 seeds the median is barely smoothed;
read trends, not significance.

Usage:  python scripts/swarm_iters_report.py
        python scripts/swarm_iters_report.py --instance eil51-k3
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
PREFIX = "swarmiters_"
DEFAULT_INSTANCE = "eil51-k3"
ALGO_ORDER = ("mopso", "dmopso")                  # row/figure order
DISPLAY = {"mopso": "MOPSO", "dmopso": "DMOPSO"}

# Reported columns: (key, label, higher_is_better). best_* off the raw front; the
# rest from the shared-reference aggregate (mirrors compare_double_iters).
COLUMNS = (
    ("nps", "NPS", True),
    ("best_mk", "best_mk", False),
    ("best_en", "best_en", False),
    ("hv", "HV", True),
    ("igd", "IGD", False),
    ("gd", "GD", False),
    ("spacing", "spacing", False),
)


def _discover(instance: str) -> dict[tuple[str, int], list[int]]:
    """Map (algo, iters) -> sorted seed list present on disk for this instance."""
    pat = re.compile(rf"{re.escape(PREFIX)}{re.escape(instance)}_(mopso|dmopso)_i(\d+)_(\d+)\.json$")
    found: dict[tuple[str, int], list[int]] = {}
    for p in glob.glob(os.path.join(TMP, f"{PREFIX}{instance}_*_i*_*.json")):
        m = pat.search(os.path.basename(p))
        if m:
            found.setdefault((m.group(1), int(m.group(2))), []).append(int(m.group(3)))
    return {k: sorted(v) for k, v in found.items()}


def _raw_path(instance: str, algo: str, iters: int, seed: int) -> str:
    return os.path.join(TMP, f"{PREFIX}{instance}_{algo}_i{iters}_{seed}.json")


def _load_raw(instance: str, algo: str, iters: int, seed: int) -> dict:
    with open(_raw_path(instance, algo, iters, seed)) as fh:
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
    ap = argparse.ArgumentParser(description="Report the MOPSO/DMOPSO iteration sweep.")
    ap.add_argument("--instance", default=DEFAULT_INSTANCE,
                    help=f"instance name (default: {DEFAULT_INSTANCE})")
    args = ap.parse_args()
    instance = args.instance

    found = _discover(instance)
    if not found:
        raise SystemExit(
            f"no {PREFIX}{instance}_* files in {os.path.relpath(TMP, ROOT)}/ — "
            f"run: python scripts/eval_swarm_iters.py --instance {instance}")

    # Ordered arms: mopso then dmopso, iters ascending. Label is the aggregate key.
    arms: list[tuple[str, int, str, list[int]]] = []   # (algo, iters, label, seeds)
    for algo in ALGO_ORDER:
        for iters in sorted(it for (a, it) in found if a == algo):
            seeds = found[(algo, iters)]
            arms.append((algo, iters, f"{DISPLAY[algo]} i{iters}", seeds))

    raws = {label: [_load_raw(instance, algo, iters, s) for s in seeds]
            for algo, iters, label, seeds in arms}
    runs_by_arm = {label: [_run_from_raw(raws[label][i], s) for i, s in enumerate(seeds)]
                   for algo, iters, label, seeds in arms}

    # One shared reference front over ALL arms -> comparable hv/igd/gd/spacing.
    agg = aggregate(runs_by_arm)

    # Per-arm metric arrays over seeds: best_* off the raw fronts, geometric from agg.
    per_arm: dict[str, dict[str, np.ndarray]] = {}
    for algo, iters, label, seeds in arms:
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
    print(f"{instance}: MOPSO/DMOPSO ITERATION SWEEP  (swarm fixed; median over seeds, "
          f"n<={n_seeds})")
    print("NOTE: off-parity, small-n -> read TRENDS not significance; NOT for the paper table.")
    print(f"reference front: {agg['reference_size']} pts (union of all arms, non-dominated)")
    print("=" * 100)
    hdr = f"{'arm':<16}{'n_evals':>10}" + "".join(f"{lab:>10}" for _, lab, _ in COLUMNS)
    print(hdr)
    print("-" * len(hdr))
    prev_algo = None
    for algo, iters, label, seeds in arms:
        if prev_algo is not None and algo != prev_algo:
            print("-" * len(hdr))                    # group separator between methods
        prev_algo = algo
        n_evals = int(raws[label][0]["n_evals"])
        cells = "".join(f"{_fmt(median_iqr(per_arm[label][key])[0]):>10}"
                        for key, _, _ in COLUMNS)
        print(f"{label:<16}{n_evals:>10}{cells}")
    print("-" * len(hdr))
    dirs = "  ".join(f"{lab}{'+' if hb else '-'}" for _, lab, hb in COLUMNS)
    print(f"dir (+ higher better / - lower better): {dirs}")

    # --- persist metrics JSON ---------------------------------------------------
    os.makedirs(TMP, exist_ok=True)
    table = {label: {key: dict(zip(("median", "iqr"), median_iqr(per_arm[label][key])))
                     for key, _, _ in COLUMNS}
             for *_ , label, _ in arms}
    out = {"instance": instance, "swarm_fixed": True,
           "arms": {label: {"algo": algo, "iters": iters, "seeds": seeds,
                            "n_evals": int(raws[label][0]["n_evals"])}
                    for algo, iters, label, seeds in arms},
           "reference_size": agg["reference_size"], "table": table}
    metrics_path = os.path.join(TMP, f"{PREFIX}{instance}_metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nsaved: {os.path.relpath(metrics_path, ROOT)}")

    # --- per-method figures: overlay the iteration budgets ----------------------
    # One Pareto + one convergence per method, overlaying its iters levels. Kept per
    # method so each figure stays within the 4-color (convergence) / 4-marker
    # (pareto) palettes, and so the iteration progression is the only varying axis.
    for algo in ALGO_ORDER:
        algo_arms = [(iters, label, seeds) for a, iters, label, seeds in arms if a == algo]
        if not algo_arms:
            continue

        # Pareto: pool each iters level across its seeds; label by iters.
        recs = {f"iters={iters}": [pt for d in raws[label] for pt in _records(d)]
                for iters, label, _ in algo_arms}
        ax = plot_pareto(recs, title=f"{DISPLAY[algo]} iteration scaling — {instance} "
                                     f"(swarm fixed, pooled seeds)")
        pareto_path = os.path.join(TMP, f"{PREFIX}{instance}_{algo}_pareto.pdf")
        ax.figure.savefig(pareto_path)
        plt.close(ax.figure)
        print(f"saved: {os.path.relpath(pareto_path, ROOT)}")

        # Convergence: seed-median per iters level, each drawn to its own horizon.
        hists = {f"iters={iters}": [d["history"] for d in raws[label]]
                 for iters, label, _ in algo_arms}
        axes = plot_convergence(hists)
        axes[0].set_title(f"{DISPLAY[algo]} convergence — {instance} (seed-median, swarm fixed)")
        conv_path = os.path.join(TMP, f"{PREFIX}{instance}_{algo}_convergence.pdf")
        axes[0].figure.savefig(conv_path)
        plt.close(axes[0].figure)
        print(f"saved: {os.path.relpath(conv_path, ROOT)}")


if __name__ == "__main__":
    main()
