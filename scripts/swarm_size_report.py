"""Report the swarm-size probe: decision table + per-method figures.

READS ``tmp_results/`` ONLY. Auto-discovers every
``swarmsize_<instance>_<algo>_s<swarm>_<seed>.json`` written by
``scripts/eval_swarm_size.py``, builds ONE shared union reference front over the swarm
arms (via ``evaluation.aggregate``), and prints a DECISION table to pick the swarm size
for the Phase-2 overnight run, plus a Pareto + convergence overlay per method.

The reference front is over the SWARM arms ONLY (NSGA-II deliberately excluded): the
goal is to discriminate *among swarm choices*, and folding in the dominating NSGA-II
front would flatten every swarm arm to HV~0 and tell us nothing. For absolute context,
NSGA-II's pop=100/gens=3000 best_en is printed as a reference line (read from the
existing genssweep g3000 JSONs when present).

The pick = the swarm minimizing best_en / best_mk per method (ideally the same swarm
serves both; if they disagree, the compromise favours MOPSO, which is the iters-
sensitive one). Off-parity/exploratory — selects a parameter, not a paper result.

Usage:  python scripts/swarm_size_report.py
        python scripts/swarm_size_report.py --instance eil51-k3
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
PREFIX = "swarmsize_"
DEFAULT_INSTANCE = "eil51-k3"
ALGO_ORDER = ("mopso", "dmopso")
DISPLAY = {"mopso": "MOPSO", "dmopso": "DMOPSO"}

# Reported columns: (key, label, higher_is_better). best_* off the raw front; the rest
# from the shared-reference aggregate.
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
    """Map (algo, swarm) -> sorted seed list present on disk for this instance."""
    pat = re.compile(rf"{re.escape(PREFIX)}{re.escape(instance)}_(mopso|dmopso)_s(\d+)_(\d+)\.json$")
    found: dict[tuple[str, int], list[int]] = {}
    for p in glob.glob(os.path.join(TMP, f"{PREFIX}{instance}_*_s*_*.json")):
        m = pat.search(os.path.basename(p))
        if m:
            found.setdefault((m.group(1), int(m.group(2))), []).append(int(m.group(3)))
    return {k: sorted(v) for k, v in found.items()}


def _load_raw(instance: str, algo: str, swarm: int, seed: int) -> dict:
    with open(os.path.join(TMP, f"{PREFIX}{instance}_{algo}_s{swarm}_{seed}.json")) as fh:
        return json.load(fh)


def _run_from_raw(data: dict, seed: int) -> Run:
    front = [
        Solution(routes=tuple(tuple(int(p) for p in r) for r in routes_i),
                 makespan=float(mk), energy=float(en))
        for (mk, en), routes_i in zip(data["front"], data["routes"])
    ]
    return Run(seed=seed, front=front,
               wall_clock_s=float(data["wall_clock_s"]), n_evals=int(data["n_evals"]))


def _records(data: dict) -> list[tuple]:
    return [(obj[0], obj[1], sum(1 for r in routes if len(r) > 2))
            for obj, routes in zip(data["front"], data["routes"])]


def _nsga_reference_best_en(instance: str) -> float | None:
    """NSGA-II pop=100/gens=3000 best_en (min over seeds), from the genssweep JSONs,
    for absolute context. Returns None if those runs are not on disk."""
    paths = glob.glob(os.path.join(TMP, f"genssweep_{instance}_nsga2_g3000_*.json"))
    if not paths:
        return None
    best = min(min(float(en) for _, en in json.load(open(p))["front"]) for p in paths)
    return best


def _fmt(x: float) -> str:
    if x != x:                       # NaN
        return "  nan"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    if abs(x) >= 1:
        return f"{x:.3f}"
    return f"{x:.4f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Report the swarm-size probe.")
    ap.add_argument("--instance", default=DEFAULT_INSTANCE,
                    help=f"instance name (default: {DEFAULT_INSTANCE})")
    args = ap.parse_args()
    instance = args.instance

    found = _discover(instance)
    if not found:
        raise SystemExit(
            f"no {PREFIX}{instance}_* files in {os.path.relpath(TMP, ROOT)}/ — "
            f"run: python scripts/eval_swarm_size.py --instance {instance}")

    # Ordered arms: mopso then dmopso, swarm ascending. Label is the aggregate key.
    arms: list[tuple[str, int, str, list[int]]] = []
    for algo in ALGO_ORDER:
        for swarm in sorted(s for (a, s) in found if a == algo):
            arms.append((algo, swarm, f"{DISPLAY[algo]} s{swarm}", found[(algo, swarm)]))

    raws = {label: [_load_raw(instance, algo, swarm, s) for s in seeds]
            for algo, swarm, label, seeds in arms}
    runs_by_arm = {label: [_run_from_raw(raws[label][i], s) for i, s in enumerate(seeds)]
                   for algo, swarm, label, seeds in arms}

    # Shared reference over the SWARM arms only -> hv/igd/gd discriminate among swarms.
    agg = aggregate(runs_by_arm)

    per_arm: dict[str, dict[str, np.ndarray]] = {}
    for algo, swarm, label, seeds in arms:
        rows = agg["per_run"][label]
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
    print("\n" + "=" * 104)
    print(f"{instance}: SWARM-SIZE PROBE at fixed budget  (median over seeds, n<={n_seeds})")
    print("Budget held ~constant across swarms (iters derived per swarm); reference front over "
          "swarm arms ONLY.")
    nsga_en = _nsga_reference_best_en(instance)
    if nsga_en is not None:
        print(f"context: NSGA-II pop=100/gens=3000 best_en = {nsga_en:,.0f} "
              f"(the co-equal target the swarm methods are measured against)")
    print(f"reference front: {agg['reference_size']} pts (union of swarm arms, non-dominated)")
    print("=" * 104)
    hdr = f"{'arm':<14}{'iters':>7}{'n_evals':>10}" + "".join(f"{lab:>10}" for _, lab, _ in COLUMNS)
    print(hdr)
    print("-" * len(hdr))
    prev_algo = None
    best_by_algo: dict[str, tuple[int, float]] = {}   # algo -> (swarm, best_en) winner
    for algo, swarm, label, seeds in arms:
        if prev_algo is not None and algo != prev_algo:
            print("-" * len(hdr))
        prev_algo = algo
        iters = int(raws[label][0]["hyperparams"]["iters"])
        n_evals = int(raws[label][0]["n_evals"])
        med_en = median_iqr(per_arm[label]["best_en"])[0]
        if algo not in best_by_algo or med_en < best_by_algo[algo][1]:
            best_by_algo[algo] = (swarm, med_en)
        cells = "".join(f"{_fmt(median_iqr(per_arm[label][key])[0]):>10}"
                        for key, _, _ in COLUMNS)
        print(f"{label:<14}{iters:>7}{n_evals:>10}{cells}")
    print("-" * len(hdr))
    dirs = "  ".join(f"{lab}{'+' if hb else '-'}" for _, lab, hb in COLUMNS)
    print(f"dir (+ higher better / - lower better): {dirs}")
    print("\nbest_en winner per method (lower is better):")
    for algo in ALGO_ORDER:
        if algo in best_by_algo:
            sw, en = best_by_algo[algo]
            print(f"  {DISPLAY[algo]:<7}: swarm={sw}  (best_en median {en:,.0f})")

    # --- persist metrics JSON ---------------------------------------------------
    os.makedirs(TMP, exist_ok=True)
    table = {label: {key: dict(zip(("median", "iqr"), median_iqr(per_arm[label][key])))
                     for key, _, _ in COLUMNS}
             for *_, label, _ in arms}
    out = {"instance": instance, "budget_fixed": True,
           "nsga2_g3000_best_en": nsga_en,
           "arms": {label: {"algo": algo, "swarm": swarm, "seeds": seeds,
                            "iters": int(raws[label][0]["hyperparams"]["iters"]),
                            "n_evals": int(raws[label][0]["n_evals"])}
                    for algo, swarm, label, seeds in arms},
           "best_en_winner": {algo: best_by_algo[algo][0] for algo in best_by_algo},
           "reference_size": agg["reference_size"], "table": table}
    metrics_path = os.path.join(TMP, f"{PREFIX}{instance}_metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nsaved: {os.path.relpath(metrics_path, ROOT)}")

    # --- per-method figures: overlay the swarm sizes ----------------------------
    for algo in ALGO_ORDER:
        algo_arms = [(swarm, label, seeds) for a, swarm, label, seeds in arms if a == algo]
        if not algo_arms:
            continue

        recs = {f"swarm={swarm}": [pt for d in raws[label] for pt in _records(d)]
                for swarm, label, _ in algo_arms}
        ax = plot_pareto(recs, title=f"{DISPLAY[algo]} swarm-size @ fixed budget — {instance} "
                                     f"(pooled seeds)")
        pareto_path = os.path.join(TMP, f"{PREFIX}{instance}_{algo}_pareto.pdf")
        ax.figure.savefig(pareto_path)
        plt.close(ax.figure)
        print(f"saved: {os.path.relpath(pareto_path, ROOT)}")

        hists = {f"swarm={swarm}": [d["history"] for d in raws[label]]
                 for swarm, label, _ in algo_arms}
        axes = plot_convergence(hists)
        axes[0].set_title(f"{DISPLAY[algo]} convergence (swarm-size @ fixed budget) — {instance}")
        conv_path = os.path.join(TMP, f"{PREFIX}{instance}_{algo}_convergence.pdf")
        axes[0].figure.savefig(conv_path)
        plt.close(axes[0].figure)
        print(f"saved: {os.path.relpath(conv_path, ROOT)}")


if __name__ == "__main__":
    main()
