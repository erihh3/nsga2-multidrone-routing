"""Report the single-seed budget probe: comparison table + Pareto + convergence.

READS ``tmp_results/`` ONLY — re-runs no optimizer. Loads the four probe arms
written by ``scripts/eval_budget_probe.py`` (NSGA-II p100g1200, NSGA-II p150g1200,
MOPSO s150i1200, DMOPSO s150i1200 on eil51-k3), builds ONE shared union reference
front over all four (via ``evaluation.aggregate``) so HV/IGD/GD/spacing share a
common reference, and emits:

  * a printed metric table (+ headline best-makespan / best-energy columns) and
    ``tmp_results/<instance>_budget_probe_metrics.json``;
  * ``tmp_results/<instance>_budget_probe_pareto.pdf``      (union fronts, fleet-colored);
  * ``tmp_results/<instance>_budget_probe_convergence.pdf`` (best/mean per objective).

SINGLE-SEED CAVEAT: this is n=1 exploratory probe data. With one seed the "median"
is just the raw value, the IQR is 0, and there is no Mann-Whitney significance — the
project's "never quote single-seed" rule is honoured by LABELLING this, not by
hiding it. These numbers are off-parity and must not enter the paper's co-equal table.

Usage:  python scripts/budget_probe_report.py
        python scripts/budget_probe_report.py --instance eil51-k3 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")  # headless: must precede pyplot
import matplotlib.pyplot as plt
import numpy as np

from uav.evaluation.aggregate import Run, aggregate
from uav.solution import Solution
from uav.viz.convergence import plot_convergence
from uav.viz.pareto import plot_pareto

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
TMP = os.path.join(ROOT, "tmp_results")

DEFAULT_INSTANCE = "eil51-k3"
DEFAULT_SEED = 0

# (label, filename tag) — must match scripts/eval_budget_probe.py. Order fixes the
# table rows, the Pareto marker shapes, and the convergence colors.
ARMS = (
    ("NSGA-II p100g1200", "nsga2-p100g1200"),
    ("NSGA-II p150g1200", "nsga2-p150g1200"),
    ("MOPSO s150i1200",   "mopso-s150i1200"),
    ("DMOPSO s150i1200",  "dmopso-s150i1200"),
)

# Reported columns: (key, label, higher_is_better). best_* come off the raw front;
# the rest come from the shared-reference aggregate (mirrors compare_double_iters).
COLUMNS = (
    ("nps", "NPS", True),
    ("best_mk", "best_mk", False),
    ("best_en", "best_en", False),
    ("hv", "HV", True),
    ("igd", "IGD", False),
    ("gd", "GD", False),
    ("spacing", "spacing", False),
)


def _raw_path(instance: str, tag: str, seed: int) -> str:
    return os.path.join(TMP, f"{instance}_{tag}_{seed}.json")


def _load_raw(instance: str, tag: str, seed: int) -> dict:
    path = _raw_path(instance, tag, seed)
    if not os.path.exists(path):
        raise SystemExit(
            f"missing probe file: {os.path.relpath(path, ROOT)}\n"
            f"run the probe first: python scripts/eval_budget_probe.py "
            f"--instance {instance} --seed {seed}")
    with open(path) as fh:
        return json.load(fh)


def _run_from_raw(data: dict, seed: int) -> Run:
    """Rebuild a Run (Solution fronts) from a persisted probe JSON.

    Logic identical to compare_double_iters._run_from_raw: n_active is recomputed
    from routes inside Solution, never read off the JSON key, so it is consistent.
    """
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


def _records(data: dict) -> list[tuple]:
    """(makespan, energy, n_active) points for the Pareto plot. Fleet size is the
    count of tours longer than [depot, depot] (reused from make_figures._records)."""
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
    ap = argparse.ArgumentParser(description="Report the single-seed budget probe.")
    ap.add_argument("--instance", default=DEFAULT_INSTANCE,
                    help=f"instance name (default: {DEFAULT_INSTANCE})")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help=f"the probe seed to report (default: {DEFAULT_SEED})")
    args = ap.parse_args()
    instance, seed = args.instance, args.seed

    raws = {label: _load_raw(instance, tag, seed) for label, tag in ARMS}
    runs_by_arm = {label: [_run_from_raw(raws[label], seed)] for label, _ in ARMS}

    # One shared reference front over ALL four arms -> comparable hv/igd/gd/spacing.
    agg = aggregate(runs_by_arm)

    # Per-arm metrics: best_* off the raw front, geometric from the aggregate. With
    # n=1 each per_run list holds a single row, so we take [0] directly.
    per_arm: dict[str, dict[str, float]] = {}
    for label, _ in ARMS:
        row = agg["per_run"][label][0]
        front = np.asarray(raws[label]["front"], dtype=float)
        per_arm[label] = {
            "nps": row["nps"],
            "best_mk": float(front[:, 0].min()),
            "best_en": float(front[:, 1].min()),
            "hv": row["hv"],
            "igd": row["igd"],
            "gd": row["gd"],
            "spacing": row["spacing"],
        }

    # --- table ------------------------------------------------------------------
    print("\n" + "=" * 96)
    print(f"{instance}: BUDGET PROBE  (single seed {seed} — EXPLORATORY, n=1)")
    print("NOTE: off-parity, n=1 -> values are raw (no IQR, no significance); NOT for the paper table.")
    print(f"reference front: {agg['reference_size']} pts (union of all 4 arms, non-dominated)")
    print("=" * 96)
    hdr = f"{'arm':<20}" + "".join(f"{lab:>11}" for _, lab, _ in COLUMNS)
    print(hdr)
    print("-" * len(hdr))
    for label, _ in ARMS:
        cells = "".join(f"{_fmt(per_arm[label][key]):>11}" for key, _, _ in COLUMNS)
        print(f"{label:<20}{cells}")
    print("-" * len(hdr))
    dirs = "  ".join(f"{lab}{'+' if hb else '-'}" for _, lab, hb in COLUMNS)
    print(f"dir (+ higher better / - lower better): {dirs}")

    # --- persist metrics JSON ---------------------------------------------------
    os.makedirs(TMP, exist_ok=True)
    out = {
        "instance": instance, "seed": seed, "n_seeds": 1, "exploratory": True,
        "reference_size": agg["reference_size"],
        "n_evals": {label: int(raws[label]["n_evals"]) for label, _ in ARMS},
        "table": {label: {key: per_arm[label][key] for key, _, _ in COLUMNS}
                  for label, _ in ARMS},
    }
    metrics_path = os.path.join(TMP, f"{instance}_budget_probe_metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nsaved: {os.path.relpath(metrics_path, ROOT)}")

    # --- Pareto overlay (fleet-colored, one marker shape per arm) ----------------
    recs = {label: _records(raws[label]) for label, _ in ARMS}
    ax = plot_pareto(recs, title=f"Pareto — {instance} budget probe (seed {seed}, n=1)")
    pareto_path = os.path.join(TMP, f"{instance}_budget_probe_pareto.pdf")
    ax.figure.savefig(pareto_path)
    plt.close(ax.figure)
    print(f"saved: {os.path.relpath(pareto_path, ROOT)}")

    # --- convergence overlay (each arm to its own horizon) -----------------------
    hists = {label: [raws[label]["history"]] for label, _ in ARMS}
    axes = plot_convergence(hists)
    conv_path = os.path.join(TMP, f"{instance}_budget_probe_convergence.pdf")
    axes[0].figure.savefig(conv_path)
    plt.close(axes[0].figure)
    print(f"saved: {os.path.relpath(conv_path, ROOT)}")


if __name__ == "__main__":
    main()
