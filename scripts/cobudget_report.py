"""Report the co-equal overnight run: per-instance table + Pareto + convergence.

READS ``results_cobudget/`` ONLY (re-runs nothing). For each instance with runs on
disk it builds ONE shared union reference front over NSGA-II + MOPSO + DMOPSO (via
``evaluation.aggregate``), prints the co-equal metric table (median over seeds, with
the NSGA-II-vs-MOPSO and NSGA-II-vs-DMOPSO Mann-Whitney p per metric), and writes a
Pareto + convergence figure.

This is the paper-style comparison: the shared reference includes NSGA-II, so the swarm
methods' HV reads honestly (≈0 when dominated). DMOPSO is the opt-in diagnostic variant —
shown here because the user compares all three; it can be dropped from the final paper
figure. Raw Mann-Whitney p is reported; the paper's Holm family correction
(``scripts/holm_correction.py``) is applied separately, downstream.

Usage:
    python scripts/cobudget_report.py                               # all instances found
    python scripts/cobudget_report.py --instances eil51-k3
    python scripts/cobudget_report.py --results-dir results_cobudget_smoke
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")  # headless: must precede pyplot
import matplotlib.pyplot as plt
import numpy as np

from uav.evaluation.aggregate import Run, aggregate
from uav.evaluation.stats import mann_whitney, median_iqr
from uav.solution import Solution
from uav.viz.convergence import plot_convergence
from uav.viz.pareto import plot_pareto

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DEFAULT_RESULTS = os.path.join(ROOT, "results_cobudget")
INSTANCES = ("eil51-k3", "berlin52-k3", "eil76-k3", "rat99-k3")
ARMS = (("nsga2", "NSGA-II"), ("mopso", "MOPSO"), ("dmopso", "DMOPSO"))

# (key, label, higher_is_better). best_* off the raw fronts; the rest from aggregate.
COLUMNS = (
    ("best_mk", "best_mk", False),
    ("best_en", "best_en", False),
    ("nps", "NPS", True),
    ("hv", "HV", True),
    ("igd", "IGD", False),
    ("gd", "GD", False),
    ("spacing", "spacing", False),
    ("ct", "CT(s)", False),
    ("n_evals", "n_evals", None),
)


def _runs_raw(results_dir: str, instance: str, algo: str) -> list[tuple[int, dict]]:
    """(seed, data) for every persisted run of (instance, algo), sorted by seed."""
    out = []
    for p in glob.glob(os.path.join(results_dir, f"{instance}_{algo}_*.json")):
        base = os.path.basename(p)
        seed_tag = base[len(f"{instance}_{algo}_"):-len(".json")]
        if seed_tag.isdigit():
            out.append((int(seed_tag), json.load(open(p))))
    return [t for t in sorted(out, key=lambda x: x[0])]


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


def _fmt(x: float) -> str:
    if x != x:                       # NaN
        return "  nan"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    if abs(x) >= 1:
        return f"{x:.3f}"
    return f"{x:.4f}"


def _report_instance(results_dir: str, instance: str) -> bool:
    raw = {algo: _runs_raw(results_dir, instance, algo) for algo, _ in ARMS}
    present = [(algo, label) for algo, label in ARMS if raw[algo]]
    if len(present) < 2:
        print(f"== {instance}: only {[l for _, l in present]} on disk — need >=2 arms; skipping ==\n")
        return False

    runs_by_algo = {label: [_run_from_raw(d, s) for s, d in raw[algo]] for algo, label in present}
    agg = aggregate(runs_by_algo)

    per_arm: dict[str, dict[str, np.ndarray]] = {}
    for algo, label in present:
        rows = agg["per_run"][label]
        fronts = [np.asarray(d["front"], dtype=float) for _, d in raw[algo]]
        per_arm[label] = {
            "best_mk": np.array([f[:, 0].min() for f in fronts]),
            "best_en": np.array([f[:, 1].min() for f in fronts]),
            "nps": np.array([r["nps"] for r in rows]),
            "hv": np.array([r["hv"] for r in rows]),
            "igd": np.array([r["igd"] for r in rows]),
            "gd": np.array([r["gd"] for r in rows]),
            "spacing": np.array([r["spacing"] for r in rows]),
            "ct": np.array([r["ct"] for r in rows]),
            "n_evals": np.array([r["n_evals"] for r in rows]),
        }

    n_seeds = max(len(v) for v in raw.values() if v)
    print("=" * 104)
    print(f"{instance}: CO-EQUAL TABLE  (median over seeds, n={n_seeds})")
    print(f"reference front: {agg['reference_size']} pts (union of {', '.join(l for _, l in present)})")
    print("=" * 104)
    hdr = f"{'arm':<10}" + "".join(f"{lab:>11}" for _, lab, _ in COLUMNS)
    print(hdr)
    print("-" * len(hdr))
    for algo, label in present:
        cells = "".join(f"{_fmt(median_iqr(per_arm[label][key])[0]):>11}" for key, _, _ in COLUMNS)
        print(f"{label:<10}{cells}")
    print("-" * len(hdr))
    dirs = "  ".join(f"{lab}{'+' if hb else ('-' if hb is False else ' ')}" for _, lab, hb in COLUMNS)
    print(f"dir (+ higher better / - lower better): {dirs}")

    # Mann-Whitney p vs NSGA-II per metric (the paper's headline pairs). Raw p — the
    # Holm family correction is applied downstream (scripts/holm_correction.py).
    labels = {algo: label for algo, label in present}
    pvalues: dict[str, dict[str, float]] = {}
    if "nsga2" in labels:
        for other in ("mopso", "dmopso"):
            if other in labels:
                name = f"NSGA-II vs {labels[other]}"
                pvalues[name] = {}
                cells = []
                for key, lab, _ in COLUMNS:
                    if key == "n_evals":
                        continue
                    _, p = mann_whitney(per_arm[labels["nsga2"]][key], per_arm[labels[other]][key])
                    pvalues[name][key] = p
                    cells.append(f"{lab}={_fmt(p)}")
                print(f"\nMann-Whitney p, {name} (raw): " + "  ".join(cells))

    # Persist metrics JSON (full median+IQR).
    table = {label: {key: dict(zip(("median", "iqr"), median_iqr(per_arm[label][key])))
                     for key, _, _ in COLUMNS}
             for _, label in present}
    out = {"instance": instance, "n_seeds": n_seeds, "co_equal": True,
           "reference_size": agg["reference_size"], "table": table, "pvalues_raw": pvalues}
    metrics_path = os.path.join(results_dir, f"{instance}_cobudget_metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nsaved: {os.path.relpath(metrics_path, ROOT)}")

    # Figures (3 arms <= palette sizes).
    recs = {label: [pt for _, d in raw[algo] for pt in _records(d)] for algo, label in present}
    ax = plot_pareto(recs, title=f"Co-equal Pareto — {instance} (n={n_seeds})")
    pareto_path = os.path.join(results_dir, f"{instance}_cobudget_pareto.pdf")
    ax.figure.savefig(pareto_path)
    plt.close(ax.figure)

    hists = {label: [d["history"] for _, d in raw[algo]] for algo, label in present}
    axes = plot_convergence(hists)
    axes[0].set_title(f"Co-equal convergence — {instance} (seed-median, n={n_seeds})")
    conv_path = os.path.join(results_dir, f"{instance}_cobudget_convergence.pdf")
    axes[0].figure.savefig(conv_path)
    plt.close(axes[0].figure)
    print(f"saved: {os.path.relpath(pareto_path, ROOT)} , {os.path.relpath(conv_path, ROOT)}\n")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Report the co-equal overnight run.")
    ap.add_argument("--instances", type=lambda s: [x for x in s.split(",") if x],
                    default=list(INSTANCES), help="comma-separated instance names (default: all 4)")
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS,
                    help="results dir to read (default: results_cobudget)")
    args = ap.parse_args()
    results_dir = os.path.abspath(args.results_dir)
    if not os.path.isdir(results_dir):
        raise SystemExit(f"no such dir: {results_dir} — run scripts/eval_cobudget.py first")

    any_done = False
    for instance in args.instances:
        any_done |= _report_instance(results_dir, instance)
    if not any_done:
        raise SystemExit(f"no complete instances found in {os.path.relpath(results_dir, ROOT)}/")


if __name__ == "__main__":
    main()
