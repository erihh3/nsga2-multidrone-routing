"""Compose the 2x-iteration final table from saved JSON only — no figures, no reruns.

A quick, read-only table builder for when the per-run results already exist on disk
(results/eil51-k3_*  1x baseline  +  results_2x/2x_eil51-k3_*  2x runs, seeds 0-4).
Same arms, same shared-reference aggregate, same columns as
``scripts/compare_double_iters.py`` (now including NPS), but it skips the matplotlib
figure step so it returns instantly.

Usage:  python scripts/table_double_iters.py
"""

from __future__ import annotations

import glob
import json
import os
import re

import numpy as np

from uav.evaluation.aggregate import Run, aggregate
from uav.evaluation.stats import mann_whitney, median_iqr
from uav.solution import Solution

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
RESULTS = os.path.join(ROOT, "results")
RESULTS_2X = os.path.join(ROOT, "results_2x")
INSTANCE = "eil51-k3"
SEEDS = tuple(range(5))

# (key, label, higher_is_better). NPS = distinct non-dominated objective points.
COLUMNS = (
    ("nps", "NPS", True),
    ("best_mk", "best_mk", False),
    ("best_en", "best_en", False),
    ("hv", "HV", True),
    ("igd", "IGD", False),
    ("gd", "GD", False),
    ("spacing", "spacing", False),
)


def _load_raw(algo: str, seed: int, mult: int) -> dict:
    name = f"{INSTANCE}_{algo}_{seed}.json" if mult == 1 else f"{mult}x_{INSTANCE}_{algo}_{seed}.json"
    with open(os.path.join(RESULTS if mult == 1 else RESULTS_2X, name)) as fh:
        return json.load(fh)


def _run_from_raw(data: dict, seed: int) -> Run:
    front = [
        Solution(routes=tuple(tuple(int(p) for p in r) for r in routes_i),
                 makespan=float(mk), energy=float(en))
        for (mk, en), routes_i in zip(data["front"], data["routes"])
    ]
    return Run(seed=seed, front=front,
               wall_clock_s=float(data["wall_clock_s"]), n_evals=int(data["n_evals"]))


def _parity_iters() -> int:
    with open(os.path.join(RESULTS, f"{INSTANCE}_parity.json")) as fh:
        return int(json.load(fh)["iters"])


def _dmopso_mults() -> tuple[int, ...]:
    """DMOPSO iteration multipliers available on disk, sorted: 1x baseline plus every
    ``Nx`` with a full seed set in results_2x/. Lets the table pick up new doublings
    (2x, 4x, 8x, 16x, ...) with no code edits — just run the optimizer at that Nx."""
    mults = {1}
    for path in glob.glob(os.path.join(RESULTS_2X, f"*x_{INSTANCE}_dmopso_0.json")):
        m = re.fullmatch(rf"(\d+)x_{re.escape(INSTANCE)}_dmopso_0\.json", os.path.basename(path))
        if m and all(os.path.exists(
                os.path.join(RESULTS_2X, f"{m.group(1)}x_{INSTANCE}_dmopso_{s}.json"))
                for s in SEEDS):
            mults.add(int(m.group(1)))
    return tuple(sorted(mults))


def _fmt(x: float) -> str:
    if x != x:
        return "  nan"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    if abs(x) >= 1:
        return f"{x:.3f}"
    return f"{x:.4f}"


def main() -> None:
    parity = _parity_iters()
    dm = _dmopso_mults()                              # discovered dmopso scaling (1x, 2x, 4x, ...)
    arms = {
        "NSGA-II (ref)": ("nsga2", 1),
        f"MOPSO 1x({parity})": ("mopso", 1),
        f"MOPSO 2x({2 * parity})": ("mopso", 2),
    }
    for m in dm:
        arms[f"DMOPSO {m}x({m * parity})"] = ("dmopso", m)

    raws = {lab: [_load_raw(a, s, m) for s in SEEDS] for lab, (a, m) in arms.items()}
    runs_by_algo = {lab: [_run_from_raw(raws[lab][i], s) for i, s in enumerate(SEEDS)]
                    for lab in arms}
    agg = aggregate(runs_by_algo)                    # one shared reference front

    per_arm: dict[str, dict[str, np.ndarray]] = {}
    for lab in arms:
        rows = agg["per_run"][lab]
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

    scaling = ", ".join(f"{m}x={m * parity}" for m in dm if m > 1)
    print("\n" + "=" * 102)
    print(f"{INSTANCE}: iteration-scaling final table  (median +- IQR, n={len(SEEDS)} seeds 0-4)")
    print(f"parity iters={parity}  |  dmopso scaling: {scaling}  |  "
          f"reference front: {agg['reference_size']} pts (union of all arms)")
    print("NOTE: 2x/4x/... break budget parity with NSGA-II — exploratory, not for the paper table.")
    print("=" * 102)
    hdr = f"{'arm':<18}" + "".join(f"{lab:>12}" for _, lab, _ in COLUMNS)
    print(hdr)
    print("-" * len(hdr))
    for lab in arms:
        cells = "".join(f"{_fmt(median_iqr(per_arm[lab][k])[0]):>12}" for k, _, _ in COLUMNS)
        print(f"{lab:<18}{cells}")
    print("-" * len(hdr))
    print("dir: " + "  ".join(f"{lab}{'+' if hb else '-'}" for _, lab, hb in COLUMNS)
          + "   (+ higher better / - lower better; NPS = distinct front points)")

    # MOPSO 1x->2x, each consecutive dmopso doubling, and the overall 1x->max.
    comparisons = [("MOPSO  1x->2x", f"MOPSO 1x({parity})", f"MOPSO 2x({2 * parity})")]
    for prev, cur in zip(dm, dm[1:]):
        comparisons.append((f"DMOPSO {prev}x->{cur}x",
                            f"DMOPSO {prev}x({prev * parity})", f"DMOPSO {cur}x({cur * parity})"))
    if len(dm) > 2:
        comparisons.append((f"DMOPSO {dm[0]}x->{dm[-1]}x",
                            f"DMOPSO {dm[0]}x({dm[0] * parity})", f"DMOPSO {dm[-1]}x({dm[-1] * parity})"))
    print("\nMann-Whitney U p-values per metric (small p => significant change):")
    for name, la, lb in comparisons:
        cells = "  ".join(
            f"{lab}={_fmt(mann_whitney(per_arm[la][k], per_arm[lb][k])[1])}"
            for k, lab, _ in COLUMNS)
        print(f"  {name}: {cells}")


if __name__ == "__main__":
    main()
