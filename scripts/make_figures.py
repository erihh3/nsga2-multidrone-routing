"""Phase 6 driver: render every paper figure from the persisted sweep JSONs.

This is the I/O layer; the four pure viz functions (`uav.viz.*`) do the drawing.
It NEVER re-runs an optimizer — figures are reproducible from
`results/<instance>_<algo>_<seed>.json` alone (the reproducibility invariant).
Co-equality holds because every figure feeds both arms through the identical viz
code; an arm contributes only a label/color.

For each instance it writes (vector PDF):
  figures/<instance>_pareto.pdf       NSGA-II vs MOPSO union fronts, colored by fleet
  figures/<instance>_convergence.pdf  seed-median best/mean + IQR, two objectives
  figures/<instance>_routes.pdf       the chosen solution's K depot-rooted tours
and, for one instance, the time-synced drone animation (MP4, GIF fallback).

Presentation interface: `--routes-json PATH --index I` renders the route map AND
the animation for *any* persisted solution, so different outcomes can be shown in
a talk. Without it, the route/animation subject is the per-instance knee solution.

Usage:
    .venv/bin/python scripts/make_figures.py                      # all figures + eil51 animation
    .venv/bin/python scripts/make_figures.py --instances eil51-k3 # one instance
    .venv/bin/python scripts/make_figures.py --no-animation       # skip the (slow) animation
    .venv/bin/python scripts/make_figures.py --routes-json results/eil51-k3_nsga2_0.json --index 0
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

import matplotlib

matplotlib.use("Agg")  # headless: must precede pyplot
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, PillowWriter

from uav.problem.instance import load_instance
from uav.viz.animation import animate_routes
from uav.viz.convergence import plot_convergence
from uav.viz.pareto import plot_pareto
from uav.viz.routes import plot_routes

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
INSTANCES = ("eil51-k3", "berlin52-k3", "eil76-k3", "rat99-k3")
# Co-equal arms only; dmopso is the opt-in diagnostic and is never plotted.
ARMS = (("nsga2", "NSGA-II"), ("mopso", "MOPSO"))


# --- loading (all disk I/O lives here) -----------------------------------------

def _runs(instance: str, algo: str) -> list[dict]:
    pattern = os.path.join(RESULTS, f"{instance}_{algo}_*.json")
    return [json.load(open(f)) for f in sorted(glob.glob(pattern))]


def _records(runs: list[dict]) -> list[tuple]:
    """Pooled (makespan, energy, n_active) points. Fleet size is derived from
    routes (count tours longer than [depot, depot]) — never read off a JSON key,
    so it is consistent across arms and across the eil51 files that lack the key."""
    out = []
    for d in runs:
        for obj, routes in zip(d["front"], d["routes"]):
            out.append((obj[0], obj[1], sum(1 for r in routes if len(r) > 2)))
    return out


def _histories(runs: list[dict]) -> list[list[dict]]:
    return [d["history"] for d in runs]


def _coords(instance: str):
    stem = instance.split("-")[0]
    return load_instance(os.path.join(ROOT, "instances", f"{stem}.tsp"), k=3).coords


# --- solution selection (a driver concern, not pure viz) -----------------------

def _select(runs: list[dict], mode: str) -> tuple[list, list, int]:
    """Pick one (objectives, routes, n_active) from pooled fronts by ``mode``.

    knee:         min-max normalize both objectives over the pool, nearest origin.
    min-makespan: fastest mission. min-energy: most frugal. max-fleet: most drones.
    """
    objs, routes = [], []
    for d in runs:
        objs.extend(d["front"])
        routes.extend(d["routes"])
    objs = np.asarray(objs, dtype=float)
    active = np.array([sum(1 for r in rt if len(r) > 2) for rt in routes])

    if mode == "knee":
        lo, hi = objs.min(axis=0), objs.max(axis=0)
        span = np.where(hi - lo > 0, hi - lo, 1.0)
        norm = (objs - lo) / span
        i = int(np.argmin(np.hypot(norm[:, 0], norm[:, 1])))
    elif mode == "min-makespan":
        i = int(np.argmin(objs[:, 0]))
    elif mode == "min-energy":
        i = int(np.argmin(objs[:, 1]))
    elif mode == "max-fleet":
        i = int(np.argmax(active))
    else:
        raise SystemExit(f"unknown selection mode: {mode}")
    return objs[i].tolist(), routes[i], int(active[i])


def _save_animation(anim, base_no_ext: str, fps: int) -> str:
    """Write MP4 if ffmpeg is available, else fall back to a pillow GIF."""
    if shutil.which("ffmpeg"):
        out = base_no_ext + ".mp4"
        anim.save(out, writer=FFMpegWriter(fps=fps))
    else:
        out = base_no_ext + ".gif"
        print("    (ffmpeg not found — writing GIF via pillow)")
        anim.save(out, writer=PillowWriter(fps=fps))
    return out


# --- per-instance figures -------------------------------------------------------

def figures_for(instance: str, selection: str) -> None:
    runs_by_algo = {label: _runs(instance, algo) for algo, label in ARMS}

    # Pareto
    recs = {label: _records(r) for label, r in runs_by_algo.items()}
    ax = plot_pareto(recs, title=f"Pareto front — {instance}")
    _savefig(ax.figure, f"{instance}_pareto.pdf")

    # Convergence
    hists = {label: _histories(r) for label, r in runs_by_algo.items()}
    axes = plot_convergence(hists)
    _savefig(axes[0].figure, f"{instance}_convergence.pdf")

    # Routes — the chosen solution from NSGA-II (the variable-fleet arm).
    obj, routes, n_active = _select(runs_by_algo["NSGA-II"], selection)
    ax = plot_routes(routes, _coords(instance),
                     title=f"{instance} ({selection}) — makespan {obj[0]:.1f}s, "
                           f"energy {obj[1]:.0f}J, {n_active} drones")
    _savefig(ax.figure, f"{instance}_routes.pdf")
    print(f"  [{instance}] {selection}: makespan={obj[0]:.2f}s energy={obj[1]:.0f}J "
          f"drones={n_active}")


def _savefig(fig, name: str) -> None:
    os.makedirs(FIGURES, exist_ok=True)
    out = os.path.join(FIGURES, name)
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote figures/{name}")


# --- presentation interface: route + animation for an arbitrary solution --------

def render_one(routes_json: str, index: int, selection: str) -> None:
    d = json.load(open(routes_json))
    routes = d["routes"][index]
    obj = d["front"][index]
    instance = d["instance"]
    n_active = sum(1 for r in routes if len(r) > 2)
    coords = _coords(instance)
    tag = f"{instance}_{d['algorithm']}_{d['seed']}_sol{index}"
    ax = plot_routes(routes, coords,
                     title=f"{instance} {d['algorithm']} sol {index} — "
                           f"makespan {obj[0]:.1f}s, {n_active} drones")
    _savefig(ax.figure, f"{tag}_routes.pdf")
    anim = animate_routes(routes, coords)
    out = _save_animation(anim, os.path.join(FIGURES, f"{tag}_animation"), fps=20)
    plt.close(anim._fig)
    print(f"  wrote {os.path.relpath(out, ROOT)}  (makespan {obj[0]:.1f}s, "
          f"{n_active} drones)")


# --- CLI ------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Render Phase-6 figures from results/ JSONs.")
    p.add_argument("--instances", nargs="*", default=list(INSTANCES),
                   help="instance name(s) (default: all four)")
    p.add_argument("--selection", default="knee",
                   choices=("knee", "min-makespan", "min-energy", "max-fleet"),
                   help="which Pareto solution the route map/animation shows (default: knee)")
    p.add_argument("--animate-instance", default="eil51-k3",
                   help="instance whose chosen solution is animated (default: eil51-k3)")
    p.add_argument("--no-animation", action="store_true",
                   help="skip the (slower) animation render")
    p.add_argument("--routes-json", help="presentation mode: render route+animation "
                                         "for a specific run JSON")
    p.add_argument("--index", type=int, default=0,
                   help="front index within --routes-json (default 0)")
    args = p.parse_args(argv)

    if args.routes_json:                       # presentation interface
        render_one(args.routes_json, args.index, args.selection)
        return 0

    for instance in args.instances:
        print(f"== {instance} ==")
        figures_for(instance, args.selection)

    if not args.no_animation:
        inst = args.animate_instance
        print(f"== animation: {inst} ==")
        obj, routes, n_active = _select(_runs(inst, "nsga2"), args.selection)
        anim = animate_routes(routes, _coords(inst),
                              title=f"{inst} ({args.selection}) — "
                                    f"makespan {obj[0]:.1f}s, {n_active} drones")
        out = _save_animation(anim, os.path.join(FIGURES, f"{inst}_animation"), fps=20)
        plt.close(anim._fig)
        print(f"  wrote {os.path.relpath(out, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
