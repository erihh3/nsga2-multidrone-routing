"""Diagnostic: overlay the DMOPSO Pareto front against MOPSO (and optionally
NSGA-II) on one instance, coloured by active-drone count.

This is a DIAGNOSTIC view, deliberately separate from the paper-figure pipeline
(``scripts/make_figures.py``), which by design plots only the two co-equal arms
and never dmopso. It reuses that script's disk loaders and the shared
``uav.viz.pareto.plot_pareto`` so the rendering style matches the paper figures
exactly (colour == fleet size, marker shape == arm). Reads JSONs only; never
re-runs an optimizer.

Usage:
  .venv/bin/python scripts/plot_dmopso_pareto.py                  # eil51-k3: DMOPSO vs MOPSO
  .venv/bin/python scripts/plot_dmopso_pareto.py --with-nsga2     # add NSGA-II for context
  .venv/bin/python scripts/plot_dmopso_pareto.py --instance eil51-k3 --out /tmp/dmopso.png
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))   # make_figures.py lives here
import make_figures as mf                        # noqa: E402  (Agg backend + loaders)
from uav.viz.pareto import plot_pareto           # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instance", default="eil51-k3")
    p.add_argument("--with-nsga2", action="store_true",
                   help="overlay NSGA-II too (shows the low-fleet region MOPSO/DMOPSO miss)")
    p.add_argument("--out", default=None,
                   help="output path (default: figures/<instance>_dmopso_pareto.png)")
    args = p.parse_args(argv)

    # DMOPSO first so it draws on top; MOPSO for the head-to-head; NSGA-II optional.
    arms = [("dmopso", "DMOPSO"), ("mopso", "MOPSO")]
    if args.with_nsga2:
        arms.append(("nsga2", "NSGA-II"))

    recs = {label: mf._records(mf._runs(args.instance, algo)) for algo, label in arms}
    if not any(recs.values()):
        sys.exit(f"no runs on disk for {args.instance} (looked for {[a for a,_ in arms]})")

    ax = plot_pareto(recs, title=f"Pareto front — {args.instance} (DMOPSO diagnostic)")
    out = args.out or os.path.join(mf.FIGURES, f"{args.instance}_dmopso_pareto.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ax.figure.savefig(out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
