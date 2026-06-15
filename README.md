# uav-routing


## Setup

```bash
uv venv --python 3.13 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .            # make `import uav` work
```

## Layout

```
src/uav/
  seeds.py            # seed setup for random + numpy
  problem/            # instance loader, decoders, fitness (shared core)
  algorithms/         # base ABC + nsga2 (DEAP) + mopso (numpy)
  evaluation/         # metrics, reference front, stats
  experiment/         # config + 4x2x10 = 80-run harness
  viz/                # pareto / convergence / routes / animation
scripts/              # make_figures.py (figure driver) + eval/sanity helpers
tests/                # one file per core module
instances/            # mTSPLIB .tsp / .k files (downloaded separately)
results/              # per-run JSON + log.csv
figures/              # output PDFs + drone animation
```

## Build order (phase-gated)

`instance.py` → `decode.py` → `fitness.py` (+ hand-calc test, **hard gate**) →
`nsga2.py` → `mopso.py` → `metrics`/`reference_front`/`stats` → `runner` → `viz`.

## Reproducing results

```bash
./run_experiments.sh                  # sweep -> results/, then figures -> figures/
./run_experiments.sh --no-figures     # sweep only (the runner step)
```

`run_experiments.sh` runs (and resumes) the 4 x 2 x 10 = 80-run sweep — persisting
per-run JSON + `log.csv` — then chains `scripts/make_figures.py`, which reads those
JSONs only (no optimizer re-run) and writes the pareto/convergence/route PDFs + the
drone animation. CLI filters (`--instance`, `--algorithm`, `--seed`) pass through to
the runner; the figure step then renders every instance with a complete arm set on
disk. To render figures alone (sweep already done):

```bash
.venv/bin/python scripts/make_figures.py
```

10 fixed seeds per (instance, algorithm); every run logged to `results/log.csv`.
Never quote a single-seed number — every reported figure is n=10 (median + IQR).

## Status

All code phases complete (Phase 0–6): shared problem core, both optimizers
(NSGA-II + MOPSO), the 80-run sweep harness, evaluation/metrics/stats, and the
visualization layer are implemented and tested — `pytest` green (**106 tests**).
Per-phase reading guides live in `planning/PHASE*_GUIDE.md`.
