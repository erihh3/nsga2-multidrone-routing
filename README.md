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
tests/                # one file per core module
instances/            # mTSPLIB .tsp / .k files (downloaded separately)
results/              # per-run JSON + log.csv
figures/              # output PDFs
```

## Build order (phase-gated)

`instance.py` → `decode.py` → `fitness.py` (+ hand-calc test, **hard gate**) →
`nsga2.py` → `mopso.py` → `metrics`/`reference_front`/`stats` → `runner` → `viz`.
Do not advance past a failing gate.

## Reproducing results

```bash
./run_experiments.sh
```

10 fixed seeds per (instance, algorithm); every run logged to `results/log.csv`.
Never quote a single-seed number — every reported figure is n=10 (median + IQR).

## Status

Phase 0 complete: skeleton + pinned environment. Core modules are contract stubs
pending Phase 1.
