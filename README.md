<h1 align="center">Multi-UAV Surveillance Routing</h1>
<p align="center">
  <b>Bi-objective mission planning for drone fleets — NSGA-II vs. MOPSO</b><br>
  <i>Minimize mission time (makespan) and total fleet energy, simultaneously.</i>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.13-blue">
  <img alt="Tests" src="https://img.shields.io/badge/tests-106%20passing-brightgreen">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-lightgrey">
  <img alt="Status" src="https://img.shields.io/badge/status-complete-success">
</p>

---

### Watch the fleet fly

Three drones depart a shared depot, split the points of interest between them, and
return — a live decode of one Pareto-optimal solution on each of the four benchmark
instances.

<table>
  <tr>
    <td align="center">
      <img src="figures/eil51-k3_animation.gif" width="380"><br>
      <b>eil51-k3</b> · 50 POIs · 3 drones
    </td>
    <td align="center">
      <img src="figures/berlin52-k3_animation.gif" width="380"><br>
      <b>berlin52-k3</b> · 51 POIs · 3 drones
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="figures/eil76-k3_animation.gif" width="380"><br>
      <b>eil76-k3</b> · 75 POIs · 3 drones
    </td>
    <td align="center">
      <img src="figures/rat99-k3_animation.gif" width="380"><br>
      <b>rat99-k3</b> · 98 POIs · 3 drones
    </td>
  </tr>
</table>

---

## What this is

A **co-equal empirical comparison** of two multi-objective metaheuristics —
**NSGA-II** (a genetic algorithm) and **MOPSO** (particle swarm) — on a bi-objective
multi-UAV surveillance routing problem. A fleet of $K$ homogeneous rotary-wing drones
must cover every point of interest exactly once, trading off two conflicting goals:

- $f_1 = T_\text{makespan}$ — mission completion time, set by the *slowest* drone
  (min–max ⇒ **balance the workload**).
- $f_2 = E_\text{total}$ — total fleet energy under Dorling's linear power model
  $P = \alpha m + \beta$ (min–sum ⇒ **minimize total distance flown**).

Balancing the fleet forces extra depot returns, which costs energy — so the two
objectives genuinely conflict, and *fleet size itself becomes part of the Pareto
trade-off*. Under the constant-mass, constant-velocity assumption, $f_2$ is
proportional to total tour length, making this a physically grounded instance of the
**bi-objective mTSP** (Necula & Breaban mTSPLIB, $K = 3$).

Both algorithms share **one** decoder, fitness function, and distance table, so the
comparison isolates exactly one variable: the optimizer loop and its genotype.

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
