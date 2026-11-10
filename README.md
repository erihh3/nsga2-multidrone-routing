<h1 align="center">Multi-UAV Surveillance Routing</h1>
<p align="center">
  <b>Bi-objective mission planning for drone fleets — NSGA-II vs. MOPSO</b><br>
  <i>Minimize mission time (makespan) and total fleet energy, simultaneously.</i>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.14-blue">
  <img alt="Tests" src="https://img.shields.io/badge/tests-155%20passing-brightgreen">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-lightgrey">
  <img alt="Status" src="https://img.shields.io/badge/status-complete-success">
</p>

---

### Example solutions

Each animation decodes one Pareto-optimal solution: three drones depart a shared
depot, partition the points of interest between them, and return. One example per
benchmark instance.

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

## Getting started

**Requirements:** Python ≥ 3.11 (3.14 pinned for reproducibility) and
[`uv`](https://github.com/astral-sh/uv). All runtime dependencies (`numpy`, `scipy`,
`matplotlib`, `seaborn`, `deap`) are pinned in `requirements.txt`.

**1 — Create the environment and install the package:**

```bash
uv venv --python 3.14 .venv
source .venv/bin/activate
uv pip install -r requirements.txt   # pinned runtime deps
uv pip install -e .                  # install `uav` so `import uav` resolves from src/
```

**2 — Provide the benchmark instances.** Place the four instances in `instances/`:

```
instances/eil51.tsp  instances/berlin52.tsp  instances/eil76.tsp  instances/rat99.tsp
```

**3 — Verify the build.** The suite includes the hand-calculated fitness
check that anchors every downstream number; it must pass before running experiments:

```bash
pytest                               # 155 tests — must be green
```

**4 — Run the exploratory sweep with one command:**

```bash
./run_experiments.sh                 # 4x2x30 sweep -> results/, then figures -> figures/
```

`run_experiments.sh` bootstraps the venv if missing, runs (and *resumes*) the sweep,
and chains `scripts/make_figures.py`. It uses the exploratory defaults
(`Hyperparams.gens = 500`), so it does **not** reproduce the DAUS 2027 paper: for that,
see the next section.

## DAUS 2027 paper: reproduce

The paper's data is `results_cobudget/` (360 runs: 4 instances × NSGA-II, MOPSO and
DMOPSO × 30 seeds), **not** `results/`, which only holds the exploratory sweep above.
`results_cobudget/` and `instances/` are not in git. Download `results_cobudget.tar.gz`,
`instances.tar.gz` and `SHA256SUMS` from the GitHub release
[`daus2027-v1`](https://github.com/erihh3/nsga2-multidrone-routing/releases/tag/daus2027-v1)
into the repository root. The paper's environment is Python 3.14, with exact package
versions in `docs/daus2027/environment.txt`. Definitions, calibration history and the
map from each table and figure to its inputs are in
[`docs/daus2027/README.md`](docs/daus2027/README.md).

**Audit without optimizing.** No optimizer runs: this recomputes every check, metric,
table and figure from the saved runs.

```bash
sha256sum -c SHA256SUMS
tar xzf results_cobudget.tar.gz && tar xzf instances.tar.gz
sha256sum -c docs/daus2027/results_cobudget.sha256
PYTHONPATH=src .venv/bin/python scripts/rebuild_daus2027.py
```

**Full rerun, separate and optional.** Give it a new directory; with the default it
skips every run already in `results_cobudget/`.

```bash
PYTHONPATH=src .venv/bin/python scripts/eval_cobudget.py --results-dir <new dir>
```

About 3.1 h sequential (0.88 h NSGA-II, 0.99 h MOPSO, 1.22 h DMOPSO). This script is the
DAUS preset (population 100, 3000 generations, swarm 100); `Hyperparams.gens = 500` is the
exploratory default of `run_experiments.sh`. MOPSO's 2790 iterations come from
`results_cobudget/*_parity.json`, which this script derives from NSGA-II's measured mean
evaluation count.

**Instances.** Coordinates are the TSPLIB95 problems
([source](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/)), as used by the
[mTSPLIB](https://profs.info.uaic.ro/~mtsplib/) single-depot multiple-TSP benchmark
(Necula, Breaban & Raschip, HAIS 2015). Node 1 of each file is the depot; $K = 3$ is a
parameter, not part of the file. `instances.tar.gz` holds the exact files the runs used;
a copy from anywhere else must match these checksums to count as the same file.

| File | SHA-256 |
| --- | --- |
| `instances/eil51.tsp` | `66c05ad077ffd64d5778e358bbd16be049f8c8ae880cbd24f034487331d1ccce` |
| `instances/berlin52.tsp` | `8496a5838133e3c3eb25a9fa8893b1ac302593fc8dc325350643a0cabd8abadc` |
| `instances/eil76.tsp` | `74a81d109969e0e46ecc62ff9bd051bfce36b508326bfcd22168e58d03af2447` |
| `instances/rat99.tsp` | `f880e9afe0b655b5b0deae0b8835a5b48622172e28a1c1fc9bf70100cb97b46a` |

## Command reference

The build script wraps two independently runnable stages: the **runner** (optimizer
sweep → `results/`) and **make_figures** (results → PDFs + animation). Every reported
figure is `n=30` (30 fixed seeds per instance×algorithm); never quote a single seed.

| Command | What it does |
| --- | --- |
| `./run_experiments.sh` | Full pipeline: 240-run sweep (4×2×30), then all figures. |
| `./run_experiments.sh --no-figures` | Sweep only (skip the figure step). |
| `./run_experiments.sh --instance eil51-k3` | Restrict the sweep to one instance. |
| `./run_experiments.sh --algorithm nsga2` | Run one arm across all instances. |
| `./run_experiments.sh --seed 0,1,2` | Split seeds across sessions (resumable). |
| `python -m uav.experiment.runner` | Run the sweep directly (same filters). |
| `python scripts/make_figures.py` | Render figures from `results/` only (no re-run). |
| `pytest` | Run the 155-test suite. |

**Runner filters** (`python -m uav.experiment.runner`, all repeatable / comma-separated):

- `--instance` — `eil51-k3`, `berlin52-k3`, `eil76-k3`, `rat99-k3` (default: all)
- `--algorithm` — `nsga2`, `mopso` (default: both); `dmopso` is an opt-in
  encoding-diagnostic arm, never run by default
- `--seed` — one or more of the 30 fixed seeds (default: all 30)

**Figure options** (`python scripts/make_figures.py`):

- `--instances eil51-k3 …` — which instances to render (default: all with a complete arm set)
- `--selection {knee,min-makespan,min-energy,max-fleet}` — which Pareto solution to draw (default: `knee`)
- `--animate-instance eil51-k3` — instance to animate; `--no-animation` to skip

```bash
# Examples
python -m uav.experiment.runner --instance eil51-k3 --algorithm nsga2   # one cell
python scripts/make_figures.py --instances rat99-k3 --selection min-makespan
```
