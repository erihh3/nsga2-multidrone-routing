# DAUS 2027 paper: code, calibration, definitions, file map

How to rebuild is in the top-level `README.md` ("DAUS 2027 paper: reproduce"). This page says which code produced the data, how its settings were chosen, how each metric is defined, and where each number comes from.

## Generating code

The 360 runs in `results_cobudget/` come from `scripts/eval_cobudget.py` over the `src/uav/` of commit `48c2da4`. Since then `src/uav/` has changed in two places only (`git diff --stat 48c2da4 daus2027-v1 -- src/uav`): a plot annotation removed from `viz/pareto.py`, and comment lines in `problem/fitness.py`. Neither can change a run. `environment.txt` records the interpreter and the replay check: rat99-k3 seed 29 and eil51-k3 seed 0 (NSGA-II and MOPSO) reproduce their saved runs exactly.

## Calibration history

The settings were not chosen independently of the test instances. There was **no train/test split and no hyperparameter sweep**.

- **MOPSO.** `vmax_frac` (0.5 to 0.1) and the turbulence rate and floor (0.5/0.1 to 0.3/0.05) were lightly calibrated on eil51-k3, with 3 seeds, at the ≈46.5k-evaluation parity budget then in use. Ten combinations of these two levers were scored by GD and best energy against NPS, and the one with the best GD and best energy at median NPS ≥ 3 was kept. **eil51-k3 is also a test instance.** The paper's budget is 279,100 evaluations, about six times larger, and the settings were not recalibrated there. `w`, `c1`, `c2`, grid and archive stayed at their Coello defaults.
- **NSGA-II.** `pmut = 0.3` was "tuned lightly"; the project notes record no procedure for it. The other parameters come from precedent or original-paper defaults.

Sources: the comments in `src/uav/experiment/config.py`, and two project notes that are not in git, `planning/methodology.md` (§1.5 and the MOPSO configuration paragraph) and `planning/MOPSO_INVESTIGATION.md` (Attempt A).

## Definitions

Code: `src/uav/evaluation/` and `scripts/rebuild_daus2027.py`. Both objectives are minimized.

- **Dominance.** `a` dominates `b` iff `a_j <= b_j` for all `j` and `a_j < b_j` for some `j`. Coverage alone uses weak dominance: `C(A,B)` is the share of `B` with some `a <= b` on both objectives, so equal points count.
- **Deduplication.** Before every metric, each run's front is collapsed to its distinct objective vectors, by exact floating-point equality (`np.unique`). The run JSONs keep the raw fronts.
- **Reference front R.** Per instance, the union of the 30 NSGA-II and 30 MOPSO final fronts, deduplicated by objective vector and filtered to non-dominated points (`reference_fronts.csv`). Adding DMOPSO leaves R unchanged (checked in `hv_audit`).
- **Normalization.** `z_j = (f_j - a_j) / (b_j - a_j)`, with `a_j`, `b_j` the min and max of objective `j` over R (`normalization.csv`), the same for every method. There is **no clipping**: a point worse than R's extremes has `z_j > 1`, so IGD > 1 is possible and is not an error. All distances and areas below use `z`, with R normalized too.
- **IGD** `= (1/|R|) * sum_{r in R} min_{a in A} ||r - a||_2`, the arithmetic mean of the distances (exponent 1).
- **GD** `= sqrt( sum_{a in A} d_a^2 ) / |A|`, with `d_a = min_{r in R} ||a - r||_2`. The root is taken before dividing by `|A|`, so this is neither the mean distance nor the RMS distance.
- **HV.** Exact 2D area dominated by the front and bounded by `r`, divided by `r_1 * r_2`. `r_j` = factor × the largest `z_j` pooled over the NSGA-II and MOPSO runs of the instance. The reported factor is 1.1 (`r_j` in `normalization.csv`); sensitivity uses 1.05, 1.1, 1.2 and 1.5 (`hv_sensitivity.csv`). A point not strictly better than `r` on both objectives contributes nothing.
- **Spacing** (Schott). `sqrt( sum_i (d_i - mean d)^2 / (n-1) )`, `d_i` the L1 nearest-neighbour distance. It is NaN below 3 points, and those runs drop out of that test (`n_nsga2`, `n_mopso` in `stats_28tests.csv`). **DM:** diagonal of the front's bounding box, 0 for one point. **NPS:** distinct points. **CT:** the run's wall-clock seconds.
- **Tests.** Two-sided Mann-Whitney U, NSGA-II against MOPSO; Holm over the 28 tests (7 metrics × 4 instances), α = 0.05. HV medians: percentile bootstrap of the difference, 10,000 resamples, seed 2027, each method resampled on its own.

## File map

Inputs under `results_cobudget/`: `R` = `<instance>_<method>_<seed>.json`, `P` = `<instance>_parity.json`, `M` = `<instance>_metrics.json`. `I` = `instances/*.tsp`. Every other file named here is in this folder.

| Output | Rebuild step | Inputs |
| --- | --- | --- |
| `solution_check.csv` | `check_solutions` | R, I |
| `initial_fleet_frequencies.csv` | `initial_fleet` | R (NSGA-II, MOPSO), I |
| `budget_per_run.csv`, `budget_summary.csv`, `config_paper.json` | `budget` | R, P, I |
| `hv_bounds.csv`, `hv_points.csv`, `hv_per_run.csv`, `hv_sensitivity.csv` | `hv_audit` | R, M |
| `coverage.csv` | `coverage` | R (NSGA-II, MOPSO) |
| `stats_28tests.csv`, `hv_effects.csv`, `fleet_coverage.csv`, `nps_by_instance.csv`, `best_energy.csv` | `stats` | R, M, `hv_per_run.csv` |
| `normalization.csv`, `reference_fronts.csv`, `metrics_per_run.csv` (240 runs; no DMOPSO) | `export_analysis` | R (NSGA-II, MOPSO), `hv_per_run.csv`, `stats_28tests.csv` |
| Table 1: `table1.csv`, `table1.tex` | `tables` | `config_paper.json`, `budget_summary.csv` |
| Table 2: `table2.csv`, `table2.tex` | `tables` | `stats_28tests.csv`, `fleet_coverage.csv` |
| Paper Fig. 2: `figures/fig1_fronts.{pdf,png}` | `figures` | R (NSGA-II, MOPSO; eil51-k3, rat99-k3) |
| Paper Fig. 3: `figures/fig2_convergence.{pdf,png}` | `figures` | R (NSGA-II, MOPSO; eil51-k3) |

`coverage.csv` has `c_min = c_max` on every row, so that one value holds for each of the 900 run pairs of the row and no per-pair file is needed. Paper Fig. 1 is a worked example of the decoder and is not built from the runs. Not written by the rebuild: `environment.txt` and the `*.sha256` manifest.
