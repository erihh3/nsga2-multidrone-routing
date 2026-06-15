"""The 80-run harness: instances x algorithms x seeds, filterable and resumable.

Generalizes ``scripts/eval_eil51.py`` (the Phase-4 reference) to the full matrix
(4 instances x 2 algorithms x 10 seeds). Each run writes
``results/<instance>_<algo>_<seed>.json`` (the same schema eval_eil51 established,
plus ``n_active_drones``) and appends a bookkeeping row to ``results/log.csv``.
Metric aggregation is **not** done here — that is Step 2, run after the full sweep
returns. This module only runs, persists, and logs.

CLI filters (``--instance``, ``--algorithm``, ``--seed``; repeatable or
comma-separated; no flags = full matrix) let the user split the sweep across
sessions or redo a single failed cell. Resume: a run is skipped if ``log.csv``
already holds a row with the same ``(algorithm, instance, seed)`` *and* matching
per-algorithm hyperparameters.

Budget parity (co-equality): MOPSO's ``iters`` is derived per instance from
NSGA-II's *measured* mean evaluation count over its 10 seeds
(``config.parity_iters``), and persisted to ``results/<instance>_parity.json`` so a
session that runs only MOPSO can reuse it. MOPSO for an instance therefore needs
all 10 NSGA-II runs on disk first; if they are missing the runner refuses to guess
a budget (it never inflates NSGA-II to flatter the comparison).

Seeding stays per run (the optimizers call ``set_all_seeds`` inside ``run``), so a
filtered cell is bit-identical to the same cell inside a full sweep.

Usage:
    python -m uav.experiment.runner                         # full 80-run matrix
    python -m uav.experiment.runner --instance eil51-k3     # one instance
    python -m uav.experiment.runner --algorithm nsga2       # one arm, all instances
    python -m uav.experiment.runner --instance eil51-k3 --algorithm mopso --seed 3
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
from datetime import datetime, timezone

import numpy as np

from uav.algorithms.base import RunResult
from uav.algorithms.dmopso import DiscreteMOPSO
from uav.algorithms.mopso import MOPSO
from uav.algorithms.nsga2 import NSGA2
from uav.experiment.config import INSTANCES, SEEDS, Budget, Hyperparams, parity_iters
from uav.problem.instance import load_instance

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
RESULTS = os.path.join(ROOT, "results")
INSTANCES_DIR = os.path.join(ROOT, "instances")

# The DEFAULT sweep set (no --algorithm filter) stays the two co-equal baselines.
# ``dmopso`` is the scoped encoding-diagnostic MOPSO *variant* (Attempt C): a valid
# but OPT-IN value — runnable only via explicit ``--algorithm dmopso`` — so adding
# it cannot change the default sweep's behaviour. Validation is against
# ``_OPTIMIZERS`` (below), not this tuple, so dmopso is accepted while staying out
# of the default matrix.
ALGORITHMS: tuple[str, ...] = ("nsga2", "mopso")
_OPTIMIZERS = {"nsga2": NSGA2, "mopso": MOPSO, "dmopso": DiscreteMOPSO}

# Hyperparams is one combined dataclass holding BOTH arms' fields; each run only
# uses (and so persists) its own subset. Identical to eval_eil51's convention so
# the JSON schema and the resume hyperparameter match are consistent across both.
# dmopso reuses MOPSO's hyperparameter fields verbatim (it differs only in the
# order genotype + its update; w/c1/c2/turbulence schedule are shared).
_HP_FIELDS = {
    "nsga2": ("pop", "gens", "pcx", "pmut", "pmut_counts"),
    "mopso": ("swarm", "iters", "archive_size", "grid_divisions", "w_inertia",
              "c1", "c2", "vmax_frac", "mut_rate", "mut_floor"),
}
_HP_FIELDS["dmopso"] = _HP_FIELDS["mopso"]

LOG_HEADER = [
    "algorithm", "instance", "seed", "hyperparams_json", "n_evals",
    "wall_clock_s", "front_size", "n_active_min", "n_active_max", "timestamp",
]


# --- persistence helpers --------------------------------------------------------

def _hp_subset(hp: Hyperparams, algo: str) -> dict:
    d = dataclasses.asdict(hp)
    sub = {k: d[k] for k in _HP_FIELDS[algo]}
    if d.get("extra"):                       # only when non-empty
        sub["extra"] = d["extra"]
    return sub


def _hp_key(hp: Hyperparams, algo: str) -> str:
    """Canonical string of the per-algorithm hyperparameters for resume matching."""
    return json.dumps(_hp_subset(hp, algo), sort_keys=True)


def _instance_spec(instance: str) -> tuple[str, int]:
    """Map an instance name like ``eil51-k3`` to its ``.tsp`` path and K."""
    base, _, k_tag = instance.partition("-k")
    k = int(k_tag) if k_tag else 3
    return os.path.join(INSTANCES_DIR, f"{base}.tsp"), k


def _serialize_run(res: RunResult, algo: str, instance: str, seed: int, hp: Hyperparams) -> dict:
    return {
        "algorithm": algo,
        "instance": instance,
        "seed": seed,
        "hyperparams": _hp_subset(hp, algo),
        "n_evals": res.n_evals,
        "wall_clock_s": res.wall_clock_s,
        "front": [[float(s.makespan), float(s.energy)] for s in res.final_front],
        "routes": [[[int(p) for p in r] for r in s.routes] for s in res.final_front],
        # Variable-fleet metadata: active-drone count per front point, parallel to
        # `front`/`routes`, so Pareto plots can be colored by fleet size later.
        "n_active_drones": [s.n_active_drones for s in res.final_front],
        "history": [
            {"gen": int(g.gen),
             "best": [float(x) for x in g.best],
             "mean": [float(x) for x in g.mean],
             "worst": [float(x) for x in g.worst]}
            for g in res.history
        ],
    }


def _run_json_path(results_dir: str, instance: str, algo: str, seed: int) -> str:
    return os.path.join(results_dir, f"{instance}_{algo}_{seed}.json")


def _save_run(payload: dict, results_dir: str, instance: str, algo: str, seed: int) -> None:
    os.makedirs(results_dir, exist_ok=True)
    with open(_run_json_path(results_dir, instance, algo, seed), "w") as fh:
        json.dump(payload, fh)


# --- log.csv (resume bookkeeping) ----------------------------------------------

def _log_path(results_dir: str) -> str:
    return os.path.join(results_dir, "log.csv")


def _load_log(results_dir: str) -> list[dict]:
    path = _log_path(results_dir)
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _append_log_row(results_dir: str, row: dict) -> None:
    os.makedirs(results_dir, exist_ok=True)
    path = _log_path(results_dir)
    exists = os.path.exists(path)
    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_HEADER)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _already_done(log_rows: list[dict], algo: str, instance: str, seed: int, hp_key: str) -> bool:
    """A run is done iff a log row matches algo/instance/seed AND the hyperparams.

    Matching hyperparameters (not just the cell) means a redo with changed config
    re-runs rather than being falsely skipped — and, for MOPSO, that the recorded
    parity ``iters`` matches the one we are about to use.
    """
    for r in log_rows:
        if (r["algorithm"] == algo and r["instance"] == instance
                and int(r["seed"]) == seed and r["hyperparams_json"] == hp_key):
            return True
    return False


# --- per-instance budget parity -------------------------------------------------

def _parity_path(results_dir: str, instance: str) -> str:
    return os.path.join(results_dir, f"{instance}_parity.json")


def _resolve_parity(results_dir: str, instance: str, swarm: int) -> int | None:
    """MOPSO ``iters`` for an instance, from NSGA-II's measured 10-seed mean.

    Computed once from the complete set of NSGA-II per-run JSONs and cached to
    ``<instance>_parity.json`` (stable thereafter, so MOPSO's budget — and its
    resume key — does not drift if a single NSGA-II seed is later redone). Returns
    ``None`` if the NSGA-II runs are not all present yet (caller refuses MOPSO).
    """
    cache = _parity_path(results_dir, instance)
    if os.path.exists(cache):
        with open(cache) as fh:
            return int(json.load(fh)["iters"])

    evals = []
    for seed in SEEDS:
        p = _run_json_path(results_dir, instance, "nsga2", seed)
        if not os.path.exists(p):
            return None
        with open(p) as fh:
            evals.append(json.load(fh)["n_evals"])

    mean = float(np.mean(evals))
    iters = parity_iters(mean, swarm)
    os.makedirs(results_dir, exist_ok=True)
    with open(cache, "w") as fh:
        json.dump({"measured_nsga2_mean": mean, "iters": iters,
                   "n_seeds": len(SEEDS), "swarm": swarm}, fh, indent=2)
    return iters


# --- one run --------------------------------------------------------------------

def _execute(algo: str, instance: str, inst, seed: int, hp: Hyperparams,
             results_dir: str, log_rows: list[dict]) -> RunResult:
    """Run one (algo, instance, seed), persist its JSON, append a log row."""
    res = _OPTIMIZERS[algo](inst, Budget(), hp).run(seed=seed)
    _save_run(_serialize_run(res, algo, instance, seed, hp), results_dir, instance, algo, seed)
    n_active = [s.n_active_drones for s in res.final_front]
    row = {
        "algorithm": algo,
        "instance": instance,
        "seed": seed,
        "hyperparams_json": _hp_key(hp, algo),
        "n_evals": res.n_evals,
        "wall_clock_s": f"{res.wall_clock_s:.4f}",
        "front_size": len(res.final_front),
        "n_active_min": min(n_active) if n_active else "",
        "n_active_max": max(n_active) if n_active else "",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _append_log_row(results_dir, row)
    log_rows.append({k: str(v) for k, v in row.items()})   # keep resume view current
    active_lo = min(n_active) if n_active else "-"
    active_hi = max(n_active) if n_active else "-"
    print(f"  [run] {algo:<5} {instance:<12} seed {seed:<2} "
          f"n_evals={res.n_evals} wall={res.wall_clock_s:.1f}s "
          f"front={len(res.final_front)} active={active_lo}-{active_hi}")
    return res


# --- orchestration --------------------------------------------------------------

def run_all(instances=INSTANCES, algorithms=ALGORITHMS, seeds=SEEDS,
            results_dir: str = RESULTS, hp_overrides: dict | None = None) -> dict:
    """Run the (filtered) matrix. Returns a {completed, skipped, errors} summary.

    ``hp_overrides`` lets a smoke test shrink the budget (e.g. ``pop``/``gens``);
    the full sweep passes nothing and uses the paper defaults.
    """
    hp_overrides = hp_overrides or {}
    log_rows = _load_log(results_dir)
    completed = skipped = 0
    errors: list[str] = []

    for instance in instances:
        path, k = _instance_spec(instance)
        inst = load_instance(path, k=k)

        # --- NSGA-II first: it defines the parity budget MOPSO must match. -------
        if "nsga2" in algorithms:
            hp = dataclasses.replace(Hyperparams(), **hp_overrides)
            for seed in seeds:
                if _already_done(log_rows, "nsga2", instance, seed, _hp_key(hp, "nsga2")):
                    print(f"  [skip] nsga2 {instance} seed {seed} (already in log.csv)")
                    skipped += 1
                    continue
                _execute("nsga2", instance, inst, seed, hp, results_dir, log_rows)
                completed += 1

        # --- MOPSO at measured-eval parity (brought DOWN to NSGA-II's mean). -----
        if "mopso" in algorithms:
            iters = _resolve_parity(results_dir, instance, Hyperparams().swarm)
            if iters is None:
                msg = (f"MOPSO requested for {instance} but its 10 NSGA-II runs are "
                       f"not all on disk — run NSGA-II for {instance} first "
                       f"(budget parity needs the measured mean).")
                print(f"  [error] {msg}")
                errors.append(msg)
                continue
            hp = dataclasses.replace(Hyperparams(iters=iters), **hp_overrides)
            for seed in seeds:
                if _already_done(log_rows, "mopso", instance, seed, _hp_key(hp, "mopso")):
                    print(f"  [skip] mopso {instance} seed {seed} (already in log.csv)")
                    skipped += 1
                    continue
                _execute("mopso", instance, inst, seed, hp, results_dir, log_rows)
                completed += 1

        # --- discrete-MOPSO (Attempt C) at the SAME measured-eval parity. --------
        # Opt-in only (never in the default set). It re-evaluates the whole swarm
        # each iteration like MOPSO, so it reuses the identical parity budget and,
        # like MOPSO, refuses to guess one without NSGA-II's 10 runs on disk.
        if "dmopso" in algorithms:
            iters = _resolve_parity(results_dir, instance, Hyperparams().swarm)
            if iters is None:
                msg = (f"dmopso requested for {instance} but its 10 NSGA-II runs are "
                       f"not all on disk — run NSGA-II for {instance} first "
                       f"(budget parity needs the measured mean).")
                print(f"  [error] {msg}")
                errors.append(msg)
                continue
            hp = dataclasses.replace(Hyperparams(iters=iters), **hp_overrides)
            for seed in seeds:
                if _already_done(log_rows, "dmopso", instance, seed, _hp_key(hp, "dmopso")):
                    print(f"  [skip] dmopso {instance} seed {seed} (already in log.csv)")
                    skipped += 1
                    continue
                _execute("dmopso", instance, inst, seed, hp, results_dir, log_rows)
                completed += 1

    print(f"\nsummary: completed={completed} skipped={skipped} errors={len(errors)}")
    return {"completed": completed, "skipped": skipped, "errors": errors}


# --- CLI ------------------------------------------------------------------------

def _split(values: list[str] | None) -> list[str] | None:
    """Flatten repeated and comma-separated CLI values; None -> None (no filter)."""
    if not values:
        return None
    out: list[str] = []
    for v in values:
        out.extend(part for part in v.split(",") if part)
    return out


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filterable, resumable experiment runner.")
    p.add_argument("--instance", action="append",
                   help="instance name(s), repeatable or comma-separated (default: all)")
    p.add_argument("--algorithm", action="append",
                   help="nsga2 and/or mopso (default: both); dmopso is the opt-in "
                        "encoding-diagnostic variant — never run by default. "
                        "Repeatable or comma-separated.")
    p.add_argument("--seed", action="append",
                   help="seed(s), repeatable or comma-separated (default: the 10 fixed seeds)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    instances = _split(args.instance) or list(INSTANCES)
    algorithms = _split(args.algorithm) or list(ALGORITHMS)
    seeds = [int(s) for s in (_split(args.seed) or [str(s) for s in SEEDS])]

    bad_i = [i for i in instances if i not in INSTANCES]
    # Validate against the registry (includes opt-in ``dmopso``), not the default set.
    bad_a = [a for a in algorithms if a not in _OPTIMIZERS]
    if bad_i:
        raise SystemExit(f"unknown instance(s): {bad_i}; valid: {list(INSTANCES)}")
    if bad_a:
        raise SystemExit(f"unknown algorithm(s): {bad_a}; valid: {list(_OPTIMIZERS)}")

    print(f"running: instances={instances} algorithms={algorithms} seeds={seeds}\n")
    summary = run_all(instances=instances, algorithms=algorithms, seeds=seeds)
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
