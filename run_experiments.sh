#!/usr/bin/env bash
# Reproduce every result AND figure from a clean clone.
#   sweep  (4 instances x 2 algorithms x 10 seeds)  -> results/
#   figures (pareto / convergence / routes / animation) -> figures/
# Phase 5 wired the runner; Phase 6 added the figure driver; this script chains them
# so a single command reproduces the whole deliverable.
set -euo pipefail

cd "$(dirname "$0")"

# --no-figures runs only the sweep (the pre-Phase-6 behaviour). Any other flags
# pass straight through to the runner (--instance / --algorithm / --seed).
RENDER_FIGURES=1
RUNNER_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--no-figures" ]]; then
        RENDER_FIGURES=0
    else
        RUNNER_ARGS+=("$arg")
    fi
done

if [[ ! -d .venv ]]; then
    uv venv --python 3.13 .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .

# --- 1. the sweep (resumes; CLI filters pass through) -------------------------
#   ./run_experiments.sh                                  # 4 x 2 x 10 = 80 runs
#   ./run_experiments.sh --instance eil51-k3              # one instance
#   ./run_experiments.sh --algorithm nsga2                # one arm, all instances
#   ./run_experiments.sh --instance eil51-k3 --seed 0,1,2 # split across sessions
#   ./run_experiments.sh --no-figures                     # sweep only
python -m uav.experiment.runner "${RUNNER_ARGS[@]}"

[[ "$RENDER_FIGURES" -eq 1 ]] || exit 0

# --- 2. figures, read from results/ only (no optimizer re-run) ----------------
# Render only instances whose full arm set (nsga2 + mopso) is on disk, so a
# filtered or partial sweep does not crash the figure step. Animate the first
# such instance (default eil51-k3 if available); skip animation if none qualify.
FIG_INSTANCES=()
for inst in eil51-k3 berlin52-k3 eil76-k3 rat99-k3; do
    if compgen -G "results/${inst}_nsga2_*.json" >/dev/null \
       && compgen -G "results/${inst}_mopso_*.json" >/dev/null; then
        FIG_INSTANCES+=("$inst")
    fi
done

if [[ ${#FIG_INSTANCES[@]} -eq 0 ]]; then
    echo "no instance has a complete nsga2+mopso result set on disk — skipping figures"
    exit 0
fi

ANIMATE="eil51-k3"
printf '%s\n' "${FIG_INSTANCES[@]}" | grep -qx "$ANIMATE" || ANIMATE="${FIG_INSTANCES[0]}"
python scripts/make_figures.py --instances "${FIG_INSTANCES[@]}" --animate-instance "$ANIMATE"
