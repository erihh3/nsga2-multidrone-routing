#!/usr/bin/env bash
# Reproduce every result from a clean clone: 4 instances x 2 algorithms x 10 seeds.
# Phase 5 wires up the runner; this script is the single reproduction entry point.
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -d .venv ]]; then
    uv venv --python 3.13 .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .

# Full matrix by default; any CLI filters pass straight through to the runner:
#   ./run_experiments.sh                                  # 4 x 2 x 10 = 80 runs
#   ./run_experiments.sh --instance eil51-k3              # one instance
#   ./run_experiments.sh --algorithm nsga2                # one arm, all instances
#   ./run_experiments.sh --instance eil51-k3 --seed 0,1,2 # split across sessions
exec python -m uav.experiment.runner "$@"
