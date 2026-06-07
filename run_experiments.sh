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

# Phase 5: python -m uav.experiment.runner
echo "run_experiments.sh: runner not yet implemented (Phase 5)." >&2
exit 1
