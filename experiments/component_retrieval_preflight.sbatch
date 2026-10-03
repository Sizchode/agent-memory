#!/bin/bash
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=amor-component-replay
set -euo pipefail
source /users/zliu328/agent-memory/experiments/final_component_environment.sh
export TMPDIR="${TMPDIR:-/tmp}"
"$python_bin" -u -m experiments.preflight_final_components "$@"
