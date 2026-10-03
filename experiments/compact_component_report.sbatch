#!/bin/bash
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --job-name=amor-compact-report
set -euo pipefail
source /users/zliu328/agent-memory/experiments/final_component_environment.sh
export TMPDIR="${TMPDIR:-/tmp}"
"$python_bin" -u -m experiments.compact_component_run report --output "$1"
