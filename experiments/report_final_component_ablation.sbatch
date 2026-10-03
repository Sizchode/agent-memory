#!/bin/bash
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --job-name=amor-component-report
set -euo pipefail
source /users/zliu328/agent-memory/experiments/final_component_environment.sh
"$python_bin" -u -m experiments.report_final_component_ablation "$1"
