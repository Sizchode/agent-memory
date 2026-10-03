#!/bin/bash
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=amor-compact-prepare
#SBATCH --array=0-5%3
set -euo pipefail
source /users/zliu328/agent-memory/experiments/final_component_environment.sh
export TMPDIR="${TMPDIR:-/tmp}"
tasks=(SH-Doc_QA MH-Doc_QA FactConsolidation-SH FactConsolidation-MH LoCoMo 2WikiMultiHopQA)
"$python_bin" -m unittest experiments.test_component_artifacts -v
"$python_bin" -u -m experiments.compact_component_run prepare --output "$1" --task "${tasks[$SLURM_ARRAY_TASK_ID]}"
