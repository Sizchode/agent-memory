#!/bin/bash
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=amor-component-prepare
set -euo pipefail
source /users/zliu328/agent-memory/experiments/final_component_environment.sh
run_root=$1
task=$2
mkdir -p "$run_root/jobs/$SLURM_JOB_ID"
scontrol show job "$SLURM_JOB_ID" > "$run_root/jobs/$SLURM_JOB_ID/slurm.txt"
cp experiments/final_component_ablation.py experiments/final_component_prepare.sbatch experiments/final_component_environment.sh "$run_root/jobs/$SLURM_JOB_ID/"
"$python_bin" -u -m experiments.final_component_ablation prepare --output "$run_root" --task "$task"
