#!/bin/bash
#SBATCH --partition=gpu-he
#SBATCH --account=default
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --job-name=amor-component-qa
set -euo pipefail
source /users/zliu328/agent-memory/experiments/final_component_environment.sh
run_root=$1
model=$2
mkdir -p "$run_root/jobs/$SLURM_JOB_ID"
scontrol show job "$SLURM_JOB_ID" > "$run_root/jobs/$SLURM_JOB_ID/slurm.txt"
nvidia-smi > "$run_root/jobs/$SLURM_JOB_ID/gpu.txt"
cp experiments/final_component_ablation.py experiments/final_component_qa.sbatch experiments/final_component_environment.sh "$run_root/jobs/$SLURM_JOB_ID/"
"$python_bin" -u -m experiments.final_component_ablation qa --output "$run_root" --model "$model" --smoke
"$python_bin" -u -m experiments.final_component_ablation qa --output "$run_root" --model "$model"
