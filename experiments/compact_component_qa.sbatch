#!/bin/bash
#SBATCH --partition=gpu-he
#SBATCH --account=default
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --job-name=amor-compact-qa
#SBATCH --array=0-3%4
set -euo pipefail
source /users/zliu328/agent-memory/experiments/final_component_environment.sh
export TMPDIR="${TMPDIR:-/tmp}"
models=(Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B google/gemma-3-4b-it meta-llama/Llama-3.1-8B-Instruct)
nvidia-smi
"$python_bin" -u -m experiments.compact_component_run qa --output "$1" --model "${models[$SLURM_ARRAY_TASK_ID]}"
