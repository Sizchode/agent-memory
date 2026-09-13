#!/bin/bash
#SBATCH --job-name=graph_optimization
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/oscar/home/zliu328/graph_optimization_%A_%a.out
set -euo pipefail
: "${EXPERIMENT_ID:?Set the output experiment}"
: "${TASK:?Set the complete task}"
PHASE="${PHASE:-retrieve}"
cd /oscar/home/zliu328/agent-memory
export HF_HOME=/oscar/scratch/zliu328/hf_output HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NLTK_DATA=/oscar/scratch/zliu328/agent-memory-envs/nltk_data
export PYTHONHASHSEED=42 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
ENV_ROOT=/oscar/scratch/zliu328/agent-memory-envs
ARGS=(--phase "$PHASE" --task "$TASK" --output-root "/oscar/scratch/zliu328/agent-memory-outputs/$EXPERIMENT_ID")
if [[ "$PHASE" == evaluate ]]; then
    MODELS=(Qwen/Qwen3.5-9B Qwen/Qwen3.5-4B Qwen/Qwen3.5-2B)
    ARGS+=(--evaluation-backbone "${MODELS[${SLURM_ARRAY_TASK_ID:-0}]}")
    PYTHON="$ENV_ROOT/runner/bin/python"
else
    PYTHON="$ENV_ROOT/hipporag/bin/python"
fi
if [[ -n "${VARIANTS:-}" ]]; then
    read -r -a SELECTED <<< "$VARIANTS"
    ARGS+=(--variants "${SELECTED[@]}")
fi
nvidia-smi
exec "$PYTHON" -u -m optimization.run_graph "${ARGS[@]}"
