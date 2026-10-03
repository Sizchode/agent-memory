#!/bin/bash
#SBATCH --job-name=catrag_qa
#SBATCH --partition=gpu-he
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/oscar/home/zliu328/catrag_qa_%A_%a.out
set -euo pipefail
: "${EXPERIMENT_ID:?Set the completed retrieval experiment}"
: "${TASK:?Set a full benchmark task}"
: "${SLURM_ARRAY_TASK_ID:?Array 0-2 selects three QA backbones}"
cd /oscar/home/zliu328/agent-memory
export HF_HOME=/oscar/scratch/zliu328/hf_output
export NLTK_DATA=/oscar/scratch/zliu328/agent-memory-envs/nltk_data
export PYTHONHASHSEED=42 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
MODELS=(Qwen/Qwen3.5-9B Qwen/Qwen3.5-4B Qwen/Qwen3.5-2B)
/oscar/scratch/zliu328/agent-memory-envs/runner/bin/python -u -m experiments.run_catrag \
    --phase evaluate --task "$TASK" --evaluation-backbone "${MODELS[$SLURM_ARRAY_TASK_ID]}" \
    --output-root "/oscar/scratch/zliu328/agent-memory-outputs/$EXPERIMENT_ID"
