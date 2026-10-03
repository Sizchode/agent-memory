#!/bin/bash
#SBATCH --job-name=anchormem_qa
#SBATCH --partition=gpu-he
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_b200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/oscar/home/zliu328/anchormem_qa_%A_%a.out

set -euo pipefail
: "${EXPERIMENT_ID:?Set a new output experiment}"
: "${MEMORY_EXPERIMENT_ID:?Set the completed memory experiment}"
PROJECT_DIR="${PROJECT_DIR:-/oscar/home/zliu328/agent-memory}"
ENV_ROOT=/oscar/scratch/zliu328/agent-memory-envs
export HF_HOME=/oscar/scratch/zliu328/hf_output
export NLTK_DATA="${ENV_ROOT}/nltk_data"
export PYTHONPATH="${PROJECT_DIR}"
export PYTHONHASHSEED=42 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
TASKS=('SH-Doc QA' 'MH-Doc QA' 'FactConsolidation-SH' 'FactConsolidation-MH' LoCoMo 2WikiMultiHopQA)
TASK="${TASKS[SLURM_ARRAY_TASK_ID]}"
cd "${PROJECT_DIR}"
ARGS=(--task "${TASK}" --seed 42
    --memory-root "/oscar/scratch/zliu328/agent-memory-outputs/${MEMORY_EXPERIMENT_ID}"
    --output-root "/oscar/scratch/zliu328/agent-memory-outputs/${EXPERIMENT_ID}")
"${ENV_ROOT}/lightmem/bin/python" -u -m experiments.run_anchormem --phase retrieve "${ARGS[@]}"
for model in Qwen/Qwen3.5-9B Qwen/Qwen3.5-4B Qwen/Qwen3.5-2B; do
    "${ENV_ROOT}/runner/bin/python" -u -m experiments.run_anchormem --phase evaluate \
        --evaluation-backbone "${model}" "${ARGS[@]}"
done
