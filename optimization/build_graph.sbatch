#!/bin/bash
#SBATCH --job-name=build_graph
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/oscar/home/zliu328/build_graph_%A_%a.out
set -euo pipefail
: "${EXPERIMENT_ID:?Set a fresh graph output directory}"
cd /oscar/home/zliu328/agent-memory
export HF_HOME=/oscar/scratch/zliu328/hf_output HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NLTK_DATA=/oscar/scratch/zliu328/agent-memory-envs/nltk_data
export PYTHONHASHSEED=42 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
TASKS=("SH-Doc QA" "MH-Doc QA" FactConsolidation-SH FactConsolidation-MH LoCoMo 2WikiMultiHopQA)
exec /oscar/scratch/zliu328/agent-memory-envs/hipporag/bin/python -u -m optimization.build_graph \
    --task "${TASKS[${SLURM_ARRAY_TASK_ID:?Set task array 0-5}]}" \
    --output-root "/oscar/scratch/zliu328/agent-memory-outputs/$EXPERIMENT_ID"
