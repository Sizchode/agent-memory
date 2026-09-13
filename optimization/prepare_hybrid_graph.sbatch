#!/bin/bash
#SBATCH --job-name=prepare_hybrid_graph
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/home/zliu328/prepare_hybrid_graph_%j.out
set -euo pipefail
TASK="${1:?Set the full task}"
EXPERIMENT_ID="${2:?Set the output experiment}"
export HOME=/oscar/home/zliu328 PATH=/usr/local/bin:/usr/bin:/bin
export HF_HOME=/oscar/scratch/zliu328/hf_output HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NLTK_DATA=/oscar/scratch/zliu328/agent-memory-envs/nltk_data
export PYTHONHASHSEED=42 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /oscar/home/zliu328/agent-memory
exec /oscar/scratch/zliu328/agent-memory-envs/hipporag/bin/python -u -m optimization.prepare_hybrid_graph \
    --task "$TASK" --output-root "/oscar/scratch/zliu328/agent-memory-outputs/$EXPERIMENT_ID"
