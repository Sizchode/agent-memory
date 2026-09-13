#!/bin/bash
#SBATCH --job-name=graph_consolidation
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/home/zliu328/graph_consolidation_%A_%a.out
set -euo pipefail
TASK="${1:-${TASK:-}}"
EXPERIMENT_ID="${2:-${EXPERIMENT_ID:-}}"
SEED_EXPERIMENT_ID="${3:-${SEED_EXPERIMENT_ID:-}}"
SCHEMA_EXPERIMENT_ID="${4:-${SCHEMA_EXPERIMENT_ID:-}}"
export HOME=/oscar/home/zliu328 PATH=/usr/local/bin:/usr/bin:/bin
: "${TASK:?Set the complete task}"
: "${SEED_EXPERIMENT_ID:?Set the verified original replay experiment}"
: "${EXPERIMENT_ID:?Set the output experiment}"
cd /oscar/home/zliu328/agent-memory
export HF_HOME=/oscar/scratch/zliu328/hf_output HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NLTK_DATA=/oscar/scratch/zliu328/agent-memory-envs/nltk_data
export PYTHONHASHSEED=42 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
ARGS=()
if [[ "${5:-0}" == 1 ]]; then
    ARGS+=(--consolidate-multiple)
fi
if [[ "${6:-0}" == 1 ]]; then
    ARGS+=(--weight-grid)
fi
if [[ "${7:-0}" == 1 ]]; then
    ARGS+=(--provenance-weights)
fi
if [[ "${8:-0}" == 1 ]]; then
    ARGS+=(--adaptive-synonyms)
fi
if [[ -n "${SCHEMA_EXPERIMENT_ID:-}" ]]; then
    ARGS+=(--schema-root "/oscar/scratch/zliu328/agent-memory-outputs/$SCHEMA_EXPERIMENT_ID")
fi
exec /oscar/scratch/zliu328/agent-memory-envs/hipporag/bin/python -u -m optimization.replay_graph \
    --task "$TASK" --seed-root "/oscar/scratch/zliu328/agent-memory-outputs/$SEED_EXPERIMENT_ID" \
    --output-root "/oscar/scratch/zliu328/agent-memory-outputs/$EXPERIMENT_ID" "${ARGS[@]}"
