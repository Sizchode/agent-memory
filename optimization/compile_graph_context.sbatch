#!/bin/bash
#SBATCH --job-name=compile_graph_context
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/oscar/home/zliu328/compile_graph_context_%j.out
set -euo pipefail
TASK="${1:?Set the full task}"
EXPERIMENT_ID="${2:?Set the output experiment}"
export HOME=/oscar/home/zliu328 PATH=/usr/local/bin:/usr/bin:/bin
export HF_HOME=/oscar/scratch/zliu328/hf_output HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NLTK_DATA=/oscar/scratch/zliu328/agent-memory-envs/nltk_data
export PYTHONHASHSEED=42 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /oscar/home/zliu328/agent-memory
ARGS=()
if [[ "${3:-0}" == 1 ]]; then
    ARGS+=(--with-source)
fi
if [[ "${4:-0}" != 0 ]]; then
    ARGS+=(--source-window "$4")
fi
exec /oscar/scratch/zliu328/agent-memory-envs/hipporag/bin/python -u -m optimization.compile_graph_context \
    --task "$TASK" --output-root "/oscar/scratch/zliu328/agent-memory-outputs/$EXPERIMENT_ID" "${ARGS[@]}"
