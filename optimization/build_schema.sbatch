#!/bin/bash
#SBATCH --job-name=graph_schema
#SBATCH --partition=gpu-he
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/oscar/home/zliu328/graph_schema_%j.out
set -euo pipefail
: "${EXPERIMENT_ID:?Set a fresh schema output directory}"
if [[ -n "${UPDATE_POLICY_SCHEMA_EXPERIMENT:-}" ]]; then
    echo "The experimental update-policy schema builder has been retired" >&2
    exit 2
fi
cd /oscar/home/zliu328/agent-memory
module load cuda/12.9.0-cinr
export HF_HOME=/oscar/scratch/zliu328/hf_output HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONHASHSEED=42 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export VLLM_API_KEY=local-qwen-generator
VLLM=/oscar/scratch/zliu328/agent-memory-envs/vllm_cu129/bin/vllm
export PATH="$(dirname "$VLLM"):$PATH"
PORT=$((10000 + SLURM_JOB_ID % 50000))
"$VLLM" serve Qwen/Qwen3-30B-A3B-Instruct-2507 --host 127.0.0.1 --port "$PORT" \
    --dtype bfloat16 --tensor-parallel-size "${SCHEMA_TENSOR_PARALLEL_SIZE:-1}" --max-model-len 32768 \
    --gpu-memory-utilization 0.85 --seed 42 --generation-config vllm --api-key "$VLLM_API_KEY" &
SERVER=$!
cleanup() {
    kill "$SERVER" 2>/dev/null || true
    wait "$SERVER" 2>/dev/null || true
}
trap cleanup EXIT
for ((i=0; i<360; i++)); do
    if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        break
    fi
    if ! kill -0 "$SERVER" 2>/dev/null; then
        wait "$SERVER"
        exit 1
    fi
    if (( i == 359 )); then
        echo "Generator startup timed out" >&2
        exit 1
    fi
    sleep 5
done
ARGS=()
if [[ "${RESUME_SCHEMA:-0}" == 1 ]]; then
    ARGS+=(--resume)
fi
MODULE=optimization.graph_construction.relation_schema
if [[ -n "${SCHEMA_SOURCE_EXPERIMENT:-}" ]]; then
    MODULE=optimization.graph_construction.canonicalize_schema
    ARGS+=(--source-root "/oscar/scratch/zliu328/agent-memory-outputs/$SCHEMA_SOURCE_EXPERIMENT")
fi
/oscar/scratch/zliu328/agent-memory-envs/runner/bin/python -u -m "$MODULE" \
    --base-url "http://127.0.0.1:$PORT/v1" \
    --output-root "/oscar/scratch/zliu328/agent-memory-outputs/$EXPERIMENT_ID" "${ARGS[@]}"
