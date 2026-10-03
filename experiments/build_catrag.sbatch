#!/bin/bash
#SBATCH --job-name=catrag_build_retrieve
#SBATCH --partition=gpu-he
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/oscar/home/zliu328/catrag_%j.out
set -euo pipefail
: "${EXPERIMENT_ID:?Set a new experiment ID}"
: "${TASK:?Set a full benchmark task}"
cd /oscar/home/zliu328/agent-memory
export HF_HOME=/oscar/scratch/zliu328/hf_output
export NLTK_DATA=/oscar/scratch/zliu328/agent-memory-envs/nltk_data
export PYTHONHASHSEED=42 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENAI_API_KEY=local-catrag-generator
module load cuda/12.9.0-cinr
VLLM_BIN=/oscar/scratch/zliu328/llm_tool_ckpt/venvs/vllm/bin/vllm
[[ -x "$VLLM_BIN" ]]
export PATH="$(dirname "$VLLM_BIN"):$PATH"
GENERATOR_PORT=$((10000 + SLURM_JOB_ID % 50000))
GENERATOR_CONTEXT=32768
if [[ "${MEMORY_METHOD:-catrag}" == hypermem ]]; then
    GENERATOR_CONTEXT=262144
fi
"$VLLM_BIN" serve Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --host 127.0.0.1 --port "$GENERATOR_PORT" --dtype bfloat16 \
    --tensor-parallel-size 1 --max-model-len "$GENERATOR_CONTEXT" --gpu-memory-utilization 0.85 \
    --seed 42 --generation-config vllm --api-key "$OPENAI_API_KEY" &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT
for attempt in $(seq 1 360); do
    if curl -fsS "http://127.0.0.1:$GENERATOR_PORT/health" >/dev/null 2>&1; then
        break
    fi
    kill -0 "$SERVER_PID" || { wait "$SERVER_PID"; exit 1; }
    if (( attempt == 360 )); then
        echo 'Generator startup failed to reach health endpoint' >&2
        exit 1
    fi
    sleep 5
done
if [[ "${MEMORY_METHOD:-catrag}" == hypermem ]]; then
    REUSE_ARGS=()
    if [[ "${HYPERMEM_STREAMING:-0}" == 1 ]]; then
        REUSE_ARGS+=(--streaming)
        REUSE_ARGS+=(--fact-concurrency "${HYPERMEM_FACT_CONCURRENCY:-1}")
    fi
    if [[ -n "${REUSE_EPISODES_ROOT:-}" ]]; then
        REUSE_ARGS+=(--reuse-episodes-root "$REUSE_EPISODES_ROOT")
    fi
    /oscar/scratch/zliu328/agent-memory-envs/lightmem/bin/python -u -m experiments.build_hypermem \
        --task "$TASK" \
        "${REUSE_ARGS[@]}" \
        --output-root "/oscar/scratch/zliu328/agent-memory-outputs/$EXPERIMENT_ID" \
        --generator-base-url "http://127.0.0.1:$GENERATOR_PORT/v1"
else
    /oscar/scratch/zliu328/agent-memory-envs/hipporag/bin/python -u -m experiments.run_catrag \
    --phase build-retrieve --task "$TASK" \
    --output-root "/oscar/scratch/zliu328/agent-memory-outputs/$EXPERIMENT_ID" \
    --generator-base-url "http://127.0.0.1:$GENERATOR_PORT/v1"
fi
