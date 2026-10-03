#!/bin/bash
#SBATCH --job-name=anchormem_build
#SBATCH --partition=gpu-he
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_b200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/oscar/home/zliu328/anchormem_build_%A_%a.out

set -euo pipefail
: "${EXPERIMENT_ID:?Set a new experiment ID}"
: "${SLURM_ARRAY_TASK_ID:?Array 0-9 covers LoCoMo, 10-14 covers the five document tasks}"
PROJECT_DIR="${PROJECT_DIR:-/oscar/home/zliu328/agent-memory}"
ENV_ROOT="/oscar/scratch/zliu328/agent-memory-envs"
export HF_HOME=/oscar/scratch/zliu328/hf_output
export NLTK_DATA="${ENV_ROOT}/nltk_data"
export PYTHONPATH="${PROJECT_DIR}"
export PYTHONHASHSEED=42 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export VLLM_API_KEY="${VLLM_API_KEY:-local-qwen-generator}"
export OPENAI_API_KEY="${VLLM_API_KEY}"
export VLLM_CACHE_ROOT="${TMPDIR:-/tmp}/vllm-${SLURM_JOB_ID}"
mkdir -p "${VLLM_CACHE_ROOT}"
module load cuda/12.9.0-cinr
cd "${PROJECT_DIR}"
GENERATOR_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
GENERATOR_PORT="$((10000 + SLURM_JOB_ID % 50000))"
GENERATOR_BASE_URL="http://127.0.0.1:${GENERATOR_PORT}/v1"
DATA_ARGS=()
RECOVERY_ARGS=()
if (( SLURM_ARRAY_TASK_ID < 10 )); then
    TASK=LoCoMo
    DATA_ARGS=(--input-protocol official-locomo-all-turns
        --path /oscar/scratch/zliu328/agent-memory-data/locomo/locomo10.json
        --group-index "${SLURM_ARRAY_TASK_ID}")
    BUILD_DIR="build_${SLURM_ARRAY_TASK_ID}"
    if [[ -n "${RECOVER_EXPERIMENT_ID:-}" ]]; then
        RECOVERY_ARGS=(--recover-from "/oscar/scratch/zliu328/agent-memory-outputs/${RECOVER_EXPERIMENT_ID}/anchormem/LoCoMo/${BUILD_DIR}")
    fi
else
    DOCUMENT_TASKS=('SH-Doc QA' 'MH-Doc QA' 'FactConsolidation-SH' 'FactConsolidation-MH' '2WikiMultiHopQA')
    TASK="${DOCUMENT_TASKS[$((SLURM_ARRAY_TASK_ID - 10))]}"
    DATA_ARGS=(--input-protocol benchmark-documents
        --data-root "${PROJECT_DIR}/baseline_algorithms/HippoRAG/reproduce/dataset")
    BUILD_DIR="${BUILD_DIR:-build_0}"
fi
if (( ${#RECOVERY_ARGS[@]} == 0 )); then
VLLM_BIN="${VLLM_BIN:-${ENV_ROOT}/qwen_generator/bin/vllm}"
if [[ ! -x "${VLLM_BIN}" ]]; then
    VLLM_BIN=/oscar/scratch/zliu328/llm_tool_ckpt/venvs/vllm/bin/vllm
fi
[[ -x "${VLLM_BIN}" ]] || { echo "Missing existing vLLM executable" >&2; exit 2; }
export PATH="$(dirname -- "${VLLM_BIN}"):${PATH}"
"${VLLM_BIN}" serve "${GENERATOR_MODEL}" \
    --host 127.0.0.1 --port "${GENERATOR_PORT}" --dtype bfloat16 \
    --tensor-parallel-size 1 --max-model-len 32768 \
    --gpu-memory-utilization "${GENERATOR_GPU_MEMORY_UTILIZATION:-0.85}" --seed 42 --generation-config vllm \
    --api-key "${VLLM_API_KEY}" &
GENERATOR_SERVER_PID=$!
cleanup_generator() {
    kill "${GENERATOR_SERVER_PID}" 2>/dev/null || true
    wait "${GENERATOR_SERVER_PID}" 2>/dev/null || true
}
trap cleanup_generator EXIT
for attempt in $(seq 1 720); do
    if curl -fsS "http://127.0.0.1:${GENERATOR_PORT}/health" >/dev/null 2>&1; then
        break
    fi
    if ! kill -0 "${GENERATOR_SERVER_PID}" 2>/dev/null; then
        wait "${GENERATOR_SERVER_PID}"
        exit 1
    fi
    if (( attempt == 720 )); then
        echo "Local generator did not become ready" >&2
        exit 1
    fi
    sleep 5
done
fi
"${ENV_ROOT}/lightmem/bin/python" -u -m experiments.build_reference_memories \
    --baseline anchormem --task "${TASK}" "${DATA_ARGS[@]}" "${RECOVERY_ARGS[@]}" \
    --output-dir "/oscar/scratch/zliu328/agent-memory-outputs/${EXPERIMENT_ID}/anchormem/${TASK// /_}/${BUILD_DIR}" \
    --generator-model "${GENERATOR_MODEL}" --generator-base-url "${GENERATOR_BASE_URL}" \
    --seed 42
