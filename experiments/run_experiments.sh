#!/bin/bash
#SBATCH --job-name=agent_memory
#SBATCH --partition=gpu-he
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=25:59:59
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=zhenkeliu@163.com
#SBATCH --output=/oscar/home/zliu328/agent_memory_%A_%a.out
#SBATCH --export=ALL,PYTORCH_JIT=0

set -euo pipefail

# A unified cross-domain benchmark evaluates every baseline on every task.
# LightMem and Mem0 receive the documented document-fact extraction adapter
# when the source is not a timestamped dialogue; their native prompts remain
# in use for LoCoMo.
BASELINES=(bm25 dense lightmem hipporag2 mem0)
GENERATED_MEMORY_BASELINES=(lightmem hipporag2 mem0)
# Evaluation backbones consume the same frozen retrieval artifacts. They do
# not rebuild memories and are never used as judges.
EVALUATION_BACKBONES=(
  "Qwen/Qwen3.5-9B"
  "Qwen/Qwen3.5-4B"
  "Qwen/Qwen3.5-2B"
)
# Llama 3.1 is gated.  Public backbones should not be held back when the
# allocation has no accepted Hugging Face token for it.
if [[ "${INCLUDE_GATED_LLAMA:-0}" == "1" ]]; then
  EVALUATION_BACKBONES+=("meta-llama/Llama-3.1-8B-Instruct")
fi
TASKS=(
  "SH-Doc QA"
  "MH-Doc QA"
  "FactConsolidation-SH"
  "FactConsolidation-MH"
  "LoCoMo"
  "2WikiMultiHopQA"
)
MAX_CONCURRENT_JOBS=6
GPU_HE_B200_JOBS=4
GPU_HE_HIPPORAG_JOBS=2

if [[ -n "${PROJECT_DIR:-}" ]]; then
  PROJECT_DIR="$(cd -- "${PROJECT_DIR}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/main.py" ]]; then
  PROJECT_DIR="$(cd -- "${SLURM_SUBMIT_DIR}" && pwd)"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
fi
SCRIPT_DIR="${PROJECT_DIR}/experiments"
RETRIEVAL_COMBINATIONS=()
for baseline in "${BASELINES[@]}"; do
  for task in "${TASKS[@]}"; do
    RETRIEVAL_COMBINATIONS+=("${baseline}"$'\t'"${task}")
  done
done

EXISTING_MEMORY_COMBINATIONS=()
for baseline in "${GENERATED_MEMORY_BASELINES[@]}"; do
  for task in "${TASKS[@]}"; do
    EXISTING_MEMORY_COMBINATIONS+=("${baseline}"$'\t'"${task}")
  done
done

EVALUATION_COMBINATIONS=()
SHORT_EVALUATION_INDICES=()
LONG_EVALUATION_INDICES=()
for combination in "${RETRIEVAL_COMBINATIONS[@]}"; do
  IFS=$'\t' read -r baseline task <<< "${combination}"
  for evaluator in "${EVALUATION_BACKBONES[@]}"; do
    EVALUATION_COMBINATIONS+=("${baseline}"$'\t'"${evaluator}"$'\t'"${task}")
    evaluation_index="$(( ${#EVALUATION_COMBINATIONS[@]} - 1 ))"
    if [[ "${task}" == "LoCoMo" || "${task}" == "2WikiMultiHopQA" ]]; then
      LONG_EVALUATION_INDICES+=("${evaluation_index}")
    else
      SHORT_EVALUATION_INDICES+=("${evaluation_index}")
    fi
  done
done

if [[ "${1:-}" == "--retrieve-existing" ]]; then
  : "${MEMORY_INPUT_EXPERIMENT_ID:?Set MEMORY_INPUT_EXPERIMENT_ID to the completed generation experiment.}"
  EXPERIMENT_ID="${EXPERIMENT_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
  existing_end="$(( ${#EXISTING_MEMORY_COMBINATIONS[@]} - 1 ))"
  retrieval_job_id="$(sbatch --parsable --array="0-${existing_end}%${MAX_CONCURRENT_JOBS}" --export="ALL,PROJECT_DIR=${PROJECT_DIR},EXPERIMENT_ID=${EXPERIMENT_ID},MEMORY_INPUT_EXPERIMENT_ID=${MEMORY_INPUT_EXPERIMENT_ID}" "${SCRIPT_DIR}/run_experiments.sh" --retrieve-existing-worker)"
  echo "experiment=${EXPERIMENT_ID} memory_input_experiment=${MEMORY_INPUT_EXPERIMENT_ID} retrieval_job=${retrieval_job_id}"
  exit 0
fi

if [[ "${1:-}" != "--retrieve-worker" && "${1:-}" != "--retrieve-existing-worker" && "${1:-}" != "--evaluate-worker" ]]; then
  if (( ${#RETRIEVAL_COMBINATIONS[@]} == 0 || ${#EVALUATION_COMBINATIONS[@]} == 0 )); then
    echo "The experiment grid is empty." >&2
    exit 2
  fi
  EXPERIMENT_ID="${EXPERIMENT_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
  SIMPLE_RETRIEVAL_END="$(( 2 * ${#TASKS[@]} - 1 ))"
  GENERATED_RETRIEVAL_START="$(( SIMPLE_RETRIEVAL_END + 1 ))"
  LIGHTMEM_RETRIEVAL_END="$(( GENERATED_RETRIEVAL_START + ${#TASKS[@]} - 1 ))"
  HIPPORAG_RETRIEVAL_START="$(( LIGHTMEM_RETRIEVAL_END + 1 ))"
  HIPPORAG_RETRIEVAL_END="$(( HIPPORAG_RETRIEVAL_START + ${#TASKS[@]} - 1 ))"
  MEM0_RETRIEVAL_START="$(( HIPPORAG_RETRIEVAL_END + 1 ))"
  RETRIEVAL_END="$(( ${#RETRIEVAL_COMBINATIONS[@]} - 1 ))"
  SHORT_EVALUATION_SPEC="$(IFS=,; echo "${SHORT_EVALUATION_INDICES[*]}")"
  LONG_EVALUATION_SPEC="$(IFS=,; echo "${LONG_EVALUATION_INDICES[*]}")"
  simple_job_id="$(sbatch --parsable --job-name=am_ret_simple --partition=gpu --gres=gpu:l40s:1 --array="0-${SIMPLE_RETRIEVAL_END}%${MAX_CONCURRENT_JOBS}" --time=02:00:00 --cpus-per-task=2 --mem=32G --export="ALL,PROJECT_DIR=${PROJECT_DIR},EXPERIMENT_ID=${EXPERIMENT_ID},SEED=42,RETRIEVAL_TOP_K=5" "${SCRIPT_DIR}/run_experiments.sh" --retrieve-worker)"
  generated_b200_job_id="$(sbatch --parsable --job-name=am_ret_memory --partition=gpu-he --gres=gpu:nvidia_b200:1 --array="${GENERATED_RETRIEVAL_START}-${LIGHTMEM_RETRIEVAL_END},${MEM0_RETRIEVAL_START}-${RETRIEVAL_END}%${GPU_HE_B200_JOBS}" --time=12:00:00 --cpus-per-task=4 --mem=64G --export="ALL,PROJECT_DIR=${PROJECT_DIR},EXPERIMENT_ID=${EXPERIMENT_ID},SEED=42,RETRIEVAL_TOP_K=5,GENERATOR_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507,GENERATOR_MAX_MODEL_LEN=32768,LIGHTMEM_GENERATOR_MAX_TOKENS=16000,MEM0_GENERATOR_MAX_TOKENS=2000" "${SCRIPT_DIR}/run_experiments.sh" --retrieve-worker)"
  # HippoRAG pins Torch 2.5.1, whose CUDA build supports H100 (sm_90) but not
  # B200 (sm_100). Keep the official software environment and run it on H100.
  hipporag_job_id="$(sbatch --parsable --job-name=am_ret_hippo --partition=gpu-he --gres=gpu:h100:1 --array="${HIPPORAG_RETRIEVAL_START}-${HIPPORAG_RETRIEVAL_END}%${GPU_HE_HIPPORAG_JOBS}" --time=12:00:00 --cpus-per-task=2 --mem=32G --export="ALL,PROJECT_DIR=${PROJECT_DIR},EXPERIMENT_ID=${EXPERIMENT_ID},SEED=42,RETRIEVAL_TOP_K=5,GENERATOR_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507,GENERATOR_MAX_MODEL_LEN=32768,GENERATOR_GPU_MEMORY_UTILIZATION=0.82" "${SCRIPT_DIR}/run_experiments.sh" --retrieve-worker)"
  retrieval_dependency="afterok:${simple_job_id}:${generated_b200_job_id}:${hipporag_job_id}"
  short_evaluation_job_id="$(sbatch --parsable --job-name=am_qa_short --partition=gpu --gres=gpu:l40s:1 --dependency="${retrieval_dependency}" --array="${SHORT_EVALUATION_SPEC}%${MAX_CONCURRENT_JOBS}" --time=02:00:00 --cpus-per-task=2 --mem=32G --export="ALL,PROJECT_DIR=${PROJECT_DIR},EXPERIMENT_ID=${EXPERIMENT_ID},RETRIEVAL_EXPERIMENT_ID=${EXPERIMENT_ID},SEED=42" "${SCRIPT_DIR}/run_experiments.sh" --evaluate-worker)"
  long_evaluation_job_id="$(sbatch --parsable --job-name=am_qa_long --partition=gpu --gres=gpu:l40s:1 --dependency="${retrieval_dependency}" --array="${LONG_EVALUATION_SPEC}%${MAX_CONCURRENT_JOBS}" --time=06:00:00 --cpus-per-task=2 --mem=32G --export="ALL,PROJECT_DIR=${PROJECT_DIR},EXPERIMENT_ID=${EXPERIMENT_ID},RETRIEVAL_EXPERIMENT_ID=${EXPERIMENT_ID},SEED=42" "${SCRIPT_DIR}/run_experiments.sh" --evaluate-worker)"
  echo "experiment=${EXPERIMENT_ID} simple_retrieval_job=${simple_job_id} generated_b200_job=${generated_b200_job_id} hipporag_job=${hipporag_job_id} short_evaluation_job=${short_evaluation_job_id} long_evaluation_job=${long_evaluation_job_id}"
  exit 0
fi

: "${SLURM_ARRAY_TASK_ID:?This script must be launched through its launcher mode.}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/oscar/scratch/zliu328}"
ENV_ROOT="${ENV_ROOT:-${SCRATCH_ROOT}/agent-memory-envs}"
HF_HOME="${HF_HOME:-${SCRATCH_ROOT}/hf_output}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH_ROOT}/agent-memory-outputs}"
OFFICIAL_CONFIG_DIR="${OFFICIAL_CONFIG_DIR:-${PROJECT_DIR}/experiments/configs}"
EXPERIMENT_ID="${EXPERIMENT_ID:?The launcher must export EXPERIMENT_ID.}"
RETRIEVAL_EXPERIMENT_ID="${RETRIEVAL_EXPERIMENT_ID:-${EXPERIMENT_ID}}"

export HF_HOME TOKENIZERS_PARALLELISM=false PYTHONHASHSEED="${SEED:-42}" NLTK_DATA="${NLTK_DATA:-${ENV_ROOT}/nltk_data}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

RUNNER_BIN="${ENV_ROOT}/runner/bin/python"
if [[ ! -x "${RUNNER_BIN}" ]]; then
  echo "Missing runner environment: ${RUNNER_BIN}" >&2
  exit 2
fi

if [[ "$1" == "--evaluate-worker" ]]; then
  if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#EVALUATION_COMBINATIONS[@]} )); then
    echo "Evaluation array index ${SLURM_ARRAY_TASK_ID} is outside the grid." >&2
    exit 2
  fi
  IFS=$'\t' read -r BASELINE EVALUATION_BACKBONE TASK <<< "${EVALUATION_COMBINATIONS[$SLURM_ARRAY_TASK_ID]}"
  RETRIEVAL_PATH="${OUTPUT_ROOT}/${RETRIEVAL_EXPERIMENT_ID}/${BASELINE}/${TASK// /_}/retrieval.jsonl"
  METHOD_DIR="${OUTPUT_ROOT}/${EXPERIMENT_ID}/${BASELINE}/${TASK// /_}"
  MODEL_TAG="${EVALUATION_BACKBONE//\//_}"
  EVALUATION_DIR="${METHOD_DIR}/evaluations/${MODEL_TAG}"
  [[ -s "${RETRIEVAL_PATH}" ]] || { echo "Missing retrieval artifact: ${RETRIEVAL_PATH}" >&2; exit 2; }
  nvidia-smi
  exec "${RUNNER_BIN}" "${PROJECT_DIR}/main.py" \
    --phase evaluate \
    --task "${TASK}" \
    --baseline "${BASELINE}" \
    --output-dir "${EVALUATION_DIR}" \
    --retrieval-input "${RETRIEVAL_PATH}" \
    --evaluation-backbone "${EVALUATION_BACKBONE}" \
    --evaluation-dtype "${EVALUATION_DTYPE:-bfloat16}" \
    --evaluation-device-map "${EVALUATION_DEVICE_MAP:-cuda}" \
    --seed "${SEED:-42}"
fi

RETRIEVAL_PHASE="retrieve"
MEMORY_INPUT_ARGS=()
if [[ "$1" == "--retrieve-existing-worker" ]]; then
  : "${MEMORY_INPUT_EXPERIMENT_ID:?The launcher must export MEMORY_INPUT_EXPERIMENT_ID.}"
  ACTIVE_RETRIEVAL_COMBINATIONS=("${EXISTING_MEMORY_COMBINATIONS[@]}")
  RETRIEVAL_PHASE="retrieve-existing"
else
  ACTIVE_RETRIEVAL_COMBINATIONS=("${RETRIEVAL_COMBINATIONS[@]}")
fi

if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#ACTIVE_RETRIEVAL_COMBINATIONS[@]} )); then
  echo "Retrieval array index ${SLURM_ARRAY_TASK_ID} is outside the grid." >&2
  exit 2
fi
IFS=$'\t' read -r BASELINE TASK <<< "${ACTIVE_RETRIEVAL_COMBINATIONS[$SLURM_ARRAY_TASK_ID]}"
RUN_DIR="${OUTPUT_ROOT}/${EXPERIMENT_ID}/${BASELINE}/${TASK// /_}${RUN_DIR_SUFFIX:-}"
if [[ "${RETRIEVAL_PHASE}" == "retrieve-existing" ]]; then
  MEMORY_INPUT_DIR="${OUTPUT_ROOT}/${MEMORY_INPUT_EXPERIMENT_ID}/${BASELINE}/${TASK// /_}${MEMORY_INPUT_RUN_DIR_SUFFIX:-}"
  MEMORY_INPUT_ARGS+=(--memory-input-dir "${MEMORY_INPUT_DIR}")
fi

# Mem0 officially supports MEM0_DIR for its local history and telemetry state.
# Keep that runtime state task-local so concurrent benchmark tasks do not open
# the same Qdrant directory or history database.
if [[ "${BASELINE}" == "mem0" ]]; then
  export MEM0_DIR="${RUN_DIR}/mem0_runtime"
  mkdir -p "${MEM0_DIR}"
fi

case "${BASELINE}" in
  bm25|dense) PYTHON_BIN="${ENV_ROOT}/runner/bin/python" ;;
  lightmem|mem0) PYTHON_BIN="${ENV_ROOT}/${BASELINE}/bin/python" ;;
  hipporag2) PYTHON_BIN="${ENV_ROOT}/hipporag/bin/python" ;;
  *) echo "Unknown baseline: ${BASELINE}" >&2; exit 2 ;;
esac

GENERATOR_MODEL="${GENERATOR_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
GENERATOR_BASE_URL="${GENERATOR_BASE_URL:-}"
GENERATOR_API_KEY_ENV="${GENERATOR_API_KEY_ENV:-VLLM_API_KEY}"
DEFAULT_GENERATOR_MAX_MODEL_LEN=32768
DEFAULT_LIGHTMEM_GENERATOR_MAX_TOKENS=16000
DEFAULT_MEM0_GENERATOR_MAX_TOKENS=2000
GENERATOR_SERVER_PID=""
cleanup_generator_server() {
  if [[ -n "${GENERATOR_SERVER_PID}" ]]; then
    kill "${GENERATOR_SERVER_PID}" 2>/dev/null || true
    wait "${GENERATOR_SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup_generator_server EXIT
NEEDS_GENERATOR=0
if [[ "${RETRIEVAL_PHASE}" == "retrieve" && ( "${BASELINE}" == "lightmem" || "${BASELINE}" == "hipporag2" || "${BASELINE}" == "mem0" ) ]]; then
  NEEDS_GENERATOR=1
elif [[ "${RETRIEVAL_PHASE}" == "retrieve-existing" && "${BASELINE}" == "hipporag2" ]]; then
  NEEDS_GENERATOR=1
fi
if [[ "${RETRIEVAL_PHASE}" == "retrieve-existing" && "${BASELINE}" != "hipporag2" ]]; then
  export REUSE_MEMORY_API_KEY="not-used"
  GENERATOR_API_KEY_ENV="REUSE_MEMORY_API_KEY"
fi
if (( NEEDS_GENERATOR )); then
  module load cuda/12.9.0-cinr
  # Concurrent workers must not share Torch/vLLM's writable compile cache on
  # the home filesystem. The cache is derived runtime state, not an experiment
  # artifact.
  export VLLM_CACHE_ROOT="${TMPDIR:-/tmp}/vllm-${SLURM_JOB_ID}"
  mkdir -p "${VLLM_CACHE_ROOT}"
  VLLM_BIN="${VLLM_BIN:-${ENV_ROOT}/qwen_generator/bin/vllm}"
  if [[ ! -x "${VLLM_BIN}" && -x "/oscar/scratch/zliu328/llm_tool_ckpt/venvs/vllm/bin/vllm" ]]; then
    VLLM_BIN="/oscar/scratch/zliu328/llm_tool_ckpt/venvs/vllm/bin/vllm"
  fi
  [[ -x "${VLLM_BIN}" ]] || { echo "Missing vLLM executable: ${VLLM_BIN}" >&2; exit 2; }
  export PATH="$(dirname -- "${VLLM_BIN}"):${PATH}"
  GENERATOR_PORT="$((10000 + SLURM_JOB_ID % 50000))"
  GENERATOR_BASE_URL="http://127.0.0.1:${GENERATOR_PORT}/v1"
  export VLLM_API_KEY="${VLLM_API_KEY:-local-qwen-generator}"
  export OPENAI_API_KEY="${VLLM_API_KEY}"
  GENERATOR_API_KEY_ENV="VLLM_API_KEY"
  "${VLLM_BIN}" serve "${GENERATOR_MODEL}" \
      --host 127.0.0.1 \
      --port "${GENERATOR_PORT}" \
      --dtype bfloat16 \
      --tensor-parallel-size "${GENERATOR_TENSOR_PARALLEL_SIZE:-1}" \
      --max-model-len "${GENERATOR_MAX_MODEL_LEN:-${DEFAULT_GENERATOR_MAX_MODEL_LEN}}" \
      --gpu-memory-utilization "${GENERATOR_GPU_MEMORY_UTILIZATION:-0.85}" \
      --seed "${SEED:-42}" \
      --generation-config vllm \
      --api-key "${VLLM_API_KEY}" &
  GENERATOR_SERVER_PID="$!"
  GENERATOR_STARTUP_ATTEMPTS="${GENERATOR_STARTUP_ATTEMPTS:-720}"
  for attempt in $(seq 1 "${GENERATOR_STARTUP_ATTEMPTS}"); do
      if curl -fsS "http://127.0.0.1:${GENERATOR_PORT}/health" >/dev/null; then
        break
      fi
      if ! kill -0 "${GENERATOR_SERVER_PID}" 2>/dev/null; then
        wait "${GENERATOR_SERVER_PID}"
        exit 1
      fi
      if (( attempt == GENERATOR_STARTUP_ATTEMPTS )); then
        echo "Qwen generator server did not become ready within $((GENERATOR_STARTUP_ATTEMPTS * 5 / 60)) minutes." >&2
        exit 1
      fi
      sleep 5
  done
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing UV environment: ${PYTHON_BIN}" >&2
  exit 2
fi

DATA_ARGS=()
case "${TASK}" in
  LoCoMo)
    LOCOMO_PATH="${LOCOMO_PATH:-${SCRATCH_ROOT}/agent-memory-data/locomo/locomo10.json}"
    [[ -f "${LOCOMO_PATH}" ]] || { echo "Missing official LoCoMo dataset: ${LOCOMO_PATH}" >&2; exit 2; }
    DATA_ARGS+=(--path "${LOCOMO_PATH}")
    ;;
  2WikiMultiHopQA)
    HIPPORAG_DATA_ROOT="${HIPPORAG_DATA_ROOT:-${PROJECT_DIR}/baseline_algorithms/HippoRAG/reproduce/dataset}"
    DATA_ARGS+=(--data-root "${HIPPORAG_DATA_ROOT}")
    ;;
  *)
    # MemoryAgentBench uses 512-token chunks for SH/MH document QA and both
    # FactConsolidation tasks in its main table.
    DATA_ARGS+=(--chunk-size "${DOCUMENT_CHUNK_SIZE:-512}")
    ;;
esac

# The retained controlled suite uses the same five-item evidence budget for
# every method and task.
RETRIEVAL_TOP_K="${RETRIEVAL_TOP_K:-5}"

OFFICIAL_ARGS=()
case "${BASELINE}" in
  lightmem|hipporag2|mem0)
    CONFIG_PATH="${OFFICIAL_CONFIG_DIR}/${BASELINE}.json"
    if [[ ! -f "${CONFIG_PATH}" ]]; then
      echo "Missing official configuration: ${CONFIG_PATH}" >&2
      exit 2
    fi
    OFFICIAL_ARGS+=(--official-config "${CONFIG_PATH}")
    ;;
esac

LIMIT_ARGS=(--max-queries "${MAX_QUERIES:-1000}")
if [[ -n "${MAX_CONTEXTS:-}" ]]; then
  LIMIT_ARGS+=(--max-contexts "${MAX_CONTEXTS}")
fi
if [[ -n "${GROUP_START:-}" ]]; then
  LIMIT_ARGS+=(--group-start "${GROUP_START}")
fi
if [[ -n "${GROUP_STOP:-}" ]]; then
  LIMIT_ARGS+=(--group-stop "${GROUP_STOP}")
fi

echo "baseline=${BASELINE} task=${TASK} phase=${RETRIEVAL_PHASE}"
echo "python=${PYTHON_BIN} hf_home=${HF_HOME} output=${RUN_DIR}"
if [[ "${BASELINE}" != "bm25" ]]; then
  nvidia-smi
fi

"${PYTHON_BIN}" "${PROJECT_DIR}/main.py" \
  --phase "${RETRIEVAL_PHASE}" \
  --task "${TASK}" \
  --baseline "${BASELINE}" \
  --output-dir "${RUN_DIR}" \
  --generator-model "${GENERATOR_MODEL}" \
  --generator-base-url "${GENERATOR_BASE_URL}" \
  --generator-api-key-env "${GENERATOR_API_KEY_ENV}" \
  --lightmem-generator-max-tokens "${LIGHTMEM_GENERATOR_MAX_TOKENS:-${DEFAULT_LIGHTMEM_GENERATOR_MAX_TOKENS}}" \
  --mem0-generator-max-tokens "${MEM0_GENERATOR_MAX_TOKENS:-${DEFAULT_MEM0_GENERATOR_MAX_TOKENS}}" \
  --top-k "${RETRIEVAL_TOP_K}" \
  --seed "${SEED:-42}" \
  "${MEMORY_INPUT_ARGS[@]}" \
  "${LIMIT_ARGS[@]}" \
  "${DATA_ARGS[@]}" \
  "${OFFICIAL_ARGS[@]}"
