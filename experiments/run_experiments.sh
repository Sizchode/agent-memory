#!/bin/bash
#SBATCH --job-name=agent_memory
#SBATCH --partition=gpu-he
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=60G
#SBATCH --time=25:59:59
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=zhenkeliu@163.com
#SBATCH --output=agent_memory_%A_%a.out
#SBATCH --export=ALL,PYTORCH_JIT=0

set -euo pipefail

# Edit only these lists to define an experiment grid. The launcher below turns
# their Cartesian product into a Slurm array automatically.
BASELINES=(
  bm25
  dense
  lightmem
  hipporag2
  mem0
)
EVALUATION_BACKBONES=(
  "Qwen/Qwen3.5-35B-A3B"
  "Qwen/Qwen3.5-27B"
  "google/gemma-4-12B-it"
  "google/gemma-4-26B-A4B-it"
)
TASKS=(
  "SH-Doc QA"
  "MH-Doc QA"
  "EventQA"
  "FactConsolidation-SH"
  "FactConsolidation-MH"
  "LoCoMo"
  "MuSiQue"
  "2WikiMultiHopQA"
  "HotpotQA"
)
MAX_CONCURRENT_JOBS=4

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
COMBINATIONS=()
for baseline in "${BASELINES[@]}"; do
  for evaluator in "${EVALUATION_BACKBONES[@]}"; do
    for task in "${TASKS[@]}"; do
      COMBINATIONS+=("${baseline}"$'\t'"${evaluator}"$'\t'"${task}")
    done
  done
done

if [[ "${1:-}" != "--worker" ]]; then
  if (( ${#COMBINATIONS[@]} == 0 )); then
    echo "The experiment grid is empty." >&2
    exit 2
  fi
  ARRAY_END="$(( ${#COMBINATIONS[@]} - 1 ))"
  echo "Submitting ${#COMBINATIONS[@]} jobs with at most ${MAX_CONCURRENT_JOBS} concurrent workers."
  exec sbatch --array="0-${ARRAY_END}%${MAX_CONCURRENT_JOBS}" --export=ALL "${SCRIPT_DIR}/run_experiments.sh" --worker
fi

: "${SLURM_ARRAY_TASK_ID:?This script must be launched through its launcher mode.}"
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#COMBINATIONS[@]} )); then
  echo "Array index ${SLURM_ARRAY_TASK_ID} is outside this experiment grid." >&2
  exit 2
fi
IFS=$'\t' read -r BASELINE EVALUATION_BACKBONE TASK <<< "${COMBINATIONS[$SLURM_ARRAY_TASK_ID]}"

SCRATCH_ROOT="${SCRATCH_ROOT:-/oscar/scratch/zliu328}"
ENV_ROOT="${ENV_ROOT:-${SCRATCH_ROOT}/agent-memory-envs}"
HF_HOME="${HF_HOME:-${SCRATCH_ROOT}/hf_output}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH_ROOT}/agent-memory-outputs}"
CONDA_ENV="${CONDA_ENV:-llm_ft}"
OFFICIAL_CONFIG_DIR="${OFFICIAL_CONFIG_DIR:-${PROJECT_DIR}/experiments/configs}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
export HF_HOME TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

case "${BASELINE}" in
  bm25|dense) PYTHON_BIN="${ENV_ROOT}/runner/bin/python" ;;
  lightmem|mem0) PYTHON_BIN="${ENV_ROOT}/${BASELINE}/bin/python" ;;
  hipporag2) PYTHON_BIN="${ENV_ROOT}/hipporag/bin/python" ;;
  *) echo "Unknown baseline: ${BASELINE}" >&2; exit 2 ;;
esac

if [[ "${BASELINE}" == "hipporag2" ]]; then
  : "${HIPPORAG_API_KEY:?Set HIPPORAG_API_KEY for HippoRAG OpenAI-compatible clients.}"
  export OPENAI_API_KEY="${HIPPORAG_API_KEY}"
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing UV environment: ${PYTHON_BIN}" >&2
  exit 2
fi

DATA_ARGS=()
case "${TASK}" in
  LoCoMo)
    : "${LOCOMO_PATH:?Set LOCOMO_PATH before launching LoCoMo.}"
    DATA_ARGS+=(--path "${LOCOMO_PATH}")
    ;;
  MuSiQue|2WikiMultiHopQA|HotpotQA)
    : "${HIPPORAG_DATA_ROOT:?Set HIPPORAG_DATA_ROOT before launching multi-hop QA.}"
    DATA_ARGS+=(--data-root "${HIPPORAG_DATA_ROOT}")
    ;;
esac

OFFICIAL_ARGS=()
EMBEDDING_ARGS=()
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

case "${BASELINE}" in
  dense|lightmem|hipporag2|mem0)
    : "${EMBEDDING_BASE_URL:?Set EMBEDDING_BASE_URL before launching this baseline.}"
    EMBEDDING_ARGS+=(--embedding-base-url "${EMBEDDING_BASE_URL}")
    ;;
esac

LIMIT_ARGS=(--max-queries "${MAX_QUERIES:-1000}")
if [[ -n "${MAX_CONTEXTS:-}" ]]; then
  LIMIT_ARGS+=(--max-contexts "${MAX_CONTEXTS}")
fi

MODEL_TAG="${EVALUATION_BACKBONE//\//_}"
RUN_DIR="${OUTPUT_ROOT}/${BASELINE}/${TASK// /_}/${MODEL_TAG}/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "baseline=${BASELINE} task=${TASK} evaluator=${EVALUATION_BACKBONE}"
echo "python=${PYTHON_BIN} hf_home=${HF_HOME} output=${RUN_DIR}"
nvidia-smi

"${PYTHON_BIN}" "${PROJECT_DIR}/main.py" \
  --task "${TASK}" \
  --baseline "${BASELINE}" \
  --output-dir "${RUN_DIR}" \
  --evaluation-backbone "${EVALUATION_BACKBONE}" \
  --evaluation-dtype "${EVALUATION_DTYPE:-bfloat16}" \
  --evaluation-device-map "${EVALUATION_DEVICE_MAP:-auto}" \
  "${EMBEDDING_ARGS[@]}" \
  "${LIMIT_ARGS[@]}" \
  "${DATA_ARGS[@]}" \
  "${OFFICIAL_ARGS[@]}"
