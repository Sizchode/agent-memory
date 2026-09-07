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
#SBATCH --output=agent_memory_%j.out
#SBATCH --export=ALL,PYTORCH_JIT=0

set -euo pipefail

: "${TASK:?Set TASK, for example: SH-Doc QA}"
: "${BASELINE:?Set BASELINE: bm25, dense, lightmem, hipporag2, or mem0}"
: "${EVALUATION_BACKBONE:?Set EVALUATION_BACKBONE to a Hugging Face model ID}"

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
HF_HOME="${HF_HOME:-/oscar/data/sbach/zliu328/hf_output}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs}"
CONDA_ENV="${CONDA_ENV:-llm_ft}"
ENV_ROOT="${ENV_ROOT:-/oscar/scratch/zliu328/agent-memory-envs}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

export HF_HOME
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

case "${BASELINE}" in
  bm25|dense) PYTHON_BIN="${ENV_ROOT}/runner/bin/python" ;;
  lightmem|mem0) PYTHON_BIN="${ENV_ROOT}/${BASELINE}/bin/python" ;;
  hipporag2) PYTHON_BIN="${ENV_ROOT}/hipporag/bin/python" ;;
  *) echo "Unknown BASELINE: ${BASELINE}" >&2; exit 2 ;;
esac

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment is missing: ${PYTHON_BIN}" >&2
  exit 2
fi

mkdir -p "${PROJECT_DIR}/logs"
MODEL_TAG="${EVALUATION_BACKBONE//\//_}"
RUN_DIR="${OUTPUT_ROOT}/${BASELINE}/${TASK// /_}/${MODEL_TAG}/${SLURM_JOB_ID}"

DATA_ARGS=()
case "${TASK}" in
  LoCoMo) : "${LOCOMO_PATH:?Set LOCOMO_PATH for LoCoMo}"; DATA_ARGS+=(--path "${LOCOMO_PATH}") ;;
  MuSiQue|2WikiMultiHopQA|HotpotQA) : "${HIPPORAG_DATA_ROOT:?Set HIPPORAG_DATA_ROOT}"; DATA_ARGS+=(--data-root "${HIPPORAG_DATA_ROOT}") ;;
esac

OFFICIAL_ARGS=()
case "${BASELINE}" in
  lightmem|hipporag2|mem0)
    : "${OFFICIAL_CONFIG:?Set OFFICIAL_CONFIG for ${BASELINE}}"
    OFFICIAL_ARGS+=(--official-config "${OFFICIAL_CONFIG}")
    ;;
esac

EMBEDDING_ARGS=()
case "${BASELINE}" in
  dense|lightmem|hipporag2|mem0)
    : "${EMBEDDING_BASE_URL:?Set EMBEDDING_BASE_URL to the OpenAI-compatible embedding service}"
    EMBEDDING_ARGS+=(--embedding-base-url "${EMBEDDING_BASE_URL}")
    ;;
esac

LIMIT_ARGS=(--max-queries "${MAX_QUERIES:-1000}")
if [[ -n "${MAX_CONTEXTS:-}" ]]; then
  LIMIT_ARGS+=(--max-contexts "${MAX_CONTEXTS}")
fi

echo "Job ${SLURM_JOB_ID} on $(hostname)"
echo "Python: ${PYTHON_BIN}"
echo "UV environment root: ${ENV_ROOT}"
echo "Evaluation backbone (local Hugging Face): ${EVALUATION_BACKBONE}"
echo "HF_HOME: ${HF_HOME}"
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
