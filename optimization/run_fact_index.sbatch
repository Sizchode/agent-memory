#!/bin/bash
#SBATCH --job-name=graph_fact_index
#SBATCH --partition=gpu-he
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/oscar/home/zliu328/graph_fact_index_%j.out
set -euo pipefail
: "${EXPERIMENT_ID:?Set a fresh output experiment}"
: "${SCHEMA_EXPERIMENT_ID:?Set the completed schema experiment}"
cd /oscar/home/zliu328/agent-memory
module load cuda/12.9.0-cinr
export HF_HOME=/oscar/scratch/zliu328/hf_output HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NLTK_DATA=/oscar/scratch/zliu328/agent-memory-envs/nltk_data
export PYTHONHASHSEED=42 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export VLLM_API_KEY=local-qwen-generator
export OPENAI_API_KEY="$VLLM_API_KEY"
VLLM=/oscar/scratch/zliu328/llm_tool_ckpt/venvs/vllm/bin/vllm
export PATH="$(dirname "$VLLM"):$PATH"
PORT=$((10000 + SLURM_JOB_ID % 50000))
PYTHON=/oscar/scratch/zliu328/agent-memory-envs/lightmem/bin/python
TP=1
GPU_MEMORY=0.85
if [[ "${SOURCE_RUNTIME:-0}" == 1 ]]; then
    PYTHON=/oscar/scratch/zliu328/agent-memory-envs/hipporag/bin/python
    TP=2
    GPU_MEMORY=0.75
fi
"$VLLM" serve Qwen/Qwen3-30B-A3B-Instruct-2507 --host 127.0.0.1 --port "$PORT" \
    --dtype bfloat16 --tensor-parallel-size "$TP" --max-model-len 32768 \
    --gpu-memory-utilization "$GPU_MEMORY" --seed 42 --generation-config vllm --api-key "$VLLM_API_KEY" &
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
TASK_ARGS=(--tasks FactConsolidation-SH FactConsolidation-MH 'SH-Doc QA' 'MH-Doc QA' LoCoMo 2WikiMultiHopQA)
if [[ "${CONTINUE_REMAINING_TASKS:-0}" == 1 ]]; then
    TASK_ARGS=(--resume-completed-groups --tasks 'MH-Doc QA' LoCoMo 2WikiMultiHopQA)
fi
if [[ "${CONTINUE_LONG_TASKS:-0}" == 1 ]]; then
    TASK_ARGS=(--resume-completed-groups --tasks LoCoMo 2WikiMultiHopQA)
fi
"$PYTHON" -u -m optimization.run_fact_index \
    --generator-base-url "http://127.0.0.1:$PORT/v1" \
    --schema-root "/oscar/scratch/zliu328/agent-memory-outputs/$SCHEMA_EXPERIMENT_ID" \
    --output-root "/oscar/scratch/zliu328/agent-memory-outputs/$EXPERIMENT_ID" \
    "${TASK_ARGS[@]}"
