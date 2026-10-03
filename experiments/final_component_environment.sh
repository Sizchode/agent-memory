#!/bin/bash
# Shared environment only. Resource allocation belongs to the stage's sbatch file.
set -euo pipefail
cd /users/zliu328/agent-memory
export PYTHONPATH="$PWD:/oscar/scratch/zliu328/agent-memory-outputs/optimization_support_ablation_seed42_20260917/experiment_code/ircot"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NLTK_DATA=/oscar/scratch/zliu328/agent-memory-envs/nltk_data
export HF_HOME=/oscar/scratch/zliu328/hf_output
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENAI_API_KEY=local-cache-only
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=42 PYTHONDONTWRITEBYTECODE=1
export TORCHINDUCTOR_COMPILE_THREADS=1 MAX_JOBS=1
export VLLM_CACHE_ROOT="${TMPDIR:-/tmp}/amor-component-vllm-${SLURM_JOB_ID}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/amor-component-matplotlib-${SLURM_JOB_ID}"
module load cuda/12.9.0-cinr
python_bin=/oscar/scratch/zliu328/agent-memory-envs/ircot/bin/python
