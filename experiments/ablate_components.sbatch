#!/bin/bash
#SBATCH --job-name=amor_components
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/oscar/scratch/zliu328/agent-memory-outputs/optimization_component_ablation_seed42_20260927/%j.out
set -euo pipefail
cd /oscar/home/zliu328/agent-memory
export PYTHONPATH="$PWD:/oscar/scratch/zliu328/agent-memory-outputs/optimization_support_ablation_seed42_20260917/experiment_code/ircot"
export HF_HOME=/oscar/scratch/zliu328/hf_output
export NLTK_DATA=/oscar/scratch/zliu328/agent-memory-envs/nltk_data
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=42
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENAI_API_KEY=local-cache-only
export TORCHINDUCTOR_COMPILE_THREADS=1 MAX_JOBS=1
export VLLM_CACHE_ROOT="${TMPDIR:-/tmp}/amor-components-${SLURM_JOB_ID}"
module load cuda/12.9.0-cinr
exec /oscar/scratch/zliu328/agent-memory-envs/ircot/bin/python -u -m experiments.ablate_components "$@"
