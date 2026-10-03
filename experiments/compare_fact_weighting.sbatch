#!/bin/bash
#SBATCH --job-name=memory_fact_weights
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/zliu328/agent-memory-outputs/memory_fact_weights_%j.out
set -euo pipefail
cd /oscar/home/zliu328/agent-memory
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD:/oscar/scratch/zliu328/agent-memory-outputs/optimization_support_ablation_seed42_20260917/experiment_code/ircot"
export NLTK_DATA=/oscar/scratch/zliu328/agent-memory-envs/nltk_data
export HF_HOME=/oscar/scratch/zliu328/hf_output
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENAI_API_KEY=local-cache-only
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=42
export MPLCONFIGDIR="${TMPDIR:-/tmp}/memory-fact-weights-${SLURM_JOB_ID}"
export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/memory-fact-weight-bytecode-${SLURM_JOB_ID}"
exec flock -n "/tmp/${USER}.agent-lock" \
 /oscar/scratch/zliu328/agent-memory-envs/ircot/bin/python -u -m experiments.compare_fact_weighting \
 --results /oscar/scratch/zliu328/agent-memory-outputs/optimization_simplified_amor_seed42_20260929 \
 --output /oscar/scratch/zliu328/agent-memory-outputs/optimization_simplified_amor_seed42_20260929/analysis/fact_weighting
