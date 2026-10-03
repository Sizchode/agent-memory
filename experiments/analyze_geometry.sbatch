#!/bin/bash
#SBATCH --job-name=amor_geometry
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/oscar/scratch/zliu328/agent-memory-outputs/analysis_geometry_20260928_%j.out
set -euo pipefail
cd /oscar/home/zliu328/agent-memory
export PYTHONPATH="$PWD:/oscar/scratch/zliu328/agent-memory-outputs/optimization_support_ablation_seed42_20260917/experiment_code/ircot"
export HF_HOME=/oscar/scratch/zliu328/hf_output
export NLTK_DATA=/oscar/scratch/zliu328/agent-memory-envs/nltk_data
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=42
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENAI_API_KEY=local-cache-only
export MPLCONFIGDIR="${TMPDIR:-/tmp}/amor-geometry-${SLURM_JOB_ID}"
exec flock -n "/tmp/${USER}.agent-lock" \
    /oscar/scratch/zliu328/agent-memory-envs/ircot/bin/python -u -m experiments.analyze_geometry \
    --output /oscar/scratch/zliu328/agent-memory-outputs/analysis_geometry_20260928 "$@"
