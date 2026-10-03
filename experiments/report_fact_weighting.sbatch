#!/bin/bash
#SBATCH --job-name=report_fact_weighting
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/oscar/scratch/zliu328/agent-memory-outputs/report_fact_weighting_%j.out
set -euo pipefail
cd /oscar/home/zliu328/agent-memory
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NLTK_DATA=/oscar/scratch/zliu328/agent-memory-envs/nltk_data
exec flock -n "/tmp/${USER}.agent-lock" \
 /oscar/scratch/zliu328/agent-memory-envs/ircot/bin/python -m experiments.report_fact_weighting \
 --results /oscar/scratch/zliu328/agent-memory-outputs/optimization_simplified_amor_seed42_20260929 \
 --document /oscar/home/zliu328/agent-memory/docs/memory_analysis
