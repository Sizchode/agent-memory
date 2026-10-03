#!/bin/bash
#SBATCH --job-name=memory_findings_report
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/oscar/scratch/zliu328/agent-memory-outputs/memory_findings_report_%j.out
set -euo pipefail
cd /oscar/home/zliu328/agent-memory
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD"
export NLTK_DATA=/oscar/scratch/zliu328/agent-memory-envs/nltk_data
export MPLCONFIGDIR="${TMPDIR:-/tmp}/memory-findings-${SLURM_JOB_ID}"
exec flock -n "/tmp/${USER}.agent-lock" \
 /oscar/scratch/zliu328/agent-memory-envs/ircot/bin/python -u -m experiments.report_memory_findings \
 --results /oscar/scratch/zliu328/agent-memory-outputs/optimization_simplified_amor_seed42_20260929 \
 --output /oscar/home/zliu328/agent-memory/docs/memory_analysis/data
