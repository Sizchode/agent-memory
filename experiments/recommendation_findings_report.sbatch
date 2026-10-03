#!/bin/bash
#SBATCH --job-name=amor_findings_audit
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/oscar/scratch/zliu328/agent-memory-outputs/two_findings_audit_%j.out
set -euo pipefail
cd /oscar/home/zliu328/agent-memory
export PYTHONPATH="$PWD:/oscar/scratch/zliu328/agent-memory-outputs/optimization_support_ablation_seed42_20260917/experiment_code/ircot"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NLTK_DATA=/oscar/scratch/zliu328/agent-memory-envs/nltk_data
export PYTHONHASHSEED=42
exec /oscar/scratch/zliu328/agent-memory-envs/ircot/bin/python -u -m experiments.report_recommendation_findings "$1"
