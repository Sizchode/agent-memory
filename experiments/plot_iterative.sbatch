#!/bin/bash
#SBATCH --job-name=amor_ircot_plots
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/oscar/scratch/zliu328/agent-memory-outputs/paper_costs_20260925/plot_%j.out
set -euo pipefail
cd /oscar/home/zliu328/agent-memory
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/amor-matplotlib-${SLURM_JOB_ID}"
exec flock -n "/tmp/${USER}.agent-lock" \
    /oscar/scratch/zliu328/agent-memory-envs/ircot/bin/python \
    -m experiments.report_paper_costs \
    --output /oscar/scratch/zliu328/agent-memory-outputs/paper_costs_20260925 \
    --iterative-ablation /oscar/scratch/zliu328/agent-memory-outputs/optimization_paper_ablation_seed42_20260927
