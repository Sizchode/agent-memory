#!/bin/bash
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=amor-encoder-preflight
set -euo pipefail
cd /users/zliu328/agent-memory
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
run_root=$1
mkdir -p "$run_root/jobs/$SLURM_JOB_ID"
scontrol show job "$SLURM_JOB_ID" > "$run_root/jobs/$SLURM_JOB_ID/slurm.txt"
checkquota
python_bin=/oscar/scratch/zliu328/agent-memory-envs/ircot/bin/python
"$python_bin" -m unittest experiments.test_encoder_service_environment experiments.test_encoder_robustness -v
"$python_bin" -u -m experiments.encoder_recovery_preflight "$run_root"
