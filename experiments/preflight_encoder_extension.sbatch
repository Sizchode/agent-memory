#!/bin/bash
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --job-name=amor-encoder-extension-check
set -euo pipefail
cd /users/zliu328/agent-memory
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
sha256sum experiments/preflight_encoder_extension.py experiments/preflight_encoder_extension.sbatch
/oscar/scratch/zliu328/agent-memory-envs/ircot/bin/python -u -m experiments.preflight_encoder_extension
