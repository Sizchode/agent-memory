#!/bin/bash
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --job-name=amor-sensitivity-plot
set -euo pipefail
cd /users/zliu328/agent-memory
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NLTK_DATA=/oscar/scratch/zliu328/agent-memory-envs/nltk_data
export HF_HOME=/oscar/scratch/zliu328/hf_output HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/oscar/scratch/zliu328/agent-memory-envs/ircot/bin/python -u -m experiments.report_final_parameter_sensitivity "$1"
