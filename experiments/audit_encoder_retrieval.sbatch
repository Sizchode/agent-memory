#!/bin/bash
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=amor-encoder-retr-audit
set -euo pipefail
cd /users/zliu328/agent-memory
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NLTK_DATA=/oscar/scratch/zliu328/agent-memory-envs/nltk_data
export HF_HOME=/oscar/scratch/zliu328/hf_output HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python_bin=/oscar/scratch/zliu328/agent-memory-envs/ircot/bin/python
"$python_bin" -m unittest experiments.test_encoder_retrieval_audit experiments.test_encoder_recognition_log_audit -v
"$python_bin" -u -m experiments.audit_encoder_retrieval "$@"
