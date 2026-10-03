#!/bin/bash
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --job-name=amor-encoder-qa-audit
set -euo pipefail
cd /users/zliu328/agent-memory
export PYTHONPATH="$PWD:/oscar/scratch/zliu328/agent-memory-outputs/optimization_support_ablation_seed42_20260917/experiment_code/ircot"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NLTK_DATA=/oscar/scratch/zliu328/agent-memory-envs/nltk_data
export HF_HOME=/oscar/scratch/zliu328/hf_output HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python_bin=/oscar/scratch/zliu328/agent-memory-envs/ircot/bin/python
"$python_bin" -m unittest experiments.test_encoder_qa_audit -v
"$python_bin" -u -m experiments.audit_encoder_qa "$@"
