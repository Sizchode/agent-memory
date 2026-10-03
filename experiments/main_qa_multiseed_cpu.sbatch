#!/bin/bash
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --job-name=amor-main-seeds-audit
set -euo pipefail
cd /users/zliu328/agent-memory
export PYTHONPATH="$PWD:/oscar/scratch/zliu328/agent-memory-outputs/optimization_support_ablation_seed42_20260917/experiment_code/ircot"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NLTK_DATA=/oscar/scratch/zliu328/agent-memory-envs/nltk_data
export HF_HOME=/oscar/scratch/zliu328/hf_output HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
python_bin=/oscar/scratch/zliu328/agent-memory-envs/ircot/bin/python
"$python_bin" -m unittest experiments.test_main_qa_multiseed experiments.test_component_artifacts -v
"$python_bin" -u -m experiments.main_qa_multiseed "$1" --output "$2"
