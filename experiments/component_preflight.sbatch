#!/bin/bash
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --job-name=amor-components-check
set -euo pipefail
cd /users/zliu328/agent-memory
export PYTHONPATH="$PWD:/oscar/scratch/zliu328/agent-memory-outputs/optimization_support_ablation_seed42_20260917/experiment_code/ircot"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HOME=/oscar/scratch/zliu328/hf_output
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=42
export PYTHONDONTWRITEBYTECODE=1
/oscar/scratch/zliu328/agent-memory-envs/ircot/bin/python -m unittest experiments.test_final_component_ablation -v
checkquota
du -x -h --max-depth=2 /oscar/home/zliu328 /oscar/scratch/zliu328
