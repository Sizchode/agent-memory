#!/bin/bash
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --job-name=amor-legacy-manifest
set -euo pipefail
cd /users/zliu328/agent-memory
export PYTHONDONTWRITEBYTECODE=1
python_bin=/oscar/scratch/zliu328/agent-memory-envs/ircot/bin/python
"$python_bin" -m unittest experiments.test_backup_manifest -v
"$python_bin" -m experiments.backup_manifest \
  --base /oscar/scratch/zliu328/agent-memory-outputs \
  --include optimization_all_support_deletions_seed42_20260917 \
  --include optimization_raw_module_ablation_seed42_20260917 \
  --output "/oscar/scratch/zliu328/agent-memory-outputs/main_qa_multiseed_20261003/jobs/legacy-manifest-${SLURM_JOB_ID}.json"
