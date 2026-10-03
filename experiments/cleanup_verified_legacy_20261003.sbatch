#!/bin/bash
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --job-name=amor-verified-legacy-cleanup
set -euo pipefail
cd /users/zliu328/agent-memory
export PYTHONDONTWRITEBYTECODE=1
cleanup_jobs=/oscar/scratch/zliu328/agent-memory-outputs/main_qa_multiseed_20261003/jobs
verified_manifest="$cleanup_jobs/legacy-backup-local-verified.json"
test "$(sha256sum "$verified_manifest" | cut -d ' ' -f 1)" = f32449215fff272e7efb4c32eedfdf4f8012609247c1a8b81b637327eb989c73

# The local backup has been independently hashed and matches the source manifest.
# Recompute source contents immediately before removing only the two audited roots.
/oscar/scratch/zliu328/agent-memory-envs/ircot/bin/python -m experiments.backup_manifest \
  --base /oscar/scratch/zliu328/agent-memory-outputs \
  --include optimization_all_support_deletions_seed42_20260917 \
  --include optimization_raw_module_ablation_seed42_20260917 \
  --output "$cleanup_jobs/legacy-predelete-${SLURM_JOB_ID}.json"
cmp "$verified_manifest" "$cleanup_jobs/legacy-predelete-${SLURM_JOB_ID}.json"
squeue -u "$USER" -o '%.18i %.40j %.12T %.60R'
du -x -s -B1 \
  /oscar/scratch/zliu328/agent-memory-outputs/optimization_all_support_deletions_seed42_20260917 \
  /oscar/scratch/zliu328/agent-memory-outputs/optimization_raw_module_ablation_seed42_20260917

# Explicit targets only. Preserve all other legacy results, runtime code and graphs.
rm -rf -- \
  /oscar/scratch/zliu328/agent-memory-outputs/optimization_all_support_deletions_seed42_20260917 \
  /oscar/scratch/zliu328/agent-memory-outputs/optimization_raw_module_ablation_seed42_20260917
test ! -e /oscar/scratch/zliu328/agent-memory-outputs/optimization_all_support_deletions_seed42_20260917
test ! -e /oscar/scratch/zliu328/agent-memory-outputs/optimization_raw_module_ablation_seed42_20260917
printf 'VERIFIED_LEGACY_CLEANUP_COMPLETE; local backup remains available.\n'
checkquota
