#!/bin/bash
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --job-name=amor-legacy-dependency-audit
set -euo pipefail
cd /users/zliu328/agent-memory
output_root=/oscar/scratch/zliu328/agent-memory-outputs
candidates=(
  optimization_full_index_module_ablation_seed42_20260917
  optimization_all_support_deletions_seed42_20260917
  optimization_raw_module_ablation_seed42_20260917
)
printf 'Read-only candidate dependency inventory. No deletion or archive creation.\n'
squeue -u "$USER" -o '%.18i %.40j %.12T %.60R'
for candidate in "${candidates[@]}"; do
  printf '\nCANDIDATE %s\n' "$candidate"
  test -d "$output_root/$candidate"
  test ! -L "$output_root/$candidate"
  realpath "$output_root/$candidate"
  du -x -B1 --max-depth=2 "$output_root/$candidate"
done
printf '\nSYMLINKS in experiment repository and outputs\n'
find /users/zliu328/agent-memory "$output_root" -type l -printf '%p -> %l\n'
printf '\nLITERAL REFERENCES outside candidate output trees\n'
search_options=(--hidden --no-ignore --text --files-with-matches --fixed-strings)
for candidate in "${candidates[@]}"; do
  search_options+=(-e "$candidate" --glob "!**/$candidate/**")
done
set +e
rg "${search_options[@]}" /users/zliu328/agent-memory/experiments /users/zliu328/agent-memory/optimization "$output_root"
search_status=$?
set -e
if (( search_status > 1 )); then
  exit "$search_status"
fi
printf '\nDEPENDENCY_INVENTORY_COMPLETE; literal absence alone is not deletion approval.\n'
