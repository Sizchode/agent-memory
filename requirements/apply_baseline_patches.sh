#!/bin/bash
# Apply the exact compatibility patches used by the controlled experiment.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

apply_exact_patch() {
  local repository="$1"
  local patch_path="$2"

  if git -C "${repository}" apply --reverse --check "${patch_path}" >/dev/null 2>&1; then
    return
  fi
  git -C "${repository}" apply --check "${patch_path}"
  git -C "${repository}" apply "${patch_path}"
}

apply_exact_patch \
  "${PROJECT_DIR}/baseline_algorithms/HippoRAG" \
  "${PROJECT_DIR}/baseline_patches/hipporag_local_models.patch"
apply_exact_patch \
  "${PROJECT_DIR}/baseline_algorithms/LightMem" \
  "${PROJECT_DIR}/baseline_patches/lightmem_vllm_and_source_id.patch"
