#!/bin/bash
# Create the isolated experiment and local-generator environments.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ENV_ROOT="${ENV_ROOT:-/oscar/scratch/zliu328/agent-memory-envs}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

"${SCRIPT_DIR}/apply_baseline_patches.sh"

for environment in runner lightmem hipporag mem0 qwen_generator; do
  environment_path="${ENV_ROOT}/${environment}"
  if [[ ! -x "${environment_path}/bin/python" ]]; then
    uv venv "${environment_path}" --python "${PYTHON_VERSION}"
  fi
done

uv pip install --python "${ENV_ROOT}/runner/bin/python" -r "${SCRIPT_DIR}/common.txt"
uv pip install --python "${ENV_ROOT}/lightmem/bin/python" -r "${SCRIPT_DIR}/lightmem/requirements.txt" -e "${PROJECT_DIR}/baseline_algorithms/LightMem"
uv pip install --python "${ENV_ROOT}/hipporag/bin/python" -r "${SCRIPT_DIR}/hipporag/requirements.txt" -e "${PROJECT_DIR}/baseline_algorithms/HippoRAG[qdrant,transformers-embedding]"
uv pip install --python "${ENV_ROOT}/mem0/bin/python" -r "${SCRIPT_DIR}/mem0/requirements.txt" -e "${PROJECT_DIR}/baseline_algorithms/mem0"
uv pip install --python "${ENV_ROOT}/qwen_generator/bin/python" \
  --torch-backend auto \
  --index-strategy unsafe-best-match \
  -r "${SCRIPT_DIR}/qwen_generator.txt"

export NLTK_DATA="${ENV_ROOT}/nltk_data"
mkdir -p "${NLTK_DATA}"
"${ENV_ROOT}/runner/bin/python" -m nltk.downloader -d "${NLTK_DATA}" punkt punkt_tab

for environment in runner lightmem hipporag mem0 qwen_generator; do
  uv pip check --python "${ENV_ROOT}/${environment}/bin/python"
done

echo "Installed five isolated environments from ${PROJECT_DIR}/requirements"
