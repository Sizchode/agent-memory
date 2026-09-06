#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 '<task>' [--path locomo10.json] [--data-root hipporag2/reproduce/dataset]" >&2
  exit 2
fi

exec python3 "${PROJECT_DIR}/main.py" \
  --task "$1" \
  "${@:2}"
