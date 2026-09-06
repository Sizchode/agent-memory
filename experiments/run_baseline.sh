#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

if [[ "$#" -lt 2 ]]; then
  echo "Usage: $0 <dataset.csv> <target-column> [additional main.py arguments]" >&2
  exit 2
fi

exec python3 "${PROJECT_DIR}/main.py" \
  --dataset "$1" \
  --target-column "$2" \
  "${@:3}"
