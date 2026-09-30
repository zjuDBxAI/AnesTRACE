#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${ANESBENCH_EVAL_PYTHON_BIN:-python3}"

command -v -- "${PYTHON_BIN}" >/dev/null 2>&1 || { echo "Python is not available: ${PYTHON_BIN}" >&2; exit 1; }
exec "${PYTHON_BIN}" "${PROJECT_ROOT}/benchmarks/multi_step/evaluation/evaluate.py" "$@"
