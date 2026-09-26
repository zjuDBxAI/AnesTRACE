#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
L3_BIN="${ANESTRACE_L3_BIN:-anestrace-l3}"
INPUT_FILE="${ANESTRACE_L3_DATA:-${PROJECT_DIR}/data/Level_three_v3_en_agent.jsonl}"
CONFIG_FILE="${ANESTRACE_L3_CONFIG:-${PROJECT_DIR}/configs/default.json}"
OUTPUT_DIR="${ANESTRACE_L3_OUTPUT:-${PROJECT_DIR}/outputs/qwen35-27b}"

if ! command -v "${L3_BIN}" >/dev/null 2>&1; then
  echo "CLI was not found: ${L3_BIN}; install this subproject with pip install -e ." >&2
  exit 1
fi
test -f "${INPUT_FILE}" || {
  echo "Level Three agent dataset does not exist: ${INPUT_FILE}" >&2
  exit 1
}

cd -- "${PROJECT_DIR}"
exec "${L3_BIN}" --config "${CONFIG_FILE}" run \
  --input "${INPUT_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
