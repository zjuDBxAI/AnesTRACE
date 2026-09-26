#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
AGENT_ROOT="${PROJECT_ROOT}/level3/agent"
PYTHON_BIN="${ANESBENCH_L3_AGENT_PYTHON_BIN:-python3}"
INPUT_FILE="${ANESTRACE_INPUT_FILE:-${AGENT_ROOT}/data/Level_three_v3_en_agent.jsonl}"
CONFIG_FILE="${ANESTRACE_CONFIG_FILE:-${AGENT_ROOT}/configs/default.json}"
OUTPUT_DIR="${ANESTRACE_OUTPUT_DIR:-${AGENT_ROOT}/outputs/level-three-qwen35-27b-level-two-aligned}"

command -v -- "${PYTHON_BIN}" >/dev/null 2>&1 || { echo "Python is not available: ${PYTHON_BIN}" >&2; exit 1; }
[[ -f "${INPUT_FILE}" ]] || { echo "Agent dataset does not exist: ${INPUT_FILE}" >&2; exit 1; }
[[ -f "${CONFIG_FILE}" ]] || { echo "Agent configuration does not exist: ${CONFIG_FILE}" >&2; exit 1; }

if (( $# )) && [[ "$1" == "--help" || "$1" == "-h" ]]; then
  cat <<'EOF'
Usage: level3/inference/run_agent.sh [agent run options]

Environment:
  ANESBENCH_L3_AGENT_PYTHON_BIN  Python interpreter.
  ANESTRACE_INPUT_FILE           Level Three Agent JSONL.
  ANESTRACE_CONFIG_FILE          Agent configuration JSON.
  ANESTRACE_OUTPUT_DIR           Output directory.

Examples:
  level3/inference/run_agent.sh --limit 1 --workers 1
  level3/inference/run_agent.sh --resume
EOF
  exit 0
fi

cd -- "${AGENT_ROOT}"
exec "${PYTHON_BIN}" "${PROJECT_ROOT}/level3/inference/run_agent.py" \
  --config "${CONFIG_FILE}" run --input "${INPUT_FILE}" \
  --output-dir "${OUTPUT_DIR}" "$@"
