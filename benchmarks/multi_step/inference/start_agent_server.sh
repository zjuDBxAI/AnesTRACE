#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
SERVER_SCRIPT="${PROJECT_ROOT}/benchmarks/multi_step/agent/scripts/start_vllm_qwen35.sh"

[[ -x "${SERVER_SCRIPT}" ]] || {
  echo "Agent server launcher does not exist or is not executable: ${SERVER_SCRIPT}" >&2
  exit 1
}
exec "${SERVER_SCRIPT}" "$@"
