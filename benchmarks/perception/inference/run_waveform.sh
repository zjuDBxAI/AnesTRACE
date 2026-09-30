#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${ANESBENCH_PYTHON_BIN:-python3}"
DATA_FILE="${ANESBENCH_WAVEFORM_DATA_FILE:-${PROJECT_ROOT}/data/private/Level_one_Waveform_v1_zh.jsonl}"
MODEL_PATH="${ANESBENCH_MODEL_PATH:-${PROJECT_ROOT}/checkpoints/Qwen/Qwen3-VL-32B-Instruct}"
OUTPUT_FILE="${ANESBENCH_OUTPUT_FILE:-${PROJECT_ROOT}/outputs/Qwen3-VL-32B-Instruct/waveform.jsonl}"
GPU_IDS="${ANESBENCH_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"

command -v -- "${PYTHON_BIN}" >/dev/null 2>&1 || { echo "Python is not available: ${PYTHON_BIN}" >&2; exit 1; }
[[ -f "${DATA_FILE}" ]] || { echo "Dataset does not exist: ${DATA_FILE}" >&2; exit 1; }
[[ -d "${MODEL_PATH}" ]] || { echo "Model does not exist: ${MODEL_PATH}" >&2; exit 1; }

mkdir -p -- "$(dirname -- "${OUTPUT_FILE}")"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
exec "${PYTHON_BIN}" "${PROJECT_ROOT}/benchmarks/perception/inference/run.py" \
  "${DATA_FILE}" \
  --language "${ANESBENCH_LANGUAGE:-zh}" \
  --model-path "${MODEL_PATH}" \
  --output "${OUTPUT_FILE}" \
  "$@"
