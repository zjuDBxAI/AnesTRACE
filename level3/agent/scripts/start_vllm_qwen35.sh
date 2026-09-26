#!/usr/bin/env bash
set -Eeuo pipefail

VLLM_BIN="${ANESTRACE_VLLM_BIN:-vllm}"
MODEL_PATH="${ANESTRACE_MODEL_PATH:-}"
SERVED_MODEL_NAME="${ANESTRACE_SERVED_MODEL_NAME:-Qwen3.5-27B}"
SERVER_HOST="${ANESTRACE_SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${ANESTRACE_SERVER_PORT:-8002}"
TP_SIZE="${ANESTRACE_TP_SIZE:-2}"
MAX_MODEL_LEN="${ANESTRACE_MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${ANESTRACE_MAX_NUM_SEQS:-8}"
GPU_MEMORY_UTILIZATION="${ANESTRACE_GPU_MEMORY_UTILIZATION:-0.42}"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "Set ANESTRACE_MODEL_PATH to a local Qwen3.5 checkpoint." >&2
  exit 2
fi
if ! command -v "${VLLM_BIN}" >/dev/null 2>&1; then
  echo "vLLM executable was not found: ${VLLM_BIN}" >&2
  exit 1
fi
test -f "${MODEL_PATH}/config.json" || {
  echo "Model config does not exist: ${MODEL_PATH}/config.json" >&2
  exit 1
}

export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
exec "${VLLM_BIN}" serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${SERVER_HOST}" \
  --port "${SERVER_PORT}" \
  --dtype bfloat16 \
  --seed 42 \
  --tensor-parallel-size "${TP_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --generation-config vllm \
  --enforce-eager \
  --disable-custom-all-reduce \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
