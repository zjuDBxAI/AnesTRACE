#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${ANESBENCH_PYTHON_BIN:-python3}"
MODEL_ROOT="${ANESTRACE_MODEL_ROOT:-${PROJECT_ROOT}/checkpoints}"
DATA_FILE="${ANESBENCH_L2_MLLM_DATA_FILE:-${PROJECT_ROOT}/level_two/Level_two_B5_v2_en.jsonl}"
OUTPUT_ROOT="${ANESBENCH_L2_MLLM_OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/level_two_mllm}"
GPU_IDS="${ANESBENCH_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
DEVICE_MAP="${ANESBENCH_DEVICE_MAP:-auto}"
MAX_MEMORY_PER_GPU="${ANESBENCH_MAX_MEMORY_PER_GPU:-}"
BATCH_SIZE="${ANESBENCH_BATCH_SIZE:-1}"
MODEL_PATHS=()
EXTRA_ARGS=()

DEFAULT_MODEL_PATHS=(
  "${MODEL_ROOT}/Qwen/Qwen3-VL-32B-Instruct"
  "${MODEL_ROOT}/Qwen/Qwen3-VL-8B-Instruct"
  "${MODEL_ROOT}/Qwen/Qwen3.5-9B"
  "${MODEL_ROOT}/Qwen/Qwen3.5-27B"
  "${MODEL_ROOT}/Qwen/Qwen3.8-27B"
  "${MODEL_ROOT}/LLaVA/LLaVA-OneVision-2-8B-Instruct"
  "${MODEL_ROOT}/MLLM/Fleming-VL-8B"
  "${MODEL_ROOT}/MLLM/Fleming-VL-38B"
  "${MODEL_ROOT}/MLLM/Lingshu-32B"
  "${MODEL_ROOT}/MLLM/Lingshu-I-8B"
  "${MODEL_ROOT}/InternVL/InternVL3_5-8B"
  "${MODEL_ROOT}/InternVL/InternVL3_5-38B"
)

usage() {
  cat <<'EOF'
Usage:
  level2/inference/run_multimodal.sh [wrapper options] [-- run_multimodal.py options]

Wrapper options:
  --model-path PATH          Model directory; repeat to select models.
  --gpus IDS                 CUDA_VISIBLE_DEVICES, for example 0 or 0,1.
  --python PATH              Python interpreter.
  --data PATH                English Level Two B5 JSONL.
  --output-root PATH         Root for per-model prediction directories.
  --device-map MAP           Transformers/Accelerate device map.
  --max-memory-per-gpu SIZE  Per-visible-GPU limit, for example 70GiB.
  --batch-size N             Requested generation batch size.
  -h, --help                 Show this help.
  --                         Pass remaining arguments to the Python runner.

With no --model-path, all 12 configured MLLMs run sequentially. A failed model
is reported but does not stop later models. Existing non-error responses resume.

Examples:
  scripts/inference/run_level_two_mllm.sh --gpus 0,1 --batch-size 2 -- --limit 1
  scripts/inference/run_level_two_mllm.sh --model-path /path/to/model --gpus 0 -- --split test
EOF
}

require_value() {
  if (( $# < 2 )); then
    echo "Missing value for $1" >&2
    usage >&2
    exit 2
  fi
}

while (( $# > 0 )); do
  case "$1" in
    --model-path)
      require_value "$@"; MODEL_PATHS+=("$2"); shift 2 ;;
    --model-path=*) MODEL_PATHS+=("${1#*=}"); shift ;;
    --gpus)
      require_value "$@"; GPU_IDS="$2"; shift 2 ;;
    --gpus=*) GPU_IDS="${1#*=}"; shift ;;
    --python)
      require_value "$@"; PYTHON_BIN="$2"; shift 2 ;;
    --python=*) PYTHON_BIN="${1#*=}"; shift ;;
    --data)
      require_value "$@"; DATA_FILE="$2"; shift 2 ;;
    --data=*) DATA_FILE="${1#*=}"; shift ;;
    --output-root)
      require_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
    --output-root=*) OUTPUT_ROOT="${1#*=}"; shift ;;
    --device-map)
      require_value "$@"; DEVICE_MAP="$2"; shift 2 ;;
    --device-map=*) DEVICE_MAP="${1#*=}"; shift ;;
    --max-memory-per-gpu)
      require_value "$@"; MAX_MEMORY_PER_GPU="$2"; shift 2 ;;
    --max-memory-per-gpu=*) MAX_MEMORY_PER_GPU="${1#*=}"; shift ;;
    --batch-size)
      require_value "$@"; BATCH_SIZE="$2"; shift 2 ;;
    --batch-size=*) BATCH_SIZE="${1#*=}"; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if (( ${#MODEL_PATHS[@]} == 0 )); then
  MODEL_PATHS=("${DEFAULT_MODEL_PATHS[@]}")
fi
if [[ "${DATA_FILE}" != /* ]]; then
  DATA_FILE="${PROJECT_ROOT}/${DATA_FILE}"
fi
if [[ "${OUTPUT_ROOT}" != /* ]]; then
  OUTPUT_ROOT="${PROJECT_ROOT}/${OUTPUT_ROOT}"
fi
if ! command -v -- "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python is not available: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "${DATA_FILE}" ]]; then
  echo "Dataset does not exist: ${DATA_FILE}" >&2
  exit 1
fi
if [[ -z "${GPU_IDS}" || -z "${DEVICE_MAP}" ]]; then
  echo "--gpus and --device-map must not be empty." >&2
  exit 2
fi
if [[ ! "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--batch-size must be a positive integer: ${BATCH_SIZE}" >&2
  exit 2
fi
if [[ -n "${MAX_MEMORY_PER_GPU}" && ! "${MAX_MEMORY_PER_GPU}" =~ ^[1-9][0-9]*(\.[0-9]+)?(GiB|MiB|GB|MB)$ ]]; then
  echo "Invalid --max-memory-per-gpu: ${MAX_MEMORY_PER_GPU}" >&2
  exit 2
fi

for index in "${!MODEL_PATHS[@]}"; do
  model_path="${MODEL_PATHS[index]%/}"
  if [[ "${model_path}" != /* ]]; then
    model_path="${PROJECT_ROOT}/${model_path}"
  fi
  if [[ ! -d "${model_path}" || ! -f "${model_path}/config.json" ]]; then
    echo "Model directory or config.json does not exist: ${model_path}" >&2
    exit 1
  fi
  MODEL_PATHS[index]="${model_path}"
done

mkdir -p -- "${OUTPUT_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
RESOURCE_ARGS=(
  --device-map "${DEVICE_MAP}"
  --batch-size "${BATCH_SIZE}"
)
if [[ -n "${MAX_MEMORY_PER_GPU}" ]]; then
  RESOURCE_ARGS+=(--max-memory-per-gpu "${MAX_MEMORY_PER_GPU}")
fi

cd -- "${PROJECT_ROOT}"
failures=()
total="${#MODEL_PATHS[@]}"
for index in "${!MODEL_PATHS[@]}"; do
  model_path="${MODEL_PATHS[index]}"
  model_name="$(basename -- "${model_path}")"
  if [[ ! "${model_name}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "Invalid model directory name: ${model_name}" >&2
    exit 2
  fi
  output="${OUTPUT_ROOT}/${model_name}/level-two-b5-paired-images-en.jsonl"
  echo
  echo "[$((index + 1))/${total}] Starting Level Two MLLM: ${model_path}"
  echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
  echo "Batch size: ${BATCH_SIZE}"
  echo "Output: ${output}"

  if "${PYTHON_BIN}" "${PROJECT_ROOT}/level2/inference/run_multimodal.py" "${DATA_FILE}" \
      --model-path "${model_path}" \
      --output "${output}" \
      "${RESOURCE_ARGS[@]}" \
      "${EXTRA_ARGS[@]}"; then
    echo "[$((index + 1))/${total}] Completed: ${model_name}"
  else
    status=$?
    failures+=("${model_name} exit=${status}")
    echo "[$((index + 1))/${total}] Failed: ${model_name} (exit ${status}); continuing with the next model." >&2
  fi
done

if (( ${#failures[@]} > 0 )); then
  echo >&2
  echo "Failed model runs:" >&2
  printf '  - %s\n' "${failures[@]}" >&2
  exit 1
fi

echo
echo "All ${total} Level Two MLLM runs completed."
