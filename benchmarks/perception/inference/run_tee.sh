#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

DEFAULT_MODEL_PATH="${PROJECT_ROOT}/checkpoints/Qwen/Qwen3-VL-32B-Instruct"
DEFAULT_PYTHON_BIN="python3"

ENV_MODEL_PATH="${ANESBENCH_MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
MODEL_NAME="${ANESBENCH_MODEL_NAME:-}"
PYTHON_BIN="${ANESBENCH_PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}"
LANGUAGE="${ANESBENCH_LANGUAGE:-zh}"
DATA_FILE="${ANESBENCH_TEE_DATA_FILE:-}"
GPU_IDS="${ANESBENCH_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
DEVICE_MAP="${ANESBENCH_DEVICE_MAP:-auto}"
MAX_MEMORY_PER_GPU="${ANESBENCH_MAX_MEMORY_PER_GPU:-}"
BATCH_SIZE="${ANESBENCH_BATCH_SIZE:-1}"
OUTPUT_FILE="${ANESBENCH_OUTPUT_FILE:-}"
MODEL_PATHS=()
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  benchmarks/perception/inference/run_tee.sh [wrapper options] [run.py options]

Wrapper options:
  --model-path PATH   Model directory; repeat to run models sequentially.
  --model-name NAME   Output directory name; valid only for a single model.
  --language LANG     Benchmark language: zh, en, or both (default: zh).
                      both runs en first, then zh, for each model.
  --gpus IDS          CUDA_VISIBLE_DEVICES value, for example 0 or 0,1.
  --device-map MAP    Transformers/Accelerate model placement (default: auto).
  --max-memory-per-gpu LIMIT
                      Per-visible-GPU model memory limit, for example 70GiB.
  --batch-size N      Same-media requests per generate call (default: 1).
  --output PATH       Prediction JSONL; single model/language modes only.
  --data PATH         Frozen TEE JSONL override; unavailable with both.
  --python PATH       Python interpreter containing inference dependencies.
  -h, --help          Show this help.
  --                  Pass all following arguments to run_level_one.py.

Examples:
  benchmarks/perception/inference/run_tee.sh
  benchmarks/perception/inference/run_tee.sh --model-path /path/to/Qwen3-VL-8B-Instruct
  benchmarks/perception/inference/run_tee.sh --model-path /path/to/model --model-name model-fp16 \
    --dtype float16 --video-num-frames 8
  benchmarks/perception/inference/run_tee.sh --model-path /path/to/large-model --gpus 0,1,2,3 \
    --device-map balanced --max-memory-per-gpu 70GiB --batch-size 2
  benchmarks/perception/inference/run_tee.sh --model-path /path/to/model --task-name cfd_presence --limit 5
  benchmarks/perception/inference/run_tee.sh \
    --model-path /path/to/model-a \
    --model-path /path/to/model-b \
    --model-path /path/to/model-c \
    --language both

Environment alternatives:
  ANESBENCH_MODEL_PATH
  ANESBENCH_MODEL_NAME
  ANESBENCH_LANGUAGE
  ANESBENCH_PYTHON_BIN
  ANESBENCH_TEE_DATA_FILE
  ANESBENCH_CUDA_VISIBLE_DEVICES
  ANESBENCH_DEVICE_MAP
  ANESBENCH_MAX_MEMORY_PER_GPU
  ANESBENCH_BATCH_SIZE
  ANESBENCH_OUTPUT_FILE

Model protocol is selected automatically from config.json. Native Transformers
VLMs, LLaVA-OneVision2, Fleming-VL, and custom-code InternVL chat models are
supported. Use --model-adapter only to override automatic selection.
EOF
}

require_option_value() {
  if (( $# < 2 )); then
    echo "Missing value for $1" >&2
    usage >&2
    exit 2
  fi
}

while (( $# > 0 )); do
  case "$1" in
    --model-path)
      require_option_value "$@"
      MODEL_PATHS+=("$2")
      shift 2
      ;;
    --model-path=*)
      MODEL_PATHS+=("${1#*=}")
      shift
      ;;
    --model-name)
      require_option_value "$@"
      MODEL_NAME="$2"
      shift 2
      ;;
    --model-name=*)
      MODEL_NAME="${1#*=}"
      shift
      ;;
    --language)
      require_option_value "$@"
      LANGUAGE="$2"
      shift 2
      ;;
    --language=*)
      LANGUAGE="${1#*=}"
      shift
      ;;
    --gpus)
      require_option_value "$@"
      GPU_IDS="$2"
      shift 2
      ;;
    --gpus=*)
      GPU_IDS="${1#*=}"
      shift
      ;;
    --device-map)
      require_option_value "$@"
      DEVICE_MAP="$2"
      shift 2
      ;;
    --device-map=*)
      DEVICE_MAP="${1#*=}"
      shift
      ;;
    --max-memory-per-gpu)
      require_option_value "$@"
      MAX_MEMORY_PER_GPU="$2"
      shift 2
      ;;
    --max-memory-per-gpu=*)
      MAX_MEMORY_PER_GPU="${1#*=}"
      shift
      ;;
    --batch-size)
      require_option_value "$@"
      BATCH_SIZE="$2"
      shift 2
      ;;
    --batch-size=*)
      BATCH_SIZE="${1#*=}"
      shift
      ;;
    --output)
      require_option_value "$@"
      OUTPUT_FILE="$2"
      shift 2
      ;;
    --output=*)
      OUTPUT_FILE="${1#*=}"
      shift
      ;;
    --data)
      require_option_value "$@"
      DATA_FILE="$2"
      shift 2
      ;;
    --data=*)
      DATA_FILE="${1#*=}"
      shift
      ;;
    --python)
      require_option_value "$@"
      PYTHON_BIN="$2"
      shift 2
      ;;
    --python=*)
      PYTHON_BIN="${1#*=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if (( ${#MODEL_PATHS[@]} == 0 )); then
  MODEL_PATHS+=("${ENV_MODEL_PATH}")
fi
if [[ -z "${GPU_IDS}" ]]; then
  echo "--gpus must not be empty." >&2
  exit 2
fi
if [[ -z "${DEVICE_MAP}" ]]; then
  echo "--device-map must not be empty." >&2
  exit 2
fi
if [[ ! "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--batch-size must be a positive integer, got: ${BATCH_SIZE}" >&2
  exit 2
fi
if [[ -n "${MAX_MEMORY_PER_GPU}" && ! "${MAX_MEMORY_PER_GPU}" =~ ^[1-9][0-9]*(\.[0-9]+)?(GiB|MiB|GB|MB)$ ]]; then
  echo "--max-memory-per-gpu must look like 70GiB or 24000MiB, got: ${MAX_MEMORY_PER_GPU}" >&2
  exit 2
fi
if [[ "${LANGUAGE}" != "zh" && "${LANGUAGE}" != "en" && "${LANGUAGE}" != "both" ]]; then
  echo "--language must be zh, en, or both, got: ${LANGUAGE}" >&2
  exit 2
fi
if (( ${#MODEL_PATHS[@]} > 1 )) && [[ -n "${MODEL_NAME}" || -n "${OUTPUT_FILE}" ]]; then
  echo "--model-name and --output cannot be shared by multiple models." >&2
  echo "Batch outputs are automatically written to outputs/<model-basename>/tee.jsonl." >&2
  exit 2
fi
if [[ "${LANGUAGE}" == "both" && -n "${DATA_FILE}" ]]; then
  echo "--data and ANESBENCH_TEE_DATA_FILE cannot be used with --language both." >&2
  echo "Both mode uses the frozen language-specific datasets automatically." >&2
  exit 2
fi
if [[ "${LANGUAGE}" == "both" && -n "${OUTPUT_FILE}" ]]; then
  echo "--output and ANESBENCH_OUTPUT_FILE cannot be used with --language both." >&2
  echo "Both mode writes tee-en.jsonl and tee.jsonl automatically." >&2
  exit 2
fi
if [[ -n "${DATA_FILE}" && "${DATA_FILE}" != /* ]]; then
  DATA_FILE="${PROJECT_ROOT}/${DATA_FILE}"
fi
if [[ "${LANGUAGE}" == "both" ]]; then
  RUN_LANGUAGES=(en zh)
else
  RUN_LANGUAGES=("${LANGUAGE}")
fi

if ! command -v -- "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python is not available: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ -n "${DATA_FILE}" && ! -f "${DATA_FILE}" ]]; then
  echo "TEE dataset does not exist: ${DATA_FILE}" >&2
  exit 1
fi
for run_language in "${RUN_LANGUAGES[@]}"; do
  if [[ -n "${DATA_FILE}" ]]; then
    checked_data_file="${DATA_FILE}"
  else
    checked_data_file="${PROJECT_ROOT}/data/private/Visual_Perception_TEE_${run_language}.jsonl"
  fi
  if [[ ! -f "${checked_data_file}" ]]; then
    echo "TEE dataset does not exist: ${checked_data_file}" >&2
    exit 1
  fi
done
for index in "${!MODEL_PATHS[@]}"; do
  model_path="${MODEL_PATHS[index]%/}"
  if [[ "${model_path}" != /* ]]; then
    model_path="${PROJECT_ROOT}/${model_path}"
  fi
  if [[ ! -d "${model_path}" ]]; then
    echo "Model directory does not exist: ${model_path}" >&2
    exit 1
  fi
  MODEL_PATHS[index]="${model_path}"
done

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
INFERENCE_RESOURCE_ARGS=(
  --device-map "${DEVICE_MAP}"
  --batch-size "${BATCH_SIZE}"
)
if [[ -n "${MAX_MEMORY_PER_GPU}" ]]; then
  INFERENCE_RESOURCE_ARGS+=(--max-memory-per-gpu "${MAX_MEMORY_PER_GPU}")
fi

cd -- "${PROJECT_ROOT}"
total_models="${#MODEL_PATHS[@]}"
for index in "${!MODEL_PATHS[@]}"; do
  model_path="${MODEL_PATHS[index]}"
  if (( total_models == 1 )) && [[ -n "${MODEL_NAME}" ]]; then
    model_name="${MODEL_NAME}"
  else
    model_name="$(basename -- "${model_path}")"
  fi
  if [[ ! "${model_name}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "Invalid model name for output directory: ${model_name}" >&2
    echo "Use --model-name with letters, numbers, dots, underscores, or hyphens." >&2
    exit 2
  fi

  echo
  echo "[$((index + 1))/${total_models}] Starting model: ${model_path}"
  echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
  echo "Device map: ${DEVICE_MAP}"
  echo "Batch size: ${BATCH_SIZE}"
  echo "Max memory per GPU: ${MAX_MEMORY_PER_GPU:-automatic}"

  total_languages="${#RUN_LANGUAGES[@]}"
  for language_index in "${!RUN_LANGUAGES[@]}"; do
    run_language="${RUN_LANGUAGES[language_index]}"
    if [[ -n "${DATA_FILE}" ]]; then
      run_data_file="${DATA_FILE}"
    else
      run_data_file="${PROJECT_ROOT}/data/private/Visual_Perception_TEE_${run_language}.jsonl"
    fi

    if (( total_models == 1 )) && [[ -n "${OUTPUT_FILE}" ]]; then
      model_output="${OUTPUT_FILE}"
      if [[ "${model_output}" != /* ]]; then
        model_output="${PROJECT_ROOT}/${model_output}"
      fi
    else
      if [[ "${run_language}" == "zh" ]]; then
        output_name="tee.jsonl"
      else
        output_name="tee-${run_language}.jsonl"
      fi
      model_output="${PROJECT_ROOT}/outputs/${model_name}/${output_name}"
    fi

    echo "[$((index + 1))/${total_models}][$((language_index + 1))/${total_languages}] Starting language: ${run_language}"
    echo "Data: ${run_data_file}"
    echo "Output: ${model_output}"

    "${PYTHON_BIN}" "${PROJECT_ROOT}/benchmarks/perception/inference/run.py" \
      "${run_data_file}" \
      --language "${run_language}" \
      --model-path "${model_path}" \
      --output "${model_output}" \
      "${INFERENCE_RESOURCE_ARGS[@]}" \
      ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

    echo "[$((index + 1))/${total_models}][$((language_index + 1))/${total_languages}] Completed language: ${run_language}"
  done
  echo "[$((index + 1))/${total_models}] Completed: ${model_name}"
done

echo
echo "All ${total_models} model run(s) completed for language mode: ${LANGUAGE}."
