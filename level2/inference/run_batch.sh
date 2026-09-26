#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

DATA_ROOT="${ANESTRACE_DATA_DIR:-${PROJECT_ROOT}/data}"
DEFAULT_DATA_FILE_ZH="${DATA_ROOT}/Level_two_B5_v2_zh.jsonl"
DEFAULT_DATA_FILE_EN="${DATA_ROOT}/Level_two_B5_v2_en.jsonl"

PYTHON_BIN="${ANESTRACE_PYTHON_BIN:-python3}"
DATA_FILE="${ANESTRACE_DATA_FILE:-}"
GPU_IDS="${ANESTRACE_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
OUTPUT_FILE="${ANESTRACE_OUTPUT_FILE:-}"
LANGUAGE="${ANESTRACE_LANGUAGE:-zh}"
MODEL_NAME=""
MODEL_PATHS=()
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  scripts/run_batch.sh [wrapper options] [run_inference.py options]

Wrapper options:
  --model-path PATH   Model directory; repeat to run models sequentially.
  --model-name NAME   Output directory name; valid only for a single model.
  --gpus IDS          CUDA_VISIBLE_DEVICES value, for example 0 or 0,1.
  --output PATH       Prediction JSONL; valid only for a single model.
  --data PATH         AnesTRACE B5 JSONL; valid only for one language.
  --language LANG     zh, en, or both (default: zh).
  --python PATH       Python interpreter containing inference dependencies.
  -h, --help          Show this help.
  --                  Pass all following arguments to run_inference.py.

Examples:
  # Run one local model on the Chinese dataset.
  scripts/run_batch.sh --model-path /path/to/model

  # Run the test split only.
  scripts/run_batch.sh --model-path /path/to/model -- --split test

  # Run multiple models sequentially in English.
  scripts/run_batch.sh --model-path /path/to/model-a \
    --model-path /path/to/model-b --language en

  # Run English then Chinese for each model.
  scripts/run_batch.sh --model-path /path/to/model --language both

  # One-record smoke test with one model and an explicit output.
  scripts/run_batch.sh \
    --model-path /path/to/model \
    --output /tmp/anestrace-smoke.jsonl \
    -- --limit 1 --max-new-tokens 3072 --overwrite

  # Generate four cases at once. Start at 1 and increase after a smoke test.
  scripts/run_batch.sh --model-path /path/to/model -- --batch-size 4

  # Opt into Qwen thinking mode.
  scripts/run_batch.sh --model-path /path/to/model -- --enable-thinking

Environment alternatives:
  ANESTRACE_PYTHON_BIN
  ANESTRACE_DATA_DIR
  ANESTRACE_DATA_FILE
  ANESTRACE_CUDA_VISIBLE_DEVICES
  ANESTRACE_OUTPUT_FILE
  ANESTRACE_LANGUAGE

The runner sends only patient_information as case-specific model input. It
never reads or sends waveform/trend images, answers, or review metadata.
Failed model runs are reported and do not block later runs. After all runs,
the script returns the first failure status; interrupts still stop immediately.
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
    --gpus)
      require_option_value "$@"
      GPU_IDS="$2"
      shift 2
      ;;
    --gpus=*)
      GPU_IDS="${1#*=}"
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
    --language)
      require_option_value "$@"
      LANGUAGE="$2"
      shift 2
      ;;
    --language=*)
      LANGUAGE="${1#*=}"
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

if [[ "${LANGUAGE}" != "zh" && "${LANGUAGE}" != "en" && "${LANGUAGE}" != "both" ]]; then
  echo "--language must be zh, en, or both." >&2
  exit 2
fi
if [[ "${LANGUAGE}" == "both" && -n "${DATA_FILE}" ]]; then
  echo "--data cannot be used with --language both." >&2
  exit 2
fi
if [[ "${LANGUAGE}" == "both" && -n "${OUTPUT_FILE}" ]]; then
  echo "--output cannot be used with --language both." >&2
  exit 2
fi
if (( ${#MODEL_PATHS[@]} == 0 )); then
  echo "At least one --model-path is required." >&2
  exit 2
fi
if (( ${#MODEL_PATHS[@]} > 1 )) && [[ -n "${MODEL_NAME}" || -n "${OUTPUT_FILE}" ]]; then
  echo "--model-name and --output are valid only when running one model." >&2
  exit 2
fi
if [[ -n "${DATA_FILE}" && "${DATA_FILE}" != /* ]]; then
  DATA_FILE="${PROJECT_ROOT}/${DATA_FILE}"
fi
if [[ "${PYTHON_BIN}" == */* ]]; then
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python is not executable: ${PYTHON_BIN}" >&2
    exit 1
  fi
elif ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable was not found on PATH: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ -n "${DATA_FILE}" && ! -f "${DATA_FILE}" ]]; then
  echo "AnesTRACE dataset does not exist: ${DATA_FILE}" >&2
  exit 1
fi
if [[ -z "${DATA_FILE}" ]]; then
  if [[ "${LANGUAGE}" != "en" && ! -f "${DEFAULT_DATA_FILE_ZH}" ]]; then
    echo "AnesTRACE Chinese dataset does not exist: ${DEFAULT_DATA_FILE_ZH}" >&2
    exit 1
  fi
  if [[ "${LANGUAGE}" != "zh" && ! -f "${DEFAULT_DATA_FILE_EN}" ]]; then
    echo "AnesTRACE English dataset does not exist: ${DEFAULT_DATA_FILE_EN}" >&2
    exit 1
  fi
fi
if [[ ! -f "${PROJECT_ROOT}/run_inference.py" ]]; then
  echo "AnesTRACE runner does not exist: ${PROJECT_ROOT}/run_inference.py" >&2
  exit 1
fi

for index in "${!MODEL_PATHS[@]}"; do
  model_path="${MODEL_PATHS[index]%/}"
  if [[ "${model_path}" != /* ]]; then
    model_path="${PROJECT_ROOT}/${model_path}"
  fi
  if [[ ! -d "${model_path}" ]]; then
    echo "Model directory does not exist: ${model_path}" >&2
    exit 1
  fi
  if [[ ! -f "${model_path}/config.json" ]]; then
    echo "Model config does not exist: ${model_path}/config.json" >&2
    exit 1
  fi
  MODEL_PATHS[index]="${model_path}"
done

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
cd -- "${PROJECT_ROOT}"

total_models="${#MODEL_PATHS[@]}"
if [[ "${LANGUAGE}" == "both" ]]; then
  RUN_LANGUAGES=("en" "zh")
else
  RUN_LANGUAGES=("${LANGUAGE}")
fi
total_runs=$((total_models * ${#RUN_LANGUAGES[@]}))
run_index=0
overall_status=0
FAILED_RUNS=()
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

  for run_language in "${RUN_LANGUAGES[@]}"; do
    run_index=$((run_index + 1))
    if [[ -n "${DATA_FILE}" ]]; then
      run_data="${DATA_FILE}"
    elif [[ "${run_language}" == "en" ]]; then
      run_data="${DEFAULT_DATA_FILE_EN}"
    else
      run_data="${DEFAULT_DATA_FILE_ZH}"
    fi

    if (( total_models == 1 )) && [[ -n "${OUTPUT_FILE}" ]]; then
      model_output="${OUTPUT_FILE}"
      if [[ "${model_output}" != /* ]]; then
        model_output="${PROJECT_ROOT}/${model_output}"
      fi
    elif [[ "${run_language}" == "en" ]]; then
      model_output="${PROJECT_ROOT}/outputs/${model_name}/b5-text-only-en.jsonl"
    else
      model_output="${PROJECT_ROOT}/outputs/${model_name}/b5-text-only.jsonl"
    fi

    echo
    echo "[${run_index}/${total_runs}] Starting AnesTRACE model: ${model_path}"
    echo "Language: ${run_language}"
    echo "Data: ${run_data}"
    echo "Output: ${model_output}"
    echo "GPUs: ${CUDA_VISIBLE_DEVICES}"

    if "${PYTHON_BIN}" "${PROJECT_ROOT}/run_inference.py" \
      "${run_data}" \
      --language "${run_language}" \
      --model-path "${model_path}" \
      --output "${model_output}" \
      "${EXTRA_ARGS[@]}"; then
      echo "Completed AnesTRACE run: ${model_name} [${run_language}]"
    else
      run_status=$?
      if (( run_status == 130 || run_status == 143 )); then
        echo "AnesTRACE run interrupted (exit ${run_status}); stopping." >&2
        exit "${run_status}"
      fi
      if (( overall_status == 0 )); then
        overall_status=${run_status}
      fi
      FAILED_RUNS+=("${model_name} [${run_language}] exit=${run_status}")
      echo "AnesTRACE run failed (exit ${run_status}); continuing with the next run." >&2
    fi
  done
done

if (( ${#FAILED_RUNS[@]} > 0 )); then
  echo >&2
  echo "Completed with ${#FAILED_RUNS[@]} failed run(s):" >&2
  for failed_run in "${FAILED_RUNS[@]}"; do
    echo "  - ${failed_run}" >&2
  done
  exit "${overall_status}"
fi

echo
echo "All ${total_runs} AnesTRACE run(s) completed successfully."
