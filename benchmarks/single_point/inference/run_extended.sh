#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

DEFAULT_PYTHON_BIN="python3"
MODEL_ROOT="${ANESTRACE_MODEL_ROOT:-${PROJECT_ROOT}/checkpoints}"
DEFAULT_DATA_FILE_ZH="${PROJECT_ROOT}/data/private/Level_two_B5_v2_zh.jsonl"
DEFAULT_DATA_FILE_EN="${PROJECT_ROOT}/data/private/Level_two_B5_v3_en_evidence.jsonl"
DEFAULT_MODEL_PATHS=(
  "${MODEL_ROOT}/Qwen/Qwen3-8B"
  "${MODEL_ROOT}/Qwen/Qwen3.5-9B"
  "${MODEL_ROOT}/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
  "${MODEL_ROOT}/FreedomIntelligence/HuatuoGPT-3-8B"
  "${MODEL_ROOT}/FreedomIntelligence/HuatuoGPT-3-32B"
  "${MODEL_ROOT}/MiliLab/Morpheus-7B"
  "${MODEL_ROOT}/MiliLab/Morpheus-32B"
  "${MODEL_ROOT}/openai/gpt-oss-20b"
  "${MODEL_ROOT}/Qwen/Qwen3.5-27B"
  "${MODEL_ROOT}/Qwen/Qwen3.8-27B"
  "${MODEL_ROOT}/google/medgemma-27b-text-it"
  "${MODEL_ROOT}/IQuestLab/Fleming-R1-7B"
  "${MODEL_ROOT}/IQuestLab/Fleming-R1-32B"
)

PYTHON_BIN="${ANESBENCH_L2_PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}"
DATA_FILE="${ANESBENCH_L2_DATA_FILE:-}"
GPU_IDS="${ANESBENCH_L2_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
OUTPUT_FILE="${ANESBENCH_L2_OUTPUT_FILE:-}"
LANGUAGE="${ANESBENCH_L2_LANGUAGE:-zh}"
MODEL_NAME=""
MODEL_PATHS=()
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  benchmarks/single_point/inference/run_extended.sh [wrapper options] [run_extended.py options]

Wrapper options:
  --model-path PATH   Model directory; repeat to run models sequentially.
  --model-name NAME   Output directory name; valid only for a single model.
  --gpus IDS          CUDA_VISIBLE_DEVICES value, for example 0 or 0,1.
  --output PATH       Prediction JSONL; valid only for a single model.
  --data PATH         Level Two B5 JSONL; valid only for one language.
  --language LANG     zh, en, or both (default: zh).
  --python PATH       Python interpreter containing inference dependencies.
  -h, --help          Show this help.
  --                  Pass all following arguments to run_level_two.py.

Examples:
  # Run all 500 records with all configured models, sequentially on GPU 0.
  benchmarks/single_point/inference/run_extended.sh

  # Run the test split only.
  benchmarks/single_point/inference/run_extended.sh -- --split test

  # Run every model in English.
  benchmarks/single_point/inference/run_extended.sh --language en

  # Run English then Chinese for each model.
  benchmarks/single_point/inference/run_extended.sh --language both

  # One-record smoke test with one model and an explicit output.
  benchmarks/single_point/inference/run_extended.sh \
    --model-path "${ANESTRACE_MODEL_ROOT:-${PROJECT_ROOT}/checkpoints}/Qwen/Qwen3-8B" \
    --output /tmp/qwen3-8b-l2-smoke.jsonl \
    -- --limit 1 --max-new-tokens 3072 --overwrite

  # Generate four cases at once. Start at 1 and increase after a smoke test.
  benchmarks/single_point/inference/run_extended.sh -- --batch-size 4

  # Opt into Qwen thinking mode.
  benchmarks/single_point/inference/run_extended.sh -- --enable-thinking

Environment alternatives:
  ANESBENCH_L2_PYTHON_BIN
  ANESBENCH_L2_DATA_FILE
  ANESBENCH_L2_CUDA_VISIBLE_DEVICES
  ANESBENCH_L2_OUTPUT_FILE
  ANESBENCH_L2_LANGUAGE

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

if (( ${#MODEL_PATHS[@]} == 0 )); then
  MODEL_PATHS=("${DEFAULT_MODEL_PATHS[@]}")
fi
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
if (( ${#MODEL_PATHS[@]} > 1 )) && [[ -n "${MODEL_NAME}" || -n "${OUTPUT_FILE}" ]]; then
  echo "--model-name and --output are valid only when running one model." >&2
  exit 2
fi
if [[ -n "${DATA_FILE}" && "${DATA_FILE}" != /* ]]; then
  DATA_FILE="${PROJECT_ROOT}/${DATA_FILE}"
fi
if ! command -v -- "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python is not available: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ -n "${DATA_FILE}" && ! -f "${DATA_FILE}" ]]; then
  echo "Level Two dataset does not exist: ${DATA_FILE}" >&2
  exit 1
fi
if [[ -z "${DATA_FILE}" ]]; then
  if [[ "${LANGUAGE}" != "en" && ! -f "${DEFAULT_DATA_FILE_ZH}" ]]; then
    echo "Level Two Chinese dataset does not exist: ${DEFAULT_DATA_FILE_ZH}" >&2
    exit 1
  fi
  if [[ "${LANGUAGE}" != "zh" && ! -f "${DEFAULT_DATA_FILE_EN}" ]]; then
    echo "Level Two English dataset does not exist: ${DEFAULT_DATA_FILE_EN}" >&2
    exit 1
  fi
fi
if [[ ! -f "${PROJECT_ROOT}/benchmarks/single_point/inference/run_extended.py" ]]; then
  echo "Level Two runner does not exist: ${PROJECT_ROOT}/benchmarks/single_point/inference/run_extended.py" >&2
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
      model_output="${PROJECT_ROOT}/outputs/level_two/${model_name}/level-two-b5-text-only-en.jsonl"
    else
      model_output="${PROJECT_ROOT}/outputs/level_two/${model_name}/level-two-b5-text-only.jsonl"
    fi

    echo
    echo "[${run_index}/${total_runs}] Starting Level Two model: ${model_path}"
    echo "Language: ${run_language}"
    echo "Data: ${run_data}"
    echo "Output: ${model_output}"
    echo "GPUs: ${CUDA_VISIBLE_DEVICES}"

    if "${PYTHON_BIN}" "${PROJECT_ROOT}/benchmarks/single_point/inference/run_extended.py" \
      "${run_data}" \
      --language "${run_language}" \
      --model-path "${model_path}" \
      --output "${model_output}" \
      ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}; then
      echo "Completed Level Two run: ${model_name} [${run_language}]"
    else
      run_status=$?
      if (( run_status == 130 || run_status == 143 )); then
        echo "Level Two run interrupted (exit ${run_status}); stopping." >&2
        exit "${run_status}"
      fi
      if (( overall_status == 0 )); then
        overall_status=${run_status}
      fi
      FAILED_RUNS+=("${model_name} [${run_language}] exit=${run_status}")
      echo "Level Two run failed (exit ${run_status}); continuing with the next run." >&2
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
echo "All ${total_runs} Level Two run(s) completed successfully."
