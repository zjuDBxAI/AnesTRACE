#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${ANESBENCH_L3_PYTHON_BIN:-python3}"
MODEL_ROOT="${ANESTRACE_MODEL_ROOT:-${PROJECT_ROOT}/checkpoints}"
GPU_IDS="${ANESBENCH_L3_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
LANGUAGE=zh
MODE=1
MODELS=()
# Keep model order identical to the Level 2 batch runner.
ALL_MODEL_PATHS=(
  "${MODEL_ROOT}/Qwen/Qwen3-8B"
  "${MODEL_ROOT}/Qwen/Qwen3.5-9B"
  # "${MODEL_ROOT}/zai-org/GLM-4.7-Flash"
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
EXTRA=()
DATA=""
OUTPUT=""
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/level_three_ablation"

usage() {
  echo 'Usage: bash benchmarks/multi_step/inference/run_ablation.sh [options] [-- Python options]'
  echo '  --language zh|en|both            Default: zh'
  echo '  --ablation-mode 1|2|3|all       Also accepts descriptive mode names'
  echo '  --model-path PATH|all           Repeat PATH, or all for the L2 model list'
  echo '  --gpus IDS --python PATH        GPU selection and interpreter'
  echo '  --input PATH                    Custom input; one language only'
  echo '  --output PATH                   Custom JSONL; one combination only'
  echo '  --output-root PATH              Root for all combination outputs'
  echo '  --                              Forward e.g. --limit 1 --dry-run'
}

while (( $# )); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA+=("$@"); break ;;
    --language|--ablation-mode|--model-path|--gpus|--python|--input|--output|--output-root)
      (( $# >= 2 )) || { echo "Missing value for $1" >&2; exit 2; }
      case "$1" in
        --language) LANGUAGE="$2" ;;
        --ablation-mode) MODE="$2" ;;
        --model-path) MODELS+=("$2") ;;
        --gpus) GPU_IDS="$2" ;;
        --python) PYTHON_BIN="$2" ;;
        --input) DATA="$2" ;;
        --output) OUTPUT="$2" ;;
        --output-root) OUTPUT_ROOT="$2" ;;
      esac
      shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

case "$LANGUAGE" in
  zh|en) LANGUAGES=("$LANGUAGE") ;;
  both) LANGUAGES=(zh en) ;;
  *) echo 'Invalid language' >&2; exit 2 ;;
esac
case "$MODE" in
  1|current_only) MODES=(current_only) ;;
  2|state_history) MODES=(state_history) ;;
  3|state_action_history) MODES=(state_action_history) ;;
  all) MODES=(current_only state_history state_action_history) ;;
  *) echo 'Invalid ablation mode' >&2; exit 2 ;;
esac
if (( ${#MODELS[@]} == 0 )); then
  MODELS=("${MODEL_ROOT}/Qwen/Qwen3-8B")
fi
for model in "${MODELS[@]}"; do
  if [[ "$model" == all ]]; then
    if (( ${#MODELS[@]} != 1 )); then
      echo '--model-path all must be used alone, without other --model-path options' >&2
      exit 2
    fi
    MODELS=("${ALL_MODEL_PATHS[@]}")
    break
  fi
done
if [[ -n "$DATA" && ${#LANGUAGES[@]} -ne 1 ]]; then
  echo '--input requires one language' >&2; exit 2
fi
if [[ -n "$OUTPUT" ]] && (( ${#MODELS[@]} * ${#LANGUAGES[@]} * ${#MODES[@]} != 1 )); then
  echo '--output requires one model/language/mode combination; use --output-root' >&2; exit 2
fi
# Prevent passthrough overrides that would overwrite another combination's output.
for arg in "${EXTRA[@]}"; do
  case "$arg" in
    --language|--language=*|--ablation-mode|--ablation-mode=*|--model-path|--model-path=*|--input|--input=*|--output|--output=*)
      echo "Pass wrapper option before --: $arg" >&2; exit 2 ;;
  esac
done
declare -A MODEL_NAMES=()
for model in "${MODELS[@]}"; do
  name="$(basename -- "$model")"
  if [[ -n "${MODEL_NAMES[$name]:-}" ]]; then
    echo "Duplicate model basename: $name; run separately with different --output-root" >&2; exit 2
  fi
  MODEL_NAMES[$name]=1
done
status=0
for model in "${MODELS[@]}"; do
  for language in "${LANGUAGES[@]}"; do
    for mode in "${MODES[@]}"; do
      destination="${OUTPUT:-${OUTPUT_ROOT}/$(basename -- "$model")/${language}/${mode}/predictions.jsonl}"
      args=(--language "$language" --ablation-mode "$mode" --model-path "$model" --output "$destination")
      if [[ -n "$DATA" ]]; then args+=(--input "$DATA"); fi
      echo "L3: $(basename -- "$model") / $language / $mode"
      if CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON_BIN" "${PROJECT_ROOT}/benchmarks/multi_step/inference/run_ablation.py" "${args[@]}" "${EXTRA[@]}"; then
        :
      else
        rc=$?
        if (( rc == 130 || rc == 143 )); then exit "$rc"; fi
        if (( status == 0 )); then status=$rc; fi
        echo "Combination failed (exit $rc): $language / $mode" >&2
      fi
    done
  done
done
exit "$status"
