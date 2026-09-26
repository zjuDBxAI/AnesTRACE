#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_PYTHON_BIN="python3"

PYTHON_BIN="${ANESBENCH_PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}"
OUTPUTS_ROOT="${ANESBENCH_OUTPUTS_ROOT:-${PROJECT_ROOT}/outputs}"
LANGUAGE="${ANESBENCH_LANGUAGE:-both}"
MODE="${ANESBENCH_EVAL_MODE:-clinical}"
ENGLISH_FACTS_VERSION="v1"
BATCH_SUMMARY=""
DISCOVER_ALL=false
SKIP_MISSING=false
MIGRATE_LEGACY=false
OFFICIAL_ONLY=false
MODEL_DIRS=()

usage() {
  cat <<'EOF'
Usage:
  level1/evaluation/run.sh --model-dir DIR [--model-dir DIR ...] [options]
  level1/evaluation/run.sh --all [options]

Options:
  --model-dir DIR       Model output directory; repeat for sequential evaluation.
                        A bare name resolves under outputs/.
  --all                 Discover direct child directories under outputs/.
  --language LANG       zh, en, or both (default: both).
  --mode MODE           clinical, legacy, or both (default: clinical).
  --english-facts-version VERSION
                        v1 (historical default) or en-v2 (corrected English rules).
  --migrate-legacy      Migrate old JSON responses before ClinicalFact evaluation.
                        Migrated scores are compatibility/provisional results.
  --skip-missing        Skip missing language predictions instead of failing.
  --official-only       Pass --official-only to the evaluator.
  --outputs-root DIR    Root used by --all and bare model names.
  --batch-summary PATH  Aggregate TSV output path.
  --python PATH         Python interpreter with evaluation dependencies.
  -h, --help            Show this help.

Prediction names:
  Chinese: tee.jsonl
  English: tee-en.jsonl

Examples:
  level1/evaluation/run.sh \
    --model-dir outputs/Qwen3.5-9B \
    --model-dir outputs/Qwen3.5-27B \
    --language both

  level1/evaluation/run.sh --all --language zh --skip-missing

  level1/evaluation/run.sh \
    --model-dir outputs/Qwen3-VL-8B-Instruct \
    --language both --mode both --migrate-legacy
EOF
}

require_option_value() {
  if (( $# < 2 )); then
    echo "Missing value for $1" >&2
    exit 2
  fi
}

while (( $# > 0 )); do
  case "$1" in
    --model-dir|--model)
      require_option_value "$@"
      MODEL_DIRS+=("$2")
      shift 2
      ;;
    --model-dir=*|--model=*)
      MODEL_DIRS+=("${1#*=}")
      shift
      ;;
    --all)
      DISCOVER_ALL=true
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
    --mode)
      require_option_value "$@"
      MODE="$2"
      shift 2
      ;;
    --mode=*)
      MODE="${1#*=}"
      shift
      ;;
    --migrate-legacy)
      MIGRATE_LEGACY=true
      shift
      ;;
    --english-facts-version)
      require_option_value "$@"
      ENGLISH_FACTS_VERSION="$2"
      shift 2
      ;;
    --english-facts-version=*)
      ENGLISH_FACTS_VERSION="${1#*=}"
      shift
      ;;
    --skip-missing)
      SKIP_MISSING=true
      shift
      ;;
    --official-only)
      OFFICIAL_ONLY=true
      shift
      ;;
    --outputs-root)
      require_option_value "$@"
      OUTPUTS_ROOT="$2"
      shift 2
      ;;
    --outputs-root=*)
      OUTPUTS_ROOT="${1#*=}"
      shift
      ;;
    --batch-summary)
      require_option_value "$@"
      BATCH_SUMMARY="$2"
      shift 2
      ;;
    --batch-summary=*)
      BATCH_SUMMARY="${1#*=}"
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
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${ENGLISH_FACTS_VERSION}" != "v1" && "${ENGLISH_FACTS_VERSION}" != "en-v2" ]]; then
  echo "--english-facts-version must be v1 or en-v2." >&2
  exit 2
fi

clinical_facts_path() {
  local selected_language="$1"
  local selected_version="v1"
  if [[ "${selected_language}" == "en" ]]; then
    selected_version="${ENGLISH_FACTS_VERSION}"
  fi
  printf '%s/level_one/TEE/clinical_facts/%s/frozen_%s.jsonl' "${PROJECT_ROOT}" "${selected_version}" "${selected_language}"
}

if [[ "${LANGUAGE}" != "zh" && "${LANGUAGE}" != "en" && "${LANGUAGE}" != "both" ]]; then
  echo "--language must be zh, en, or both, got: ${LANGUAGE}" >&2
  exit 2
fi
if [[ "${MODE}" != "clinical" && "${MODE}" != "legacy" && "${MODE}" != "both" ]]; then
  echo "--mode must be clinical, legacy, or both, got: ${MODE}" >&2
  exit 2
fi
if [[ "${DISCOVER_ALL}" == true && ${#MODEL_DIRS[@]} -gt 0 ]]; then
  echo "--all cannot be combined with --model-dir." >&2
  exit 2
fi
if [[ "${DISCOVER_ALL}" == false && ${#MODEL_DIRS[@]} -eq 0 ]]; then
  echo "Provide at least one --model-dir or use --all." >&2
  usage >&2
  exit 2
fi
if ! command -v -- "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python is not available: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ "${OUTPUTS_ROOT}" != /* ]]; then
  OUTPUTS_ROOT="${PROJECT_ROOT}/${OUTPUTS_ROOT}"
fi
if [[ ! -d "${OUTPUTS_ROOT}" ]]; then
  echo "Outputs root does not exist: ${OUTPUTS_ROOT}" >&2
  exit 1
fi
if [[ "${DISCOVER_ALL}" == true ]]; then
  mapfile -d '' MODEL_DIRS < <(
    find "${OUTPUTS_ROOT}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z
  )
  if (( ${#MODEL_DIRS[@]} == 0 )); then
    echo "No model directories found under: ${OUTPUTS_ROOT}" >&2
    exit 1
  fi
fi

for index in "${!MODEL_DIRS[@]}"; do
  model_dir="${MODEL_DIRS[index]%/}"
  if [[ "${model_dir}" != /* ]]; then
    if [[ "${model_dir}" == */* ]]; then
      model_dir="${PROJECT_ROOT}/${model_dir}"
    else
      model_dir="${OUTPUTS_ROOT}/${model_dir}"
    fi
  fi
  if [[ ! -d "${model_dir}" ]]; then
    echo "Model output directory does not exist: ${model_dir}" >&2
    exit 1
  fi
  MODEL_DIRS[index]="${model_dir}"
done

if [[ "${LANGUAGE}" == "both" ]]; then
  RUN_LANGUAGES=(en zh)
else
  RUN_LANGUAGES=("${LANGUAGE}")
fi
if [[ "${MODE}" == "both" ]]; then
  RUN_MODES=(legacy clinical)
else
  RUN_MODES=("${MODE}")
fi
if [[ -z "${BATCH_SUMMARY}" ]]; then
  BATCH_SUMMARY="${OUTPUTS_ROOT}/tee.${MODE}.${LANGUAGE}.batch-summary.tsv"
  if [[ "${ENGLISH_FACTS_VERSION}" != "v1" && "${LANGUAGE}" != "zh" && "${MODE}" != "legacy" ]]; then
    BATCH_SUMMARY="${OUTPUTS_ROOT}/tee.${MODE}.${LANGUAGE}.${ENGLISH_FACTS_VERSION}.batch-summary.tsv"
  fi
elif [[ "${BATCH_SUMMARY}" != /* ]]; then
  BATCH_SUMMARY="${PROJECT_ROOT}/${BATCH_SUMMARY}"
fi

for run_language in "${RUN_LANGUAGES[@]}"; do
  gold="${PROJECT_ROOT}/level_one/TEE/Visual Perception_TEE_${run_language}.jsonl"
  facts="$(clinical_facts_path "${run_language}")"
  if [[ ! -f "${gold}" || ! -f "${facts}" ]]; then
    echo "Missing ${run_language} Ground Truth or ClinicalFact sidecar." >&2
    exit 1
  fi
done

tmp_summary="$(mktemp "${TMPDIR:-/tmp}/anesbench-tee-eval.XXXXXX")"
cleanup() {
  rm -f -- "${tmp_summary}"
}
trap cleanup EXIT
printf 'model	language	mode	macro_score	category_macro_score	foundational_perception	anatomical_grounding	clinical_assessment	item_macro_score	evaluated_items	missing_predictions	invalid_predictions	duplicate_prediction_records	predictions	summary
' >"${tmp_summary}"

json_scalar() {
  local path="$1"
  local key="$2"
  awk -v wanted="\"${key}\"" '
    index($0, wanted) {
      value = $0
      sub(/^[^:]*:[[:space:]]*/, "", value)
      sub(/,[[:space:]]*$/, "", value)
      gsub(/^"|"$/, "", value)
      print value
      exit
    }
  ' "${path}"
}

json_category_score() {
  local path="$1"
  local category="$2"
  awk -v wanted="\"${category}\"" '
    index($0, wanted) {
      value = $0
      if (value ~ /"score"[[:space:]]*:/) {
        sub(/^.*"score"[[:space:]]*:[[:space:]]*/, "", value)
        sub(/[},].*$/, "", value)
        print value
        exit
      }
      in_category = 1
      next
    }
    in_category && /"score"[[:space:]]*:/ {
      value = $0
      sub(/^[^:]*:[[:space:]]*/, "", value)
      sub(/,[[:space:]]*$/, "", value)
      print value
      exit
    }
  ' "${path}"
}

protocol_counts() {
  local path="$1"
  awk '
    {
      qa_id = $0
      if (qa_id !~ /"qa_id"[[:space:]]*:[[:space:]]*"/) {
        next
      }
      sub(/^.*"qa_id"[[:space:]]*:[[:space:]]*"/, "", qa_id)
      sub(/".*$/, "", qa_id)
      is_open[qa_id] = $0 ~ /"task_name"[[:space:]]*:[[:space:]]*"(functional_assessment|abnormality_detection)"/
      is_controlled[qa_id] = $0 ~ /"response_contract"[[:space:]]*:[[:space:]]*"controlled_natural_language_v1"/
    }
    END {
      for (qa_id in is_open) {
        if (is_open[qa_id]) {
          open += 1
          controlled += is_controlled[qa_id]
        }
      }
      print open + 0, controlled + 0
    }
  ' "${path}"
}

evaluations=0
skipped=0
total_models="${#MODEL_DIRS[@]}"
cd -- "${PROJECT_ROOT}"

for model_index in "${!MODEL_DIRS[@]}"; do
  model_dir="${MODEL_DIRS[model_index]}"
  model_name="$(basename -- "${model_dir}")"
  echo
  echo "[$((model_index + 1))/${total_models}] Model: ${model_name}"

  for run_language in "${RUN_LANGUAGES[@]}"; do
    if [[ "${run_language}" == "zh" ]]; then
      prediction_stem="tee"
    else
      prediction_stem="tee-${run_language}"
    fi
    prediction="${model_dir}/${prediction_stem}.jsonl"
    if [[ ! -f "${prediction}" ]]; then
      if [[ "${SKIP_MISSING}" == true ]]; then
        echo "  [skip] ${run_language}: missing ${prediction}"
        skipped=$((skipped + 1))
        continue
      fi
      echo "Missing prediction file: ${prediction}" >&2
      echo "Use --skip-missing to continue past unavailable languages." >&2
      exit 1
    fi

    gold="${PROJECT_ROOT}/level_one/TEE/Visual Perception_TEE_${run_language}.jsonl"
    facts="$(clinical_facts_path "${run_language}")"
    read -r open_count controlled_count < <(protocol_counts "${prediction}")

    for run_mode in "${RUN_MODES[@]}"; do
      eval_prediction="${prediction}"
      report_stem="${model_dir}/${prediction_stem}.${run_mode}"

      if [[ "${run_mode}" == "clinical" ]]; then
        report_stem="${model_dir}/${prediction_stem}.clinical-fact"
        if [[ "${run_language}" == "en" && "${ENGLISH_FACTS_VERSION}" != "v1" ]]; then
          report_stem="${report_stem}.${ENGLISH_FACTS_VERSION}"
        fi
        if (( open_count == 0 )); then
          echo "No open-task records in: ${prediction}" >&2
          exit 1
        elif (( controlled_count == open_count )); then
          :
        elif (( controlled_count > 0 )); then
          echo "Mixed old/new contracts in ${prediction}: ${controlled_count}/${open_count} controlled." >&2
          exit 1
        elif [[ "${MIGRATE_LEGACY}" == true ]]; then
          migrated="${model_dir}/${prediction_stem}.controlled-v1.migrated.jsonl"
          migration_audit="${model_dir}/${prediction_stem}.controlled-v1.migration.audit.jsonl"
          migration_summary="${model_dir}/${prediction_stem}.controlled-v1.migration.summary.json"
          if [[ -f "${migrated}" ]]; then
            if [[ "${prediction}" -nt "${migrated}" ]]; then
              echo "Existing migration is older than source: ${migrated}" >&2
              exit 1
            fi
          elif [[ -e "${migration_audit}" || -e "${migration_summary}" ]]; then
            echo "Partial migration artifacts exist for ${prediction_stem}." >&2
            exit 1
          else
            echo "  [migrate] ${run_language}: legacy JSON -> controlled v1"
            "${PYTHON_BIN}" scripts/migrate_tee_legacy_predictions.py \
              --input "${prediction}" \
              --output "${migrated}" \
              --language "${run_language}" \
              --audit-output "${migration_audit}" \
              --summary-output "${migration_summary}"
          fi
          eval_prediction="${migrated}"
        else
          echo "ClinicalFact evaluation requires controlled_natural_language_v1: ${prediction}" >&2
          echo "Rerun inference or pass --migrate-legacy for a provisional score." >&2
          exit 1
        fi
      fi

      audit_output="${report_stem}.eval.jsonl"
      summary_output="${report_stem}.eval.summary.json"
      command=(
        "${PYTHON_BIN}" "${PROJECT_ROOT}/level1/evaluation/evaluate.py"
        --gold "${gold}"
        --predictions "${eval_prediction}"
        --profile tee
        --language "${run_language}"
        --output "${audit_output}"
        --summary-output "${summary_output}"
      )
      if [[ "${run_mode}" == "clinical" ]]; then
        command+=(--clinical-facts-gold "${facts}")
      fi
      if [[ "${OFFICIAL_ONLY}" == true ]]; then
        command+=(--official-only)
      fi

      echo "  [evaluate] language=${run_language} mode=${run_mode}"
      "${command[@]}"
      macro_score="$(json_scalar "${summary_output}" macro_score)"
      category_macro_score="$(json_scalar "${summary_output}" category_macro_score)"
      foundational_score="$(json_category_score "${summary_output}" foundational_perception)"
      grounding_score="$(json_category_score "${summary_output}" anatomical_grounding)"
      clinical_score="$(json_category_score "${summary_output}" clinical_assessment)"
      item_macro_score="$(json_scalar "${summary_output}" item_macro_score)"
      evaluated_items="$(json_scalar "${summary_output}" evaluated_items)"
      invalid_predictions="$(json_scalar "${summary_output}" invalid_predictions)"
      missing_predictions="$(json_scalar "${summary_output}" missing_predictions)"
      duplicate_predictions="$(json_scalar "${summary_output}" duplicate_prediction_records)"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${model_name}" "${run_language}" "${run_mode}" \
        "${macro_score}" "${category_macro_score}" "${foundational_score}" \
        "${grounding_score}" "${clinical_score}" "${item_macro_score}" "${evaluated_items}" \
        "${missing_predictions}" "${invalid_predictions}" "${duplicate_predictions}" \
        "${eval_prediction}" "${summary_output}" >>"${tmp_summary}"
      evaluations=$((evaluations + 1))
    done
  done
done

mkdir -p -- "$(dirname -- "${BATCH_SUMMARY}")"
mv -f -- "${tmp_summary}" "${BATCH_SUMMARY}"
trap - EXIT

echo
echo "Batch evaluation summary:"
if command -v column >/dev/null 2>&1; then
  column -t -s $'\t' "${BATCH_SUMMARY}"
else
  cat "${BATCH_SUMMARY}"
fi
echo
echo "Completed ${evaluations} evaluation(s); skipped ${skipped} missing prediction file(s)."
echo "Summary TSV: ${BATCH_SUMMARY}"
