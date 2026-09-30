#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${ANESBENCH_PYTHON_BIN:-python3}"
cd "${PROJECT_ROOT}"

gold_path="${PROJECT_ROOT}/evaluation_data/gold/Level_two_B5_v3_en_evidence.jsonl"

for model_name in \
  Fleming-R1-7B
do
  prediction_path="outputs/level_two/${model_name}/level-two-b5-text-only-en.jsonl"
  output_path="outputs/level_two/${model_name}/level-two-b5-text-only-en.v3.local_judge.jsonl"
  summary_path="outputs/level_two/${model_name}/level-two-b5-text-only-en.v3.judge_summary.json"

  CUDA_VISIBLE_DEVICES=0 \
  "${PYTHON_BIN}" \
    benchmarks/single_point/evaluation/evaluate.py \
    --gold "${gold_path}" \
    --predictions "${prediction_path}" \
    --output "${output_path}" \
    --summary-output "${summary_path}" \
    --device-map auto \
    --device cuda:0 \
    --dtype bfloat16 \
    --batch-size 4 \
    --candidate-invalid-policy record_zero \
    --resume
done
