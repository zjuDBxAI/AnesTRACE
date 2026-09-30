set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${ANESBENCH_PYTHON_BIN:-python3}"
cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" \
    benchmarks/single_point/inference/run_extended.py \
    --language en \
    --api-provider gemini \
    --api-key-env GEMINI_API_KEY \
    --max-workers 4
