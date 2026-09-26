set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${ANESBENCH_PYTHON_BIN:-python3}"
cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" \
    level2/inference/run_extended.py \
    --language en \
    --api-provider gemini \
    --api-key-env GEMINI_API_KEY \
    --max-workers 4
