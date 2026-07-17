#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_ENV_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${TEST_ENV_DIR}/.venv/bin/python"
OUTPUT_DIR="${TEST_ENV_DIR}/results/layer2_dynamic_events"
EMBEDDING_CACHE="${OUTPUT_DIR}/embedding_cache.sqlite3"
RUN_ID="${PLASMOD_TABLE6_RUN_ID:-table6_full_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${OUTPUT_DIR}/${RUN_ID}"
LOG_FILE="${RUN_DIR}/run.log"

if [[ ! -x "${PYTHON}" ]]; then
  echo "missing benchmark interpreter: ${PYTHON}" >&2
  exit 1
fi
if [[ ! -f "${EMBEDDING_CACHE}" ]]; then
  echo "missing prepared embedding cache: ${EMBEDDING_CACHE}" >&2
  exit 1
fi

mkdir -p "${RUN_DIR}"
cd "${TEST_ENV_DIR}"

"${TEST_ENV_DIR}/.venv/bin/python" scripts/layer2_dynamic_event_benchmark.py run \
  --tables 6 \
  --systems plasmod milvus \
  --run-id "${RUN_ID}" \
  --output-dir "${OUTPUT_DIR}" \
  --events-per-rate 0 \
  --fixed-write-rate 100 \
  --query-qps 5 \
  --query-limit 5000 \
  --workers 32 \
  --visibility-probe-limit 5000 \
  --bounded-sla-ms 1000 \
  --embedding-provider minilm \
  --embedding-cache "${EMBEDDING_CACHE}" \
  --embedding-batch-size 512 \
  --milvus-visibility-policy deferred \
  --milvus-index-type FLAT \
  --milvus-payload-json-bytes 0 \
  --reset-between-runs \
  --no-progress-timeout-s 600 \
  2>&1 | tee -a "${LOG_FILE}"
