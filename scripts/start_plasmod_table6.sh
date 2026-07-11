#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_ENV_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="$(cd "${TEST_ENV_DIR}/.." && pwd)"
PLASMOD_DIR="${ROOT_DIR}/Plasmod"
SESSION_NAME="${PLASMOD_TABLE6_SCREEN:-plasmod_table6}"
LOG_FILE="${PLASMOD_TABLE6_LOG:-/tmp/plasmod_table6.log}"

if curl -fsS -m 2 http://127.0.0.1:8080/healthz >/dev/null 2>&1; then
  echo "Plasmod: already healthy at http://127.0.0.1:8080"
  exit 0
fi

if [[ ! -x "${PLASMOD_DIR}/bin/plasmod" ]]; then
  echo "missing Plasmod binary: ${PLASMOD_DIR}/bin/plasmod" >&2
  exit 1
fi

screen -S "${SESSION_NAME}" -X quit >/dev/null 2>&1 || true
: > "${LOG_FILE}"

screen -dmS "${SESSION_NAME}" bash -lc "
  cd '${TEST_ENV_DIR}' &&
  env \
    DYLD_LIBRARY_PATH='${PLASMOD_DIR}/cpp/build:${PLASMOD_DIR}/cpp/build/vendor:\${DYLD_LIBRARY_PATH:-}' \
    PLASMOD_STORAGE=disk \
    PLASMOD_GRPC_ENABLED=0 \
    PLASMOD_BATCH_PLUGIN=1 \
    PLASMOD_HNSW_BATCH_DIRECT=1 \
    PLASMOD_IVF_BATCH_DIRECT=auto \
    PLASMOD_IVF_SERIAL_DIRECT=0 \
    PLASMOD_HNSW_EF_SEARCH=96 \
    OMP_NUM_THREADS=10 \
    '${PLASMOD_DIR}/bin/plasmod' >> '${LOG_FILE}' 2>&1
"

for _ in $(seq 1 30); do
  if curl -fsS -m 2 http://127.0.0.1:8080/healthz >/dev/null 2>&1; then
    echo "Plasmod: started in screen ${SESSION_NAME}"
    exit 0
  fi
  sleep 1
done

echo "Plasmod failed to become healthy; tailing ${LOG_FILE}" >&2
tail -n 120 "${LOG_FILE}" >&2 || true
exit 1
