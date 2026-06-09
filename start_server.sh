#!/bin/bash
# =============================================================================
# Plasmod server startup script
# Uses ONNX embedder, CPU-only mode
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/.env"

LOG_FILE="${PLASMOD_LOG_DIR}/server_$(date +%Y%m%d_%H%M%S).log"

echo "========================================="
echo "Plasmod ONNX Server (CPU mode)"
echo "========================================="
echo "Model:     ${PLASMOD_EMBEDDER_MODEL_PATH}"
echo "Storage:   ${PLASMOD_STORAGE:-disk (default)}"
echo "Data dir:  ${PLASMOD_DATA_DIR}"
echo "Log:       ${LOG_FILE}"
echo "========================================="

cd "${PLASMOD_ROOT}"

# Set library path for CGO HNSW
export DYLD_LIBRARY_PATH="${PLASMOD_ROOT}/cpp/build:${PLASMOD_ROOT}/cpp/build/vendor:${DYLD_LIBRARY_PATH:-}"
export PLASMOD_BATCH_PLUGIN=1
export PLASMOD_HNSW_BATCH_DIRECT="${PLASMOD_HNSW_BATCH_DIRECT:-1}"
export PLASMOD_IVF_BATCH_DIRECT="${PLASMOD_IVF_BATCH_DIRECT:-auto}"
export PLASMOD_IVF_SERIAL_DIRECT="${PLASMOD_IVF_SERIAL_DIRECT:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}"
export PLASMOD_PLUGIN_CHUNK_SIZE="${PLASMOD_PLUGIN_CHUNK_SIZE:-10000}"
export PLASMOD_HNSW_EF_SEARCH="${PLASMOD_HNSW_EF_SEARCH:-96}"
export PLASMOD_GRPC_ADDR="${PLASMOD_GRPC_ADDR:-0.0.0.0:19531}"
export PLASMOD_GRPC_MAX_MESSAGE_BYTES="${PLASMOD_GRPC_MAX_MESSAGE_BYTES:-1073741824}"
export PLASMOD_BENCH_QUERY_TRANSPORT="${PLASMOD_BENCH_QUERY_TRANSPORT:-grpc}"
export PLASMOD_HTTP_READ_TIMEOUT_SECONDS="${PLASMOD_HTTP_READ_TIMEOUT_SECONDS:-7200}"
export PLASMOD_HTTP_WRITE_TIMEOUT_SECONDS="${PLASMOD_HTTP_WRITE_TIMEOUT_SECONDS:-7200}"
export PLASMOD_HTTP_IDLE_TIMEOUT_SECONDS="${PLASMOD_HTTP_IDLE_TIMEOUT_SECONDS:-7200}"

nohup env \
PLASMOD_EMBEDDER=${PLASMOD_EMBEDDER} \
PLASMOD_EMBEDDER_MODEL_PATH=${PLASMOD_EMBEDDER_MODEL_PATH} \
PLASMOD_EMBEDDER_DIM=${PLASMOD_EMBEDDER_DIM} \
ONNXRUNTIME_LIB_PATH=${ONNXRUNTIME_LIB_PATH} \
PLASMOD_STORAGE=disk \
PLASMOD_DATA_DIR=${PLASMOD_DATA_DIR} \
APP_MODE=${APP_MODE} \
PLASMOD_HTTP_READ_TIMEOUT_SECONDS=${PLASMOD_HTTP_READ_TIMEOUT_SECONDS} \
PLASMOD_HTTP_WRITE_TIMEOUT_SECONDS=${PLASMOD_HTTP_WRITE_TIMEOUT_SECONDS} \
PLASMOD_HTTP_IDLE_TIMEOUT_SECONDS=${PLASMOD_HTTP_IDLE_TIMEOUT_SECONDS} \
PLASMOD_BATCH_PLUGIN=${PLASMOD_BATCH_PLUGIN} \
PLASMOD_PLUGIN_CHUNK_SIZE=${PLASMOD_PLUGIN_CHUNK_SIZE} \
PLASMOD_HNSW_BATCH_DIRECT=${PLASMOD_HNSW_BATCH_DIRECT} \
PLASMOD_IVF_BATCH_DIRECT=${PLASMOD_IVF_BATCH_DIRECT} \
PLASMOD_IVF_SERIAL_DIRECT=${PLASMOD_IVF_SERIAL_DIRECT} \
PLASMOD_HNSW_EF_SEARCH=${PLASMOD_HNSW_EF_SEARCH} \
PLASMOD_BENCH_QUERY_TRANSPORT=${PLASMOD_BENCH_QUERY_TRANSPORT} \
OMP_NUM_THREADS=${OMP_NUM_THREADS} \
PLASMOD_GRPC_ADDR=${PLASMOD_GRPC_ADDR} \
PLASMOD_GRPC_MAX_MESSAGE_BYTES=${PLASMOD_GRPC_MAX_MESSAGE_BYTES} \
"${PLASMOD_ROOT}/bin/plasmod" \
  > "${LOG_FILE}" 2>&1 &

SERVER_PID=$!
echo $SERVER_PID > "${SCRIPT_DIR}/.server.pid"
disown "${SERVER_PID}" 2>/dev/null || true

echo "Server started (PID: $SERVER_PID)"
echo "Log: ${LOG_FILE}"

for i in $(seq 1 20); do
    if curl -s --connect-timeout 1 http://127.0.0.1:8080/healthz > /dev/null 2>&1; then
        echo "Server is running on http://127.0.0.1:8080"
        curl -s http://127.0.0.1:8080/healthz
        echo ""
        exit 0
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        break
    fi
    sleep 1
done

echo "ERROR: Server failed to start or did not pass health check"
echo "Log output:"
cat "${LOG_FILE}"
exit 1
