#!/bin/bash
# Start all benchmark services in correct dependency order.
# MinIO (local) → Milvus (docker) → Qdrant (local) → Plasmod (local)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MINIO_DATA_DIR="${SCRIPT_DIR}/../minio/data"
MINIO_PID="${SCRIPT_DIR}/../minio/.minio.pid"

echo "=== Starting all services ==="

# ── 1. MinIO (local, port 9000/9001) ─────────────────────────────────────────
start_minio() {
  if curl -s --connect-timeout 2 http://127.0.0.1:9000/minio/health/live > /dev/null 2>&1; then
    echo "MinIO: already running (port 9000)"
  else
    if ! command -v minio > /dev/null 2>&1; then
      echo "MinIO: binary not found; skipping local MinIO"
      return
    fi
    echo "Starting MinIO on port 9000..."
    mkdir -p "${MINIO_DATA_DIR}"
    nohup minio server "${MINIO_DATA_DIR}" \
      --address ":9000" --console-address ":9001" \
      > /tmp/minio.log 2>&1 &
    echo $! > "${MINIO_PID}"
    for i in $(seq 1 10); do
      sleep 1
      curl -s --connect-timeout 2 http://127.0.0.1:9000/minio/health/live > /dev/null 2>&1 && break
    done
    if command -v mc > /dev/null 2>&1; then
      mc alias set myminio http://127.0.0.1:9000 minioadmin minioadmin 2>/dev/null || true
      mc mb myminio/plasmod-experiments 2>/dev/null || true
    fi
    echo "MinIO: started"
  fi
}

# ── 2. Milvus (docker, port 19530/9091) ─────────────────────────────────────
start_milvus() {
  cd "${SCRIPT_DIR}/../milvus"
  # Ensure etcd + minio (port 9002/9003) + milvus are up
  docker compose up -d
  echo "Milvus: started (Milvus:19530, Console:9091, Milvus-MinIO:9002)"
  cd "${SCRIPT_DIR}"
}

# ── 3. Qdrant (local binary, port 6333/6334) ─────────────────────────────────
start_qdrant() {
  if curl -s --connect-timeout 2 http://127.0.0.1:6333 > /dev/null 2>&1; then
    echo "Qdrant: already running (port 6333)"
  else
    QDRANT_DIR="${SCRIPT_DIR}/../qdrant"
    QDRANT_DATA_DIR="${QDRANT_DIR}/storage"
    if [[ ! -x "${QDRANT_DIR}/bin/qdrant" ]]; then
      echo "Qdrant: binary not found; skipping local Qdrant"
      return
    fi
    mkdir -p "${QDRANT_DATA_DIR}"
    nohup env QDRANT__STORAGE__STORAGE_PATH="${QDRANT_DATA_DIR}" \
      "${QDRANT_DIR}/bin/qdrant" \
      > /tmp/qdrant.log 2>&1 &
    echo "Qdrant: started (PID: $!)"
    sleep 3
  fi
}

# ── 4. Plasmod (local, port 8080) ────────────────────────────────────────────
start_plasmod() {
  if curl -s --connect-timeout 2 http://127.0.0.1:8080/healthz > /dev/null 2>&1; then
    echo "Plasmod: already running (port 8080)"
  else
    source "${SCRIPT_DIR}/.env"
    PLASMOD_ROOT="${PLASMOD_ROOT:-${SCRIPT_DIR}/..}"
    if [[ -x "${PLASMOD_ROOT}/bin/plasmod" ]]; then
      PLASMOD_BIN="${PLASMOD_ROOT}/bin/plasmod"
      PLASMOD_CPP_DIR="${PLASMOD_ROOT}/cpp/build"
    else
      PLASMOD_BIN="${PLASMOD_ROOT}/Plasmod/bin/plasmod"
      PLASMOD_CPP_DIR="${PLASMOD_ROOT}/Plasmod/cpp/build"
    fi
    export DYLD_LIBRARY_PATH="${PLASMOD_CPP_DIR}:${PLASMOD_CPP_DIR}/vendor:${DYLD_LIBRARY_PATH:-}"
    export PLASMOD_BATCH_PLUGIN=1
    export PLASMOD_HNSW_BATCH_DIRECT="${PLASMOD_HNSW_BATCH_DIRECT:-1}"
    export PLASMOD_IVF_BATCH_DIRECT="${PLASMOD_IVF_BATCH_DIRECT:-auto}"
    export PLASMOD_IVF_SERIAL_DIRECT="${PLASMOD_IVF_SERIAL_DIRECT:-0}"
    export PLASMOD_HNSW_EF_SEARCH="${PLASMOD_HNSW_EF_SEARCH:-96}"
    export PLASMOD_GRPC_ADDR="${PLASMOD_GRPC_ADDR:-0.0.0.0:19531}"
    export PLASMOD_GRPC_MAX_MESSAGE_BYTES="${PLASMOD_GRPC_MAX_MESSAGE_BYTES:-1073741824}"
    export PLASMOD_GRPC_ENABLED="${PLASMOD_GRPC_ENABLED:-0}"
    export PLASMOD_BENCH_QUERY_TRANSPORT="${PLASMOD_BENCH_QUERY_TRANSPORT:-grpc}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}"
    S3_ENV_ARGS=()
    if curl -s --connect-timeout 2 http://127.0.0.1:9000/minio/health/live > /dev/null 2>&1; then
      S3_ENV_ARGS=(
        "S3_ENDPOINT=${S3_ENDPOINT:-127.0.0.1:9000}"
        "S3_ACCESS_KEY=${S3_ACCESS_KEY:-minioadmin}"
        "S3_SECRET_KEY=${S3_SECRET_KEY:-minioadmin}"
        "S3_BUCKET=${S3_BUCKET:-plasmod-experiments}"
        "S3_SECURE=${S3_SECURE:-false}"
      )
    else
      echo "Plasmod: local MinIO unavailable; using in-memory cold store"
    fi

    nohup env \
      "${S3_ENV_ARGS[@]}" \
      PLASMOD_STORAGE=disk \
      PLASMOD_BATCH_PLUGIN="${PLASMOD_BATCH_PLUGIN}" \
      PLASMOD_HNSW_BATCH_DIRECT="${PLASMOD_HNSW_BATCH_DIRECT}" \
      PLASMOD_IVF_BATCH_DIRECT="${PLASMOD_IVF_BATCH_DIRECT}" \
      PLASMOD_IVF_SERIAL_DIRECT="${PLASMOD_IVF_SERIAL_DIRECT}" \
      PLASMOD_HNSW_EF_SEARCH="${PLASMOD_HNSW_EF_SEARCH}" \
      PLASMOD_BENCH_QUERY_TRANSPORT="${PLASMOD_BENCH_QUERY_TRANSPORT}" \
      PLASMOD_GRPC_ENABLED="${PLASMOD_GRPC_ENABLED}" \
      PLASMOD_GRPC_ADDR="${PLASMOD_GRPC_ADDR}" \
      PLASMOD_GRPC_MAX_MESSAGE_BYTES="${PLASMOD_GRPC_MAX_MESSAGE_BYTES}" \
      OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
    "${PLASMOD_BIN}" \
      >> /tmp/plasmod.log 2>&1 &
    plasmod_pid=$!
    disown "${plasmod_pid}" 2>/dev/null || true
    echo "Plasmod: started (PID: ${plasmod_pid})"
    sleep 5
  fi
}

# ── Start in order ────────────────────────────────────────────────────────────
start_minio
start_milvus
start_qdrant
start_plasmod

# ── Port status ───────────────────────────────────────────────────────────────
echo ""
echo "=== Port Status ==="
for entry in "9000:MinIO" "9001:MinIO-Console" "6333:Qdrant" "6334:Qdrant-grpc" "8080:Plasmod" "19530:Milvus" "9091:Milvus-Console"; do
  port="${entry%%:*}"
  name="${entry##*:}"
  curl -s --connect-timeout 2 http://127.0.0.1:${port}/ > /dev/null 2>&1 && echo "$port ($name): UP" || echo "$port ($name): DOWN"
done
echo "LanceDB: ready (embedded, no port)"
echo "Milvus-MinIO: 9002 (internal port 9000 mapped)"
echo ""
echo "MinIO S3 endpoint: 127.0.0.1:9000 | Bucket: plasmod-experiments"
