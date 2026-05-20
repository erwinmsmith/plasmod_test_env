#!/bin/bash
# Stop all benchmark services
set -e

cd "$(dirname "$0")"
MINIO_PID="$(dirname "$0")/../minio/.minio.pid"

echo "=== Stopping all services ==="

# Plasmod
pkill -f "bin/plasmod" 2>/dev/null && echo "Plasmod: stopped" || echo "Plasmod: not running"

# Qdrant
pkill -f "qdrant/bin/qdrant" 2>/dev/null && echo "Qdrant: stopped" || echo "Qdrant: not running"

# Milvus (docker)
cd ../milvus && docker compose down 2>/dev/null && echo "Milvus: stopped" || echo "Milvus: not running"
cd ..

# MinIO (local)
if [ -f "${MINIO_PID}" ]; then
  kill "$(cat "${MINIO_PID}")" 2>/dev/null && echo "MinIO: stopped" || echo "MinIO: not running"
  rm -f "${MINIO_PID}"
else
  pkill -f "minio server" 2>/dev/null && echo "MinIO: stopped" || echo "MinIO: not running"
fi

echo "All services stopped."
