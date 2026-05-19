#!/bin/bash
# Start all benchmark services
set -e

cd "$(dirname "$0")/.."

echo "=== Starting all services ==="

# Qdrant
QDRANT_DATA_DIR="$(pwd)/qdrant/storage" \
  nohup qdrant/bin/qdrant > /tmp/qdrant.log 2>&1 &
echo "Qdrant PID: $!"
sleep 3

# Milvus (docker)
cd milvus && docker compose up -d
cd ..

# Plasmod
./bin/plasmod > /tmp/plasmod.log 2>&1 &
echo "Plasmod PID: $!"
sleep 3

# Verify all
echo ""
echo "=== Port Status ==="
for port in 6333 8080 19530 9091; do
  curl -s --connect-timeout 3 http://127.0.0.1:$port/ >/dev/null 2>&1 && echo "$port: UP" || echo "$port: DOWN"
done
echo "LanceDB: ready (embedded, no port)"