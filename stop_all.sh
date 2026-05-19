#!/bin/bash
# Stop all benchmark services
set -e

cd "$(dirname "$0")"

echo "=== Stopping all services ==="

# Plasmod
pkill -f "bin/plasmod" 2>/dev/null && echo "Plasmod stopped" || echo "Plasmod not running"

# Qdrant
pkill -f "qdrant/bin/qdrant" 2>/dev/null && echo "Qdrant stopped" || echo "Qdrant not running"

# Milvus
cd ../milvus && docker compose down 2>/dev/null && echo "Milvus containers stopped" || echo "Milvus not running"

echo "All services stopped."