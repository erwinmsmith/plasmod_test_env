#!/bin/bash
# Stop the running Plasmod server

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="${SCRIPT_DIR}/.server.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        kill "$PID"
        echo "Server (PID: $PID) stopped"
    else
        echo "Server not running (stale PID file)"
    fi
    rm -f "$PID_FILE"
fi

PORT_PID=$(lsof -nP -tiTCP:8080 -sTCP:LISTEN 2>/dev/null | head -1 || true)
if [ -n "$PORT_PID" ]; then
    kill "$PORT_PID" 2>/dev/null || true
    echo "Server listener on 8080 (PID: $PORT_PID) stopped"
elif [ ! -f "$PID_FILE" ]; then
    echo "No Plasmod listener found on 8080"
fi
