#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-smoke}"
shift || true

case "${MODE}" in
  smoke)
    RUNNER_MODE="smoke"
    ;;
  run)
    RUNNER_MODE="run"
    ;;
  full)
    RUNNER_MODE="run"
    set -- --event-limit 0 "$@"
    ;;
  *)
    echo "usage: $0 {smoke|run|full} [runner arguments]" >&2
    exit 2
    ;;
esac

RUN_ID="agent_native_ablation_${MODE}_$(date -u +%Y%m%d_%H%M%S)"
cd "${ROOT}"
exec python3 scripts/agent_native_ablation_benchmark.py \
  "${RUNNER_MODE}" \
  --run-id "${RUN_ID}" \
  "$@"
