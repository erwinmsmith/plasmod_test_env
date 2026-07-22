#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CORE="${ROOT}/../Plasmod"
MODE="${1:-smoke}"
PORT=18080

if [[ "${MODE}" != "smoke" && "${MODE}" != "full" ]]; then
  echo "usage: $0 {smoke|full} [--port PORT]" >&2
  exit 2
fi
shift || true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      [[ $# -ge 2 ]] || { echo "--port requires a value" >&2; exit 2; }
      PORT="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if ! [[ "${PORT}" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
  echo "invalid port: ${PORT}" >&2
  exit 2
fi

errors=0
warnings=0

pass() { printf '[PASS] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*"; warnings=$((warnings + 1)); }
fail() { printf '[FAIL] %s\n' "$*"; errors=$((errors + 1)); }

need_command() {
  if command -v "$1" >/dev/null 2>&1; then
    pass "command available: $1"
  else
    fail "missing command: $1"
  fi
}

version_ge() {
  python3 - "$1" "$2" <<'PY'
import re
import sys

def parts(value):
    numbers = [int(item) for item in re.findall(r"\d+", value)[:3]]
    return tuple((numbers + [0, 0, 0])[:3])

raise SystemExit(0 if parts(sys.argv[1]) >= parts(sys.argv[2]) else 1)
PY
}

port_open() {
  python3 - "$1" <<'PY'
import socket
import sys

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(0.5)
    raise SystemExit(0 if sock.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
PY
}

echo "Agent-native ablation preflight"
echo "  mode:             ${MODE}"
echo "  experiment repo:  ${ROOT}"
echo "  core repo:        ${CORE}"
echo "  Plasmod port:     ${PORT}"
echo

for command in git python3 go cmake c++ make curl minio mc ps; do
  need_command "${command}"
done

if command -v python3 >/dev/null 2>&1; then
  python_version="$(python3 -c 'import platform; print(platform.python_version())')"
  if version_ge "${python_version}" "3.9.0"; then
    pass "Python ${python_version} >= 3.9"
  else
    fail "Python ${python_version} is too old; Python >= 3.9 is required"
  fi
fi

if command -v go >/dev/null 2>&1; then
  go_version="$(go env GOVERSION 2>/dev/null | sed 's/^go//')"
  required_go="$(awk '$1 == "go" {print $2; exit}' "${CORE}/go.mod" 2>/dev/null || true)"
  required_go="${required_go:-1.25.0}"
  if version_ge "${go_version}" "${required_go}"; then
    pass "Go ${go_version} >= ${required_go}"
  else
    fail "Go ${go_version} is too old; core go.mod requires ${required_go}"
  fi
fi

if command -v cmake >/dev/null 2>&1; then
  cmake_version="$(cmake --version | awk 'NR == 1 {print $3}')"
  if version_ge "${cmake_version}" "3.20.0"; then
    pass "CMake ${cmake_version} >= 3.20"
  else
    fail "CMake ${cmake_version} is too old; CMake >= 3.20 is required"
  fi
fi

case "$(uname -s)" in
  Linux) pass "Linux host detected ($(uname -m))" ;;
  Darwin) warn "macOS host detected; use Linux for the documented server run" ;;
  *) warn "untested operating system: $(uname -s) $(uname -m)" ;;
esac

if [[ -d "${CORE}/.git" ]]; then
  core_branch="$(git -C "${CORE}" branch --show-current)"
  core_commit="$(git -C "${CORE}" rev-parse --short HEAD)"
  if [[ "${core_branch}" == "dev" ]]; then
    pass "Plasmod is on dev (${core_commit})"
  else
    fail "Plasmod must be on dev; current branch is '${core_branch:-detached}' (${core_commit})"
  fi
  if [[ -n "$(git -C "${CORE}" status --porcelain --untracked-files=no)" ]]; then
    warn "Plasmod has tracked local changes; commit or record them before a formal run"
  fi
else
  fail "missing sibling core repository: ${CORE}"
fi

if [[ -d "${ROOT}/.git" ]]; then
  experiment_branch="$(git -C "${ROOT}" branch --show-current)"
  experiment_commit="$(git -C "${ROOT}" rev-parse --short HEAD)"
  if [[ "${experiment_branch}" == "main" ]]; then
    pass "plasmod_test_env is on main (${experiment_commit})"
  else
    fail "plasmod_test_env must be on main; current branch is '${experiment_branch:-detached}' (${experiment_commit})"
  fi
  if [[ -n "$(git -C "${ROOT}" status --porcelain --untracked-files=no)" ]]; then
    warn "plasmod_test_env has tracked local changes; commit or record them before a formal run"
  fi
else
  fail "experiment repository is not a Git checkout: ${ROOT}"
fi

if [[ -x "${CORE}/bin/plasmod" ]]; then
  pass "Plasmod binary is executable"
else
  fail "missing ${CORE}/bin/plasmod; run the documented C++ build and make build"
fi

if [[ "$(uname -s)" == "Linux" ]]; then
  retrieval_library="${CORE}/cpp/build/libplasmod_retrieval.so"
else
  retrieval_library="${CORE}/cpp/build/libplasmod_retrieval.dylib"
fi
if [[ -f "${retrieval_library}" ]]; then
  pass "retrieval shared library exists: ${retrieval_library}"
else
  fail "missing retrieval shared library: ${retrieval_library}"
fi

if [[ "$(uname -s)" == "Linux" ]] && command -v ldd >/dev/null 2>&1 && [[ -x "${CORE}/bin/plasmod" ]]; then
  missing_libraries="$(ldd "${CORE}/bin/plasmod" 2>/dev/null | grep 'not found' || true)"
  if [[ -z "${missing_libraries}" ]]; then
    pass "Plasmod dynamic-library resolution has no missing entries"
  else
    fail "Plasmod has unresolved dynamic libraries: ${missing_libraries}"
  fi
fi

events_file="${ROOT}/data/layer2_dynamic_events/events.jsonl"
traces_dir="${ROOT}/data/layer2_dynamic_events/traces_collected"
cache_file="${ROOT}/results/layer2_dynamic_events/embedding_cache.sqlite3"

if [[ -s "${events_file}" ]]; then
  pass "replay event input exists ($(wc -l < "${events_file}" | tr -d ' ') lines)"
else
  fail "missing or empty replay input: ${events_file}"
fi

trace_count=0
if [[ -d "${traces_dir}" ]]; then
  trace_count="$(find "${traces_dir}" -type f -name '*.jsonl' | wc -l | tr -d ' ')"
fi
if (( trace_count > 0 )); then
  pass "recorded trace input exists (${trace_count} JSONL files)"
else
  fail "no trace JSONL files found under ${traces_dir}"
fi

if [[ -s "${cache_file}" ]]; then
  if python3 - "${cache_file}" <<'PY'
import sqlite3
import sys

connection = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
result = connection.execute("PRAGMA quick_check").fetchone()[0]
connection.close()
raise SystemExit(0 if result == "ok" else 1)
PY
  then
    pass "embedding cache exists and passes SQLite quick_check"
  else
    fail "embedding cache is corrupt or unreadable: ${cache_file}"
  fi
else
  warn "embedding cache is absent; the runner will recreate it and add preparation time"
fi

mkdir -p "${ROOT}/results/agent_native_ablation" "${ROOT}/logs"
for path in "${ROOT}/results/agent_native_ablation" "${ROOT}/logs" "$(dirname "${cache_file}")"; do
  if [[ -w "${path}" ]]; then
    pass "writable path: ${path}"
  else
    fail "path is not writable: ${path}"
  fi
done

free_kb="$(df -Pk "${ROOT}" | awk 'NR == 2 {print $4}')"
free_gb=$((free_kb / 1024 / 1024))
if [[ "${MODE}" == "full" ]]; then
  minimum_free_gb="${PLASMOD_ABLATION_MIN_FREE_GB:-250}"
else
  minimum_free_gb="${PLASMOD_ABLATION_MIN_FREE_GB:-10}"
fi
if (( free_gb >= minimum_free_gb )); then
  pass "free disk ${free_gb} GB >= ${minimum_free_gb} GB required for ${MODE}"
else
  fail "free disk ${free_gb} GB < ${minimum_free_gb} GB required for ${MODE}"
fi

cpu_count="$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 0)"
if (( cpu_count >= 8 )); then
  pass "logical CPU count: ${cpu_count}"
else
  warn "logical CPU count is ${cpu_count}; 8 or more is recommended"
fi

memory_gb=0
if [[ -r /proc/meminfo ]]; then
  memory_kb="$(awk '/MemTotal:/ {print $2; exit}' /proc/meminfo)"
  memory_gb=$((memory_kb / 1024 / 1024))
elif command -v sysctl >/dev/null 2>&1; then
  memory_bytes="$(sysctl -n hw.memsize 2>/dev/null || echo 0)"
  memory_gb=$((memory_bytes / 1024 / 1024 / 1024))
fi
if (( memory_gb >= 32 )); then
  pass "physical memory: ${memory_gb} GB"
else
  warn "physical memory is ${memory_gb} GB; 32 GB or more is recommended"
fi

open_files="$(ulimit -n)"
if [[ "${open_files}" == "unlimited" ]] || (( open_files >= 65536 )); then
  pass "open-file limit: ${open_files}"
else
  warn "open-file limit is ${open_files}; set ulimit -n 65536 for a full run"
fi

if port_open "${PORT}"; then
  fail "Plasmod port ${PORT} is already in use; stop that process or select another --port"
else
  pass "Plasmod port ${PORT} is free"
fi

if curl -fsS --max-time 2 http://127.0.0.1:9000/minio/health/live >/dev/null 2>&1; then
  if mc alias set plasmod-ablation-preflight http://127.0.0.1:9000 minioadmin minioadmin >/dev/null 2>&1 \
      && mc ls plasmod-ablation-preflight >/dev/null 2>&1; then
    pass "existing MinIO is healthy and accepts minioadmin credentials"
  else
    fail "existing MinIO is healthy but does not accept the credentials required by the runner"
  fi
else
  if port_open 9000; then
    fail "port 9000 is occupied by a service that is not a healthy MinIO"
  else
    pass "MinIO API port 9000 is free; the runner will start MinIO"
  fi
  if port_open 9001; then
    fail "MinIO console port 9001 is occupied; the runner cannot start its isolated MinIO"
  else
    pass "MinIO console port 9001 is free"
  fi
fi

if command -v pgrep >/dev/null 2>&1; then
  if pgrep -f '[a]gent_native_ablation_benchmark.py' >/dev/null 2>&1; then
    fail "another agent-native ablation runner is active"
  else
    pass "no other agent-native ablation runner detected"
  fi
  if pgrep -f '/bin/plasmod|[p]lasmod.*server' >/dev/null 2>&1; then
    warn "another Plasmod process is active; stop unrelated workloads for formal measurements"
  fi
fi

echo
printf 'Preflight summary: %d error(s), %d warning(s)\n' "${errors}" "${warnings}"
if (( errors > 0 )); then
  echo "Do not start the experiment until all FAIL items are resolved."
  exit 1
fi
echo "Preflight passed. Warnings should be reviewed before a formal full run."
