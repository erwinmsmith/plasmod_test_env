#!/usr/bin/env python3
"""Real-service Plasmod capability ablations over recorded agent events.

The runner owns only experiment orchestration. Every variant starts the same
Plasmod binary and selects a documented production capability profile through
environment variables. Results are fail-fast and every CSV cell is validated.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import sqlite3
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT.parent / "Plasmod"
DATA = ROOT / "data" / "layer2_dynamic_events"
DEFAULT_BUCKET = "plasmod-experiments"
EMBEDDING_CACHE_PATH = ROOT / "results" / "layer2_dynamic_events" / "embedding_cache.sqlite3"
DEFAULT_PLASMOD_START_TIMEOUT_S = 300.0
DEFAULT_RECOVERY_REPLAY_TIMEOUT_S = 300.0
DEFAULT_RECOVERY_RESET_TIMEOUT_S = 300.0
RECOVERY_REPLAY_TIMEOUT_GRACE_S = 60.0
RECOVERY_REPLAY_TIMEOUT_EVENTS_PER_SECOND = 45.0
RECOVERY_RESET_TIMEOUT_GRACE_S = 600.0
RECOVERY_RESET_TIMEOUT_EVENTS_PER_SECOND = 30.0

CAPABILITY_ENV_FIELDS = {
    "PLASMOD_WAL_MODE": "wal_mode",
    "PLASMOD_RECOVERY_PROJECTION": "recovery_projection",
    "PLASMOD_MATERIALIZATION_PROFILE": "materialization_profile",
    "PLASMOD_EVIDENCE_PROFILE": "evidence_profile",
    "PLASMOD_GOVERNANCE_PROFILE": "governance_profile",
    "PLASMOD_TIER_PROFILE": "tier_profile",
}
FATAL_SERVICE_LOG = re.compile(
    r"panic:|fatal error:|segmentation fault|SignatureDoesNotMatch|AddressSanitizer|\[ERROR\]|"
    r"can't assign requested address|connection refused|s3cold: (?:get|put|delete|ensure).*: s3 ",
    re.IGNORECASE,
)

WAL_FIELDS = [
    "System", "Variant", "Event Log Size", "Recovered Objects (%)",
    "Recovered Relations (%)", "Recovered Latest State (%)", "Recovery Time (s)",
    "Replay Throughput (events/s)", "Query Available During Recovery",
    "Lost Event Count", "Duplicate Object Count",
]
MATERIALIZATION_FIELDS = [
    "System", "Variant", "Write QPS", "Write p95 (ms)",
    "Write-to-Visible p95 (ms)", "Materialization Lag p95 (ms)",
    "Object Visibility Coverage (%)", "Latest-state Hit Rate (%)",
    "Artifact Lookup Accuracy (%)", "Relation Recovery Rate (%)",
    "Stale Result Rate (%)",
]
EVIDENCE_FIELDS = [
    "System", "Variant", "Query p95 (ms)", "Evidence Assembly Latency p95 (ms)",
    "Provenance Completeness (%)", "Edge Recall (%)", "Proof Completeness (%)",
    "Citation / Evidence Correctness (%)", "Stale Evidence Rate (%)",
]
GOVERNANCE_FIELDS = [
    "System", "Variant", "Private Memory Leakage Rate (%)",
    "Authorized Hit Rate (%)", "Unauthorized Hit Rate (%)",
    "Delete Visibility Delay (ms)", "Quarantine Exclusion Rate (%)",
    "Policy Overhead (ms)",
]
TIER_FIELDS = [
    "System", "Variant", "Hot Cache Size", "Query p50 (ms)", "Query p95 (ms)",
    "Query p99 (ms)", "Hot Hit Rate (%)", "Warm Hit Rate (%)",
    "Cold Hit Rate (%)", "Promotion Latency p95 (ms)", "Memory (MB)",
    "Stale Rate (%)",
]

COMMON_PARAMETER_SET = "agent-native-common-v1"
COMMON_FIELDS = [
    "Common | Event Count", "Common | Query Count", "Common | Stale Check Count",
    "Common | TopK", "Common | Embedding Dimension", "Common | Write QPS",
    "Common | Write p50 (ms)", "Common | Write p95 (ms)", "Common | Write p99 (ms)",
    "Common | Write-to-Visible p50 (ms)", "Common | Write-to-Visible p95 (ms)",
    "Common | Materialization Lag p95 (ms)", "Common | Query QPS",
    "Common | Query p50 (ms)", "Common | Query p95 (ms)", "Common | Query p99 (ms)",
    "Common | Memory (MB)", "Common | Object Visibility Coverage (%)",
    "Common | Target Stale Rate (%)",
]
MODULE_FIELDS = {
    "wal": WAL_FIELDS[2:],
    "materialization": MATERIALIZATION_FIELDS[2:],
    "evidence": EVIDENCE_FIELDS[2:],
    "governance": GOVERNANCE_FIELDS[2:],
    "tier": TIER_FIELDS[2:],
}
MASTER_IDENTITY_FIELDS = [
    "System", "Module", "Original Variant", "Comparison Label", "Ablated Capability",
    "Parameter Set", "Write Consistency", "Query Consistency", "Storage Backend",
    "Cold Store", "WAL Mode", "Recovery Replay", "Recovery Projection",
    "Materialization Profile", "Evidence Profile", "Governance Profile", "Tier Profile",
    "Hot Cache Size",
]
MASTER_FIELDS = MASTER_IDENTITY_FIELDS + COMMON_FIELDS + [
    f"{group.upper()} | {field}"
    for group, fields in MODULE_FIELDS.items()
    for field in fields
]
NOT_APPLICABLE = "N/A (not applicable)"

COMPARISON_LABELS = {
    "wal": {
        "Full Plasmod": ("Full", "None"),
        "No-WAL": ("w/o WAL / Event Log", "Durable event log"),
        "In-memory WAL": ("w/ In-memory WAL", "Durable WAL persistence"),
        "File WAL": ("File WAL control", "None"),
        "WAL without replay": ("w/o Replay", "Recovery replay"),
        "Replay without index rebuild": ("w/o Projection Rebuild", "Retrieval projection rebuild"),
    },
    "materialization": {
        "Full Plasmod": ("Full", "None"),
        "No-materialization": ("w/o Canonical Materialization", "Canonical object materialization"),
        "Memory-only": ("Memory-only Materialization", "State, artifact, edge, and version materialization"),
        "No-agent-state": ("w/o Agent State", "State materialization"),
        "No-artifact": ("w/o Artifact", "Artifact materialization"),
        "No-edge": ("w/o Edge", "Relation edge materialization"),
        "No-object-version": ("w/o Object Version", "Object version materialization"),
    },
    "evidence": {
        "Full Plasmod": ("Full", "None"),
        "No-evidence": ("w/o Evidence Assembly", "Evidence assembly"),
        "No-provenance": ("w/o Provenance", "Provenance resolution"),
        "No-edge-expansion": ("w/o Edge Expansion", "Graph edge expansion"),
        "One-hop only": ("One-hop Evidence Only", "Multi-hop evidence expansion"),
        "No-proof-trace": ("w/o Proof Trace", "Proof trace construction"),
        "Vector-only": ("Vector-only Retrieval", "Canonical evidence hydration"),
    },
    "governance": {
        "Full Plasmod": ("Full", "None"),
        "No-access-policy": ("w/o Access Policy", "Access policy enforcement"),
        "Metadata-filter-only": ("Metadata Filter Only", "Policy engine and share contracts"),
        "No-share-contract": ("w/o Share Contract", "Share contract enforcement"),
        "No-quarantine": ("w/o Quarantine", "Quarantine exclusion"),
        "No-delete-propagation": ("w/o Delete Propagation", "Deletion propagation"),
    },
    "tier": {
        "Full Tiering": ("Full", "None"),
        "No-hot-cache": ("w/o Hot Cache", "Hot cache"),
        "Warm-only": ("Warm Tier Only", "Hot and cold tiers"),
        "No-cold": ("w/o Cold Tier", "Cold S3 tier"),
        "No-promotion": ("w/o Promotion", "Tier promotion"),
        "Hot-size-64": ("Hot Cache = 64", "Hot cache capacity control"),
        "Hot-size-512": ("Hot Cache = 512", "Hot cache capacity control"),
        "Hot-size-2000": ("Hot Cache = 2000", "Hot cache capacity control"),
    },
}


def utc_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def percentile(values: Iterable[float], q: float) -> float:
    xs = sorted(float(v) for v in values)
    if not xs:
        return 0.0
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return xs[lo] if lo == hi else xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def pct(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 100.0
    return max(0.0, min(100.0, 100.0 * numerator / denominator))


def require_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        return float(value)
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise RuntimeError(f"{name} is not finite: {value!r}")
    return number


def validate_result_row(fields: list[str], row: dict[str, Any], source: str) -> None:
    missing = [field for field in fields if field not in row or row[field] in (None, "")]
    if missing:
        raise RuntimeError(f"{source} missing fields: {missing}")
    unexpected = [field for field in row if field not in fields]
    if unexpected:
        raise RuntimeError(f"{source} has unexpected fields: {unexpected}")
    for field, value in row.items():
        if field not in ("System", "Variant", "Query Available During Recovery") and not isinstance(value, str):
            require_number(value, f"{source}:{field}")


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def mark_run_started(run_dir: Path) -> None:
    for name in ("FAILED", "COMPLETE"):
        try:
            (run_dir / name).unlink()
        except FileNotFoundError:
            pass
    atomic_write_json(run_dir / "RUNNING", {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
    })


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty result table {path.name}")
    for row_no, row in enumerate(rows, 1):
        validate_result_row(fields, row, f"{path.name} row {row_no}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_service_logs(run_dir: Path) -> None:
    failures: list[str] = []
    for path in sorted((run_dir / "variants").glob("*/server.log")):
        for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if FATAL_SERVICE_LOG.search(line):
                failures.append(f"{path}:{line_no}: {line[:300]}")
                if len(failures) >= 20:
                    break
        if len(failures) >= 20:
            break
    if failures:
        raise RuntimeError("fatal service log entries detected:\n" + "\n".join(failures))


def http_json(base: str, method: str, path: str, body: Any | None = None, timeout: float = 60.0) -> Any:
    payload = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = Request(
        base + path,
        data=payload,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"{method} {path} returned HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(f"{method} {path} failed: {exc}") from exc
    if not raw:
        return {}
    return json.loads(raw)


def plasmod_start_timeout_s() -> float:
    raw = os.getenv("PLASMOD_ABLATION_PLASMOD_START_TIMEOUT_S")
    if raw is None or not raw.strip():
        return DEFAULT_PLASMOD_START_TIMEOUT_S
    try:
        timeout_s = float(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"PLASMOD_ABLATION_PLASMOD_START_TIMEOUT_S must be numeric, got {raw!r}"
        ) from exc
    if timeout_s <= 0:
        raise RuntimeError(
            f"PLASMOD_ABLATION_PLASMOD_START_TIMEOUT_S must be positive, got {raw!r}"
        )
    return timeout_s


def recovery_replay_timeout_s(event_count: int) -> float:
    scaled_timeout = (
        math.ceil(max(event_count, 0) / RECOVERY_REPLAY_TIMEOUT_EVENTS_PER_SECOND)
        + RECOVERY_REPLAY_TIMEOUT_GRACE_S
    )
    return max(DEFAULT_RECOVERY_REPLAY_TIMEOUT_S, scaled_timeout)


def recovery_reset_timeout_s(event_count: int) -> float:
    if event_count <= DEFAULT_RECOVERY_RESET_TIMEOUT_S * RECOVERY_RESET_TIMEOUT_EVENTS_PER_SECOND:
        return DEFAULT_RECOVERY_RESET_TIMEOUT_S
    scaled_timeout = (
        math.ceil(max(event_count, 0) / RECOVERY_RESET_TIMEOUT_EVENTS_PER_SECOND)
        + RECOVERY_RESET_TIMEOUT_GRACE_S
    )
    return max(DEFAULT_RECOVERY_RESET_TIMEOUT_S, scaled_timeout)


def text_from_event(event: dict[str, Any]) -> str:
    retrieval = event.get("retrieval") or {}
    if retrieval.get("index_text"):
        return str(retrieval["index_text"])
    payload = event.get("payload") or {}
    for key in ("content", "text", "result", "observation", "thought"):
        if key in payload and payload[key] not in (None, ""):
            return str(payload[key])
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def hash_vector(text: str, dim: int = 384) -> list[float]:
    vector = [0.0] * dim
    tokens = text.lower().split()[:1024] or [text or "empty"]
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8", "ignore"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "little") % dim
        vector[index] += 1.0 if digest[4] & 1 else -1.0
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


class EmbeddingCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS agent_native_hash_embeddings "
            "(cache_key TEXT PRIMARY KEY, dim INTEGER NOT NULL, vector BLOB NOT NULL)"
        )
        self.pending = 0

    def get(self, text: str, dim: int = 384) -> list[float]:
        key = hashlib.sha256(f"agent-native-hash-v1:{dim}:{text}".encode("utf-8", "ignore")).hexdigest()
        row = self.connection.execute(
            "SELECT dim, vector FROM agent_native_hash_embeddings WHERE cache_key=?", (key,)
        ).fetchone()
        if row is not None and int(row[0]) == dim:
            return list(struct.unpack(f"<{dim}f", row[1]))
        vector = hash_vector(text, dim)
        self.connection.execute(
            "INSERT OR REPLACE INTO agent_native_hash_embeddings(cache_key, dim, vector) VALUES(?,?,?)",
            (key, dim, struct.pack(f"<{dim}f", *vector)),
        )
        self.pending += 1
        if self.pending >= 1000:
            self.connection.commit()
            self.pending = 0
        return vector

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()


EMBEDDING_CACHE: EmbeddingCache | None = None


def event_files() -> list[Path]:
    replay = [DATA / "events.jsonl"]
    synthetic = sorted((DATA / "traces_collected").glob("*.jsonl"))
    paths = [path for path in replay + synthetic if path.exists()]
    if not paths:
        raise RuntimeError(f"no event data found under {DATA}")
    return paths


def iter_events(limit: int) -> Iterable[dict[str, Any]]:
    emitted = 0
    for path in event_files():
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"invalid JSON {path}:{line_no}: {exc}") from exc
                yield event
                emitted += 1
                if limit > 0 and emitted >= limit:
                    return


def prepare_event(source: dict[str, Any], ordinal: int, prefix: str) -> tuple[dict[str, Any], str, list[float]]:
    event = copy.deepcopy(source)
    identity = event.setdefault("identity", {})
    actor = event.setdefault("actor", {})
    access = event.setdefault("access", {})
    retrieval = event.setdefault("retrieval", {})
    original = str(identity.get("event_id") or event.get("event_id") or f"event_{ordinal}")
    event_id = f"{prefix}_{ordinal:08d}_{original}"[-240:]
    identity["event_id"] = event_id
    identity["tenant_id"] = "default"
    identity["workspace_id"] = "plasmod-ablation"
    actor.setdefault("agent_id", "agent-runtime")
    actor.setdefault("session_id", f"{prefix}-session")
    access["consistency"] = "strict"
    text = text_from_event(event)
    retrieval["retrieval_namespace"] = "plasmod-ablation"
    vector = EMBEDDING_CACHE.get(text) if EMBEDDING_CACHE is not None else hash_vector(text)
    retrieval["embedding_vector"] = vector
    retrieval["embedding_dim"] = len(vector)
    retrieval["has_embedding"] = True
    retrieval["index_text"] = text
    return event, text, vector


@dataclass(frozen=True)
class Variant:
    group: str
    name: str
    env: dict[str, str] = field(default_factory=dict)
    hot_size: int = 2000

    @property
    def slug(self) -> str:
        raw = f"{self.group}-{self.name}".lower()
        return "".join(char if char.isalnum() else "-" for char in raw).strip("-")


class RetentionManager:
    VALID_MODES = {"full", "metrics-only"}

    def __init__(self, run_dir: Path, mode: str, disk_floor_gb: float = 10):
        if mode not in self.VALID_MODES:
            raise ValueError(f"unsupported retention mode: {mode}")
        if disk_floor_gb <= 0:
            raise ValueError("disk_floor_gb must be positive")
        self.run_dir = run_dir
        self.mode = mode
        self.disk_floor_gb = float(disk_floor_gb)

    def ensure_capacity(self, stage: str) -> None:
        if self.mode != "metrics-only":
            return
        free_bytes = shutil.disk_usage(self.run_dir).free
        floor_bytes = int(self.disk_floor_gb * 1024**3)
        if free_bytes < floor_bytes:
            raise RuntimeError(
                f"disk safety floor breached during {stage}: "
                f"{free_bytes / 1024**3:.2f} GB free < {self.disk_floor_gb:g} GB")

    def prepare_variant(self, variant: Variant) -> None:
        variant_dir = self._variant_dir(variant)
        if (
            (variant_dir / "result_checkpoint.json").exists()
            or (variant_dir / "METRICS_RETAINED").exists()
        ):
            raise RuntimeError(
                f"refusing to prepare checkpointed variant {variant.slug}")
        self.ensure_capacity(f"{variant.slug} start")
        log(f"clearing incomplete S3 prefix: {variant.slug}")
        self._delete_s3_prefix(variant)

    def record_variant(self, variant: Variant, module: str, fields: list[str],
                       row: dict[str, Any]) -> None:
        validate_result_row(fields, row, f"{variant.slug} checkpoint")
        variant_dir = self._variant_dir(variant)
        checkpoint = {
            "schema_version": "variant-result-checkpoint-v1",
            "module": module,
            "variant_slug": variant.slug,
            "variant_name": variant.name,
            "fields": fields,
            "row": row,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(variant_dir / "result_checkpoint.json", checkpoint)
        if self.mode == "metrics-only":
            self.cleanup_variant(
                variant, self._required_metric_files(module))

    def load_variant_row(self, variant: Variant, module: str,
                         fields: list[str]) -> dict[str, Any] | None:
        path = self._variant_dir(variant) / "result_checkpoint.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "variant-result-checkpoint-v1":
            raise RuntimeError(f"unsupported checkpoint schema in {path}")
        if payload.get("module") != module or payload.get("variant_slug") != variant.slug:
            raise RuntimeError(f"checkpoint identity mismatch in {path}")
        if payload.get("fields") != fields:
            raise RuntimeError(f"checkpoint fields mismatch in {path}")
        row = payload.get("row")
        if not isinstance(row, dict):
            raise RuntimeError(f"checkpoint row is invalid in {path}")
        validate_result_row(fields, row, f"{variant.slug} checkpoint")
        if self.mode == "metrics-only":
            self.cleanup_variant(
                variant, self._required_metric_files(module))
        return row

    def cleanup_variant(self, variant: Variant,
                        required_files: Iterable[str] = ()) -> None:
        if self.mode != "metrics-only":
            return
        variant_dir = self._variant_dir(variant)
        data_dir = variant_dir / "data"
        marker = variant_dir / "METRICS_RETAINED"
        if marker.exists() and not data_dir.exists():
            return
        missing = [
            name for name in required_files
            if not (variant_dir / name).is_file()
        ]
        if missing:
            raise RuntimeError(
                f"{variant.slug} missing retained metric artifacts: {missing}")
        reclaimed_bytes = self._directory_size(data_dir)
        if data_dir.exists():
            shutil.rmtree(data_dir)
        self._delete_s3_prefix(variant)
        atomic_write_json(marker, {
            "status": "metrics-retained",
            "variant_slug": variant.slug,
            "local_data_removed": True,
            "s3_prefix_removed": True,
            "reclaimed_local_bytes": reclaimed_bytes,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })

    def summary(self) -> dict[str, Any]:
        markers = list((self.run_dir / "variants").glob("*/METRICS_RETAINED"))
        reclaimed_bytes = 0
        for marker in markers:
            payload = json.loads(marker.read_text(encoding="utf-8"))
            reclaimed_bytes += int(payload.get("reclaimed_local_bytes", 0))
        return {
            "retention_mode": self.mode,
            "retained_variant_count": len(markers),
            "reclaimed_local_bytes": reclaimed_bytes,
            "disk_floor_gb": self.disk_floor_gb,
        }

    def _variant_dir(self, variant: Variant) -> Path:
        return self.run_dir / "variants" / variant.slug

    @staticmethod
    def _required_metric_files(module: str) -> tuple[str, ...]:
        common = (
            "capabilities.json",
            "measurements.json",
            "common_metrics.json",
            "server.log",
        )
        module_files = {
            "wal": ("recovery.json",),
            "governance": ("governance_measurement.json",),
        }
        return common + module_files.get(module, ())

    @staticmethod
    def _directory_size(path: Path) -> int:
        if not path.exists():
            return 0
        allocated_bytes = 0
        for item in path.rglob("*"):
            if not item.is_file():
                continue
            stat = item.stat()
            allocated_bytes += (
                stat.st_blocks * 512
                if hasattr(stat, "st_blocks")
                else stat.st_size
            )
        return allocated_bytes

    def _delete_s3_prefix(self, variant: Variant) -> None:
        mc = shutil.which("mc")
        if not mc:
            raise RuntimeError("MinIO client 'mc' is required for metrics-only cleanup")
        target = (
            f"plasmod-ablation/{DEFAULT_BUCKET}/agent-native-ablation/"
            f"{self.run_dir.name}/{variant.slug}"
        )
        completed = subprocess.run(
            [mc, "rm", "--recursive", "--force", target],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if completed.returncode != 0:
            raise RuntimeError(
                f"failed to remove MinIO prefix for {variant.slug}: "
                f"{completed.stderr.strip()}")


class MinioManager:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.process: subprocess.Popen[str] | None = None
        self.preexisting = False

    def start(self) -> None:
        try:
            http_json("http://127.0.0.1:9000", "GET", "/minio/health/live", timeout=2)
            self.preexisting = True
            log("MinIO already healthy at 127.0.0.1:9000")
        except Exception:
            binary = shutil.which("minio")
            if not binary:
                raise RuntimeError("MinIO is required but the minio binary is not installed")
            data_dir = self.run_dir / "minio-data"
            data_dir.mkdir(parents=True, exist_ok=True)
            log_path = self.run_dir / "minio.log"
            handle = log_path.open("a", encoding="utf-8")
            self.process = subprocess.Popen(
                [binary, "server", str(data_dir), "--address", ":9000", "--console-address", ":9001"],
                stdout=handle, stderr=subprocess.STDOUT, text=True,
            )
            deadline = time.time() + 30
            while time.time() < deadline:
                try:
                    http_json("http://127.0.0.1:9000", "GET", "/minio/health/live", timeout=1)
                    break
                except Exception:
                    if self.process.poll() is not None:
                        raise RuntimeError(f"MinIO exited; inspect {log_path}")
                    time.sleep(0.25)
            else:
                raise RuntimeError(f"MinIO health check timed out; inspect {log_path}")
        mc = shutil.which("mc")
        if not mc:
            raise RuntimeError("MinIO client 'mc' is required to create the experiment bucket")
        subprocess.run([mc, "alias", "set", "plasmod-ablation", "http://127.0.0.1:9000",
                        "minioadmin", "minioadmin"], check=True, stdout=subprocess.DEVNULL)
        subprocess.run([mc, "mb", "--ignore-existing", f"plasmod-ablation/{DEFAULT_BUCKET}"],
                       check=True, stdout=subprocess.DEVNULL)

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(10)
            except subprocess.TimeoutExpired:
                self.process.kill()


class PlasmodProcess:
    def __init__(self, variant: Variant, run_dir: Path, port: int):
        self.variant = variant
        self.run_dir = run_dir
        self.port = port
        self.base = f"http://127.0.0.1:{port}"
        self.process: subprocess.Popen[str] | None = None
        self.log_handle = None
        self.data_dir = run_dir / "variants" / variant.slug / "data"
        self.variant_dir = self.data_dir.parent

    def start(self, fresh: bool) -> None:
        if fresh and self.variant_dir.exists():
            shutil.rmtree(self.variant_dir)
        self.variant_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.log_handle = (self.variant_dir / "server.log").open("a", encoding="utf-8")
        env = os.environ.copy()
        env.update({
            "APP_MODE": "dev",
            "PLASMOD_LISTEN_MODE": "unified",
            "PLASMOD_HTTP_ADDR": f"127.0.0.1:{self.port}",
            "PLASMOD_GRPC_ENABLED": "0",
            "PLASMOD_STORAGE": "disk",
            "PLASMOD_DATA_DIR": str(self.data_dir),
            "PLASMOD_EMBEDDER": "tfidf",
            "PLASMOD_EMBEDDER_DIM": "384",
            "PLASMOD_FLUSH_INTERVAL": "0",
            "PLASMOD_CONSISTENCY_DEFAULT_MODE": "strict",
            "PLASMOD_CONSISTENCY_CHECKPOINT_FLUSH_INTERVAL": "50ms",
            "PLASMOD_HOT_CACHE_SIZE": str(self.variant.hot_size),
            "S3_ENDPOINT": "127.0.0.1:9000",
            "S3_ACCESS_KEY": "minioadmin",
            "S3_SECRET_KEY": "minioadmin",
            "S3_BUCKET": DEFAULT_BUCKET,
            "S3_SECURE": "false",
            "S3_REGION": "us-east-1",
            "S3_PREFIX": f"agent-native-ablation/{self.run_dir.name}/{self.variant.slug}",
            "LD_LIBRARY_PATH": f"{CORE / 'cpp/build'}:{CORE / 'cpp/build/vendor'}:{env.get('LD_LIBRARY_PATH', '')}",
            "DYLD_LIBRARY_PATH": f"{CORE / 'cpp/build'}:{CORE / 'cpp/build/vendor'}:{env.get('DYLD_LIBRARY_PATH', '')}",
        })
        env.update(self.variant.env)
        self.process = subprocess.Popen(
            [str(CORE / "bin" / "plasmod")], cwd=CORE, env=env,
            stdout=self.log_handle, stderr=subprocess.STDOUT, text=True,
        )
        (self.variant_dir / "server.pid").write_text(str(self.process.pid), encoding="utf-8")
        start_timeout_s = plasmod_start_timeout_s()
        deadline = time.time() + start_timeout_s
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"Plasmod exited for {self.variant.name}; inspect {self.variant_dir / 'server.log'}")
            try:
                health = http_json(self.base, "GET", "/healthz", timeout=1)
                if health.get("status") == "ok":
                    break
            except Exception:
                time.sleep(0.25)
        else:
            raise RuntimeError(
                f"Plasmod health timeout for {self.variant.name} after {start_timeout_s:.1f}s; "
                f"inspect {self.variant_dir / 'server.log'}"
            )
        capabilities = http_json(self.base, "GET", "/v1/admin/capabilities")
        active = capabilities.get("capabilities") or {}
        for env_name, field_name in CAPABILITY_ENV_FIELDS.items():
            expected = self.variant.env.get(env_name)
            if expected is not None and str(active.get(field_name)) != str(expected):
                raise RuntimeError(
                    f"{self.variant.name} capability mismatch: {field_name}={active.get(field_name)!r}, "
                    f"expected {expected!r}"
                )
        if "PLASMOD_RECOVERY_REPLAY" in self.variant.env:
            expected_replay = self.variant.env["PLASMOD_RECOVERY_REPLAY"].lower() in ("1", "true", "yes", "on")
            if active.get("recovery_replay") is not expected_replay:
                raise RuntimeError(
                    f"{self.variant.name} recovery_replay={active.get('recovery_replay')!r}, "
                    f"expected {expected_replay!r}"
                )
        if int(active.get("hot_cache_size", 0)) != self.variant.hot_size:
            raise RuntimeError(
                f"{self.variant.name} hot_cache_size={active.get('hot_cache_size')!r}, "
                f"expected {self.variant.hot_size}"
            )
        (self.variant_dir / "capabilities.json").write_text(
            json.dumps(capabilities, indent=2, ensure_ascii=False), encoding="utf-8")

    def offline_reset_materialized_state(self) -> None:
        if not self.data_dir.exists():
            return
        for path in self.data_dir.iterdir():
            if path.name == "wal.log":
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(20)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(5)
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None

    def restart(self) -> None:
        self.stop()
        self.start(fresh=False)

    def rss_mb(self) -> float:
        if self.process is None or self.process.poll() is not None:
            return 0.0
        output = subprocess.check_output(["ps", "-o", "rss=", "-p", str(self.process.pid)], text=True).strip()
        return float(output or 0) / 1024.0


@dataclass
class RunData:
    writes: int = 0
    write_latencies: list[float] = field(default_factory=list)
    visibility_latencies: list[float] = field(default_factory=list)
    materialization_latencies: list[float] = field(default_factory=list)
    query_latencies: list[float] = field(default_factory=list)
    evidence_latencies: list[float] = field(default_factory=list)
    promotion_latencies: list[float] = field(default_factory=list)
    responses: list[dict[str, Any]] = field(default_factory=list)
    event_ids: set[str] = field(default_factory=set)
    memory_ids: set[str] = field(default_factory=set)
    state_ids: set[str] = field(default_factory=set)
    artifact_ids: set[str] = field(default_factory=set)
    edge_ids: set[str] = field(default_factory=set)
    contexts: dict[str, tuple[str, str, str]] = field(default_factory=dict)
    query_samples: list[tuple[str, list[float], str, str, str, str]] = field(default_factory=list)
    latest_query_samples: list[tuple[str, list[float], str, str, str, str]] = field(default_factory=list)
    wall_seconds: float = 0.0
    state: dict[str, Any] = field(default_factory=dict)
    memory_mb: float = 0.0
    stale_checks: int = 0
    stale_misses: int = 0

    @property
    def object_ids(self) -> set[str]:
        return self.event_ids | self.memory_ids | self.state_ids | self.artifact_ids


@dataclass
class SharedFullBaseline:
    module_rows: dict[str, dict[str, Any]]
    materialization_type_counts: dict[str, int]
    state_ids: set[str]
    artifact_ids: set[str]
    contexts: dict[str, tuple[str, str, str]]
    evidence_totals: tuple[int, int, int, int]
    governance_measurement: dict[str, Any]

    def write(self, path: Path) -> None:
        payload = {
            "schema_version": "shared-full-baseline-v1",
            "module_rows": self.module_rows,
            "materialization_type_counts": self.materialization_type_counts,
            "state_ids": sorted(self.state_ids),
            "artifact_ids": sorted(self.artifact_ids),
            "contexts": {key: list(value) for key, value in sorted(self.contexts.items())},
            "evidence_totals": list(self.evidence_totals),
            "governance_measurement": self.governance_measurement,
        }
        atomic_write_json(path, payload)

    @classmethod
    def read(cls, path: Path) -> "SharedFullBaseline":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "shared-full-baseline-v1":
            raise RuntimeError(f"unsupported shared Full baseline: {path}")
        return cls(
            module_rows=payload["module_rows"],
            materialization_type_counts={
                key: int(value) for key, value in payload["materialization_type_counts"].items()
            },
            state_ids=set(payload["state_ids"]),
            artifact_ids=set(payload["artifact_ids"]),
            contexts={key: tuple(value) for key, value in payload["contexts"].items()},
            evidence_totals=tuple(int(value) for value in payload["evidence_totals"]),
            governance_measurement=payload["governance_measurement"],
        )


def common_metric_row(data: RunData) -> dict[str, Any]:
    query_seconds = sum(data.query_latencies) / 1000.0
    visible = data.stale_checks - data.stale_misses
    return {
        "Common | Event Count": data.writes,
        "Common | Query Count": len(data.query_latencies),
        "Common | Stale Check Count": data.stale_checks,
        "Common | TopK": 20,
        "Common | Embedding Dimension": 384,
        "Common | Write QPS": round(data.writes / max(data.wall_seconds, 1e-9), 6),
        "Common | Write p50 (ms)": round(percentile(data.write_latencies, 0.50), 6),
        "Common | Write p95 (ms)": round(percentile(data.write_latencies, 0.95), 6),
        "Common | Write p99 (ms)": round(percentile(data.write_latencies, 0.99), 6),
        "Common | Write-to-Visible p50 (ms)": round(percentile(data.visibility_latencies, 0.50), 6),
        "Common | Write-to-Visible p95 (ms)": round(percentile(data.visibility_latencies, 0.95), 6),
        "Common | Materialization Lag p95 (ms)": round(
            percentile(data.materialization_latencies, 0.95), 6),
        "Common | Query QPS": round(len(data.query_latencies) / max(query_seconds, 1e-9), 6),
        "Common | Query p50 (ms)": round(percentile(data.query_latencies, 0.50), 6),
        "Common | Query p95 (ms)": round(percentile(data.query_latencies, 0.95), 6),
        "Common | Query p99 (ms)": round(percentile(data.query_latencies, 0.99), 6),
        "Common | Memory (MB)": round(data.memory_mb, 6),
        "Common | Object Visibility Coverage (%)": round(pct(visible, data.stale_checks), 6),
        "Common | Target Stale Rate (%)": round(pct(data.stale_misses, data.stale_checks), 6),
    }


def write_common_metrics(server: PlasmodProcess, variant: Variant, data: RunData) -> None:
    payload = {
        "system": "Plasmod",
        "module": variant.group,
        "variant": variant.name,
        "parameter_set": COMMON_PARAMETER_SET,
        "metrics": common_metric_row(data),
    }
    (server.variant_dir / "common_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def query(base: str, text: str, vector: list[float], target_ids: list[str] | None = None,
          response_mode: str = "structured_evidence", requester: str = "", include_cold: bool = False,
          workspace: str = "plasmod-ablation", session: str = "") -> tuple[dict[str, Any], float]:
    body = {
        "query_text": text,
        "query_scope": workspace,
        "workspace_id": workspace,
        "tenant_id": "default",
        "session_id": session,
        "requester_agent_id": requester,
        # requester_agent_id drives governance. agent_id is a content-owner
        # selector and would incorrectly hide cross-agent sharing results.
        "agent_id": "",
        "top_k": 20,
        "response_mode": response_mode,
        "embedding_vector": vector,
        "include_cold": include_cold,
        "access_consistency": "eventual",
    }
    if target_ids:
        body["target_object_ids"] = target_ids
    started = time.perf_counter()
    response = http_json(base, "POST", "/v1/query", body)
    return response, (time.perf_counter() - started) * 1000


def ingest_workload(server: PlasmodProcess, variant: Variant, event_limit: int,
                    query_limit: int,
                    retention: RetentionManager | None = None) -> RunData:
    data = RunData()
    latest_by_scope: dict[tuple[str, str, str], tuple[str, list[float], str, str, str, str]] = {}
    if retention is not None:
        retention.ensure_capacity(f"{variant.slug} ingest start")
    started_all = time.perf_counter()
    for ordinal, source in enumerate(iter_events(event_limit), 1):
        event, text, vector = prepare_event(source, ordinal, variant.group)
        started = time.perf_counter()
        ack = http_json(server.base, "POST", "/v1/ingest/events", event, timeout=120)
        write_ms = (time.perf_counter() - started) * 1000
        if ack.get("status") not in ("accepted", "duplicate"):
            raise RuntimeError(f"{variant.name} ingest {ordinal} failed: {ack}")
        data.writes += 1
        data.write_latencies.append(write_ms)
        data.materialization_latencies.append(require_number(
            ack.get("materialization_latency_ms", 0), "materialization_latency_ms"))
        event_id = str(ack["event_id"])
        data.event_ids.add(event_id)
        if ack.get("memory_id"):
            data.memory_ids.add(str(ack["memory_id"]))
        if ack.get("state_id"):
            data.state_ids.add(str(ack["state_id"]))
        if ack.get("artifact_id"):
            data.artifact_ids.add(str(ack["artifact_id"]))
        data.edge_ids.update(str(item) for item in ack.get("edge_ids", []))
        target = str(ack.get("retrieval_object_id") or ack.get("memory_id") or event_id)
        requester = str((event.get("actor") or {}).get("agent_id") or "")
        workspace = str((event.get("identity") or {}).get("workspace_id") or "plasmod-ablation")
        session = str((event.get("actor") or {}).get("session_id") or "")
        for object_id in (
            event_id, ack.get("memory_id"), ack.get("state_id"), ack.get("artifact_id"),
            *ack.get("edge_ids", []),
        ):
            if object_id:
                data.contexts[str(object_id)] = (requester, workspace, session)
        visible_started = time.perf_counter()
        visibility_response, _ = query(
            server.base, text, vector, [target], "objects_only",
            requester=requester, workspace=workspace, session=session,
        )
        data.visibility_latencies.append((time.perf_counter() - visible_started) * 1000)
        if target not in visibility_response.get("objects", []):
            raise RuntimeError(f"{variant.name} accepted object {target} was not query-visible")
        if len(data.query_samples) < query_limit:
            data.query_samples.append((text, vector, target, requester, workspace, session))
        scope_key = (requester, workspace, session)
        latest_by_scope.pop(scope_key, None)
        latest_by_scope[scope_key] = (text, vector, target, requester, workspace, session)
        if retention is not None and ordinal % 100 == 0:
            retention.ensure_capacity(f"{variant.slug} ingest event {ordinal}")
        if ordinal == 1 or ordinal % 1000 == 0:
            log(f"{variant.group}/{variant.name}: ingested {ordinal}")
    data.wall_seconds = time.perf_counter() - started_all
    data.latest_query_samples = list(reversed(latest_by_scope.values()))[:query_limit]
    for text, vector, _, requester, workspace, session in data.query_samples:
        response, latency = query(
            server.base, text, vector, requester=requester, workspace=workspace, session=session)
        data.responses.append(response)
        data.query_latencies.append(latency)
        diagnostics = response.get("diagnostics") or {}
        data.evidence_latencies.append(require_number(
            diagnostics.get("evidence_assembly_latency_ms", 0), "evidence latency"))
        data.promotion_latencies.append(require_number(
            diagnostics.get("promotion_latency_ms", 0), "promotion latency"))
    for text, vector, expected, requester, workspace, session in data.latest_query_samples:
        response, _ = query(
            server.base, text, vector, [expected], "objects_only",
            requester=requester, workspace=workspace, session=session,
        )
        data.stale_checks += 1
        data.stale_misses += int(expected not in response.get("objects", []))
    state = http_json(server.base, "GET", "/v1/admin/runtime/state")
    data.state = state.get("state") or {}
    data.memory_mb = server.rss_mb()
    raw = {
        "variant": variant.name,
        "writes": data.writes,
        "write_latencies_ms": data.write_latencies,
        "visibility_latencies_ms": data.visibility_latencies,
        "materialization_latencies_ms": data.materialization_latencies,
        "query_latencies_ms": data.query_latencies,
        "evidence_latencies_ms": data.evidence_latencies,
        "promotion_latencies_ms": data.promotion_latencies,
        "state": data.state,
        "memory_mb": data.memory_mb,
        "stale_checks": data.stale_checks,
        "stale_misses": data.stale_misses,
        "sample_responses": data.responses[:3],
    }
    (server.variant_dir / "measurements.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    write_common_metrics(server, variant, data)
    if retention is not None:
        retention.ensure_capacity(f"{variant.slug} measurement complete")
    return data


def run_once(variant: Variant, run_dir: Path, port: int, event_limit: int,
             query_limit: int, retention: RetentionManager | None = None) -> RunData:
    server = PlasmodProcess(variant, run_dir, port)
    try:
        server.start(fresh=True)
        return ingest_workload(
            server, variant, event_limit, query_limit, retention)
    finally:
        server.stop()


def recovery_variants() -> list[Variant]:
    return [
        Variant("wal", "Full Plasmod", {"PLASMOD_WAL_MODE": "file", "PLASMOD_RECOVERY_REPLAY": "true"}),
        Variant("wal", "No-WAL", {"PLASMOD_WAL_MODE": "disabled", "PLASMOD_RECOVERY_REPLAY": "false"}),
        Variant("wal", "In-memory WAL", {"PLASMOD_WAL_MODE": "memory", "PLASMOD_RECOVERY_REPLAY": "true"}),
        Variant("wal", "File WAL", {"PLASMOD_WAL_MODE": "file", "PLASMOD_RECOVERY_REPLAY": "true"}),
        Variant("wal", "WAL without replay", {"PLASMOD_WAL_MODE": "file", "PLASMOD_RECOVERY_REPLAY": "false"}),
        Variant("wal", "Replay without index rebuild", {
            "PLASMOD_WAL_MODE": "file", "PLASMOD_RECOVERY_REPLAY": "true",
            "PLASMOD_RECOVERY_PROJECTION": "canonical_only",
        }),
    ]


def measure_recovery(server: PlasmodProcess, variant: Variant, before: RunData,
                     output_variant: str | None = None) -> dict[str, Any]:
    server.restart()
    reset_timeout_s = recovery_reset_timeout_s(before.writes)
    log(
        f"{variant.group}/{variant.name}: resetting materialized state for "
        f"{before.writes} WAL entries with timeout {reset_timeout_s:.0f}s"
    )
    http_json(
        server.base, "POST", "/v1/admin/recovery/reset",
        {"confirm": "reset_materialized"}, timeout=reset_timeout_s)
    replay_response: dict[str, Any] = {}
    query_available = False
    replay_wall = 0.0
    replay_enabled = variant.env.get("PLASMOD_RECOVERY_REPLAY", "true").lower() in (
        "1", "true", "yes", "on")
    if replay_enabled:
        replay_timeout_s = recovery_replay_timeout_s(before.writes)
        log(
            f"{variant.group}/{variant.name}: replaying {before.writes} WAL entries "
            f"with timeout {replay_timeout_s:.0f}s"
        )
        result_holder: dict[str, Any] = {}
        error_holder: list[BaseException] = []

        def apply_replay() -> None:
            try:
                started = time.perf_counter()
                result_holder.update(http_json(server.base, "POST", "/v1/admin/replay", {
                    "from_lsn": 0, "limit": 0, "apply": True, "confirm": "apply_replay",
                }, timeout=replay_timeout_s))
                result_holder["_wall"] = time.perf_counter() - started
            except BaseException as exc:  # propagated below
                error_holder.append(exc)

        thread = threading.Thread(target=apply_replay, daemon=True)
        thread.start()
        sample = next(iter(before.query_samples), (
            "state", hash_vector("state"), "", "", "plasmod-ablation", ""))
        try:
            query(server.base, sample[0], sample[1], requester=sample[3],
                  workspace=sample[4], session=sample[5])
            query_available = True
        except Exception:
            query_available = False
        thread.join(replay_timeout_s)
        if thread.is_alive():
            raise RuntimeError(
                f"replay timed out for {variant.name} after {replay_timeout_s:.0f}s"
            )
        if error_holder:
            raise RuntimeError(f"replay failed for {variant.name}: {error_holder[0]}")
        replay_response = result_holder
        replay_wall = float(result_holder.get("_wall", 0))
    after_state = replay_response.get("state") if isinstance(replay_response, dict) else None
    if isinstance(after_state, dict):
        after = after_state
    else:
        after = (http_json(
            server.base, "GET", "/v1/admin/runtime/state",
            timeout=recovery_replay_timeout_s(before.writes)).get("state") or {})
    row = {
        "System": "Plasmod", "Variant": output_variant or variant.name,
        "Event Log Size": int(replay_response.get("scanned_entries", 0)),
        "Recovered Objects (%)": round(pct(int(after.get("objects", 0)), len(before.object_ids)), 6),
        "Recovered Relations (%)": round(pct(int(after.get("edges", 0)), len(before.edge_ids)), 6),
        "Recovered Latest State (%)": round(pct(int(after.get("latest_states", 0)), len(before.state_ids)), 6),
        "Recovery Time (s)": round(float(
            replay_response.get("recovery_time_ms", replay_wall * 1000)) / 1000, 6),
        "Replay Throughput (events/s)": round(float(
            replay_response.get("replay_throughput_events_s", 0)), 6),
        "Query Available During Recovery": "yes" if query_available else "no",
        "Lost Event Count": max(0, len(before.event_ids) - int(after.get("events", 0))),
        "Duplicate Object Count": int(replay_response.get("duplicate_objects", 0)),
    }
    (server.variant_dir / "recovery.json").write_text(json.dumps({
        "before": before.state, "replay": replay_response, "after": after, "row": row,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return row


def run_recovery(run_dir: Path, port: int, event_limit: int, query_limit: int,
                 baseline: SharedFullBaseline,
                 retention: RetentionManager) -> list[dict[str, Any]]:
    rows = [baseline.module_rows["wal"]]
    for variant in recovery_variants()[1:]:
        checkpoint = retention.load_variant_row(variant, "wal", WAL_FIELDS)
        if checkpoint is not None:
            log(f"reusing checkpointed WAL variant: {variant.name}")
            rows.append(checkpoint)
            continue
        log(f"starting WAL variant: {variant.name}")
        retention.prepare_variant(variant)
        server = PlasmodProcess(variant, run_dir, port)
        row: dict[str, Any] | None = None
        try:
            server.start(fresh=True)
            before = ingest_workload(
                server, variant, event_limit, query_limit, retention)
            row = measure_recovery(server, variant, before)
        finally:
            server.stop()
        if row is None:
            raise RuntimeError(f"{variant.name} did not produce a WAL result")
        retention.record_variant(variant, "wal", WAL_FIELDS, row)
        rows.append(row)
    return rows


def materialization_variants() -> list[Variant]:
    return [
        Variant("materialization", "Full Plasmod", {"PLASMOD_MATERIALIZATION_PROFILE": "full"}),
        Variant("materialization", "No-materialization", {"PLASMOD_MATERIALIZATION_PROFILE": "none"}),
        Variant("materialization", "Memory-only", {"PLASMOD_MATERIALIZATION_PROFILE": "memory_only"}),
        Variant("materialization", "No-agent-state", {"PLASMOD_MATERIALIZATION_PROFILE": "no_state"}),
        Variant("materialization", "No-artifact", {"PLASMOD_MATERIALIZATION_PROFILE": "no_artifact"}),
        Variant("materialization", "No-edge", {"PLASMOD_MATERIALIZATION_PROFILE": "no_edge"}),
        Variant("materialization", "No-object-version", {"PLASMOD_MATERIALIZATION_PROFILE": "no_version"}),
    ]


def target_hit_rate(base: str, ids: set[str], contexts: dict[str, tuple[str, str, str]]) -> float:
    if not ids:
        return 100.0
    hits = 0
    vector = hash_vector("canonical object lookup")
    for object_id in ids:
        requester, workspace, session = contexts.get(object_id, ("", "plasmod-ablation", ""))
        response, _ = query(
            base, "canonical object lookup", vector, [object_id], "objects_only",
            requester=requester, workspace=workspace, session=session)
        hits += int(object_id in response.get("objects", []))
    return pct(hits, len(ids))


def materialization_type_counts(data: RunData) -> dict[str, int]:
    return {
        "events": int(data.state.get("events", 0)),
        "memories": int(data.state.get("memories", 0)),
        "states": int(data.state.get("states", 0)),
        "artifacts": int(data.state.get("artifacts", 0)),
        "edges": int(data.state.get("edges", 0)),
        "versions": int(data.state.get("versions", 0)),
    }


def measure_materialization(server: PlasmodProcess, variant: Variant, data: RunData,
                            baseline: SharedFullBaseline,
                            output_variant: str | None = None) -> dict[str, Any]:
    state_hit = target_hit_rate(server.base, baseline.state_ids, baseline.contexts)
    artifact_hit = target_hit_rate(server.base, baseline.artifact_ids, baseline.contexts)
    baseline_objects = sum(baseline.materialization_type_counts.values())
    visible_objects = sum(
        min(int(data.state.get(key, 0)), baseline_count)
        for key, baseline_count in baseline.materialization_type_counts.items()
    )
    baseline_edges = baseline.materialization_type_counts.get("edges", 0)
    relation_rate = pct(min(int(data.state.get("edges", 0)), baseline_edges), baseline_edges)
    return {
        "System": "Plasmod", "Variant": output_variant or variant.name,
        "Write QPS": round(data.writes / max(data.wall_seconds, 1e-9), 6),
        "Write p95 (ms)": round(percentile(data.write_latencies, 0.95), 6),
        "Write-to-Visible p95 (ms)": round(percentile(data.visibility_latencies, 0.95), 6),
        "Materialization Lag p95 (ms)": round(
            percentile(data.materialization_latencies, 0.95), 6),
        "Object Visibility Coverage (%)": round(pct(visible_objects, baseline_objects), 6),
        "Latest-state Hit Rate (%)": round(state_hit, 6),
        "Artifact Lookup Accuracy (%)": round(artifact_hit, 6),
        "Relation Recovery Rate (%)": round(relation_rate, 6),
        "Stale Result Rate (%)": round(100.0 - state_hit, 6),
    }


def run_materialization(run_dir: Path, port: int, event_limit: int, query_limit: int,
                        baseline: SharedFullBaseline,
                        retention: RetentionManager) -> list[dict[str, Any]]:
    rows = [baseline.module_rows["materialization"]]
    for variant in materialization_variants()[1:]:
        checkpoint = retention.load_variant_row(
            variant, "materialization", MATERIALIZATION_FIELDS)
        if checkpoint is not None:
            log(f"reusing checkpointed materialization variant: {variant.name}")
            rows.append(checkpoint)
            continue
        log(f"starting materialization variant: {variant.name}")
        retention.prepare_variant(variant)
        server = PlasmodProcess(variant, run_dir, port)
        row: dict[str, Any] | None = None
        try:
            server.start(fresh=True)
            data = ingest_workload(
                server, variant, event_limit, query_limit, retention)
            row = measure_materialization(server, variant, data, baseline)
        finally:
            server.stop()
        if row is None:
            raise RuntimeError(
                f"{variant.name} did not produce a materialization result")
        retention.record_variant(
            variant, "materialization", MATERIALIZATION_FIELDS, row)
        rows.append(row)
    return rows


def evidence_variants() -> list[Variant]:
    return [
        Variant("evidence", "Full Plasmod", {"PLASMOD_EVIDENCE_PROFILE": "full"}),
        Variant("evidence", "No-evidence", {"PLASMOD_EVIDENCE_PROFILE": "none"}),
        Variant("evidence", "No-provenance", {"PLASMOD_EVIDENCE_PROFILE": "no_provenance"}),
        Variant("evidence", "No-edge-expansion", {"PLASMOD_EVIDENCE_PROFILE": "no_edge_expansion"}),
        Variant("evidence", "One-hop only", {"PLASMOD_EVIDENCE_PROFILE": "one_hop"}),
        Variant("evidence", "No-proof-trace", {"PLASMOD_EVIDENCE_PROFILE": "no_proof"}),
        Variant("evidence", "Vector-only", {"PLASMOD_EVIDENCE_PROFILE": "vector_only"}),
    ]


def evidence_totals(data: RunData) -> tuple[int, int, int, int]:
    provenance = sum(len(response.get("provenance") or []) for response in data.responses)
    edges = sum(len(response.get("edges") or []) for response in data.responses)
    proof = sum(len(response.get("proof_trace") or []) for response in data.responses)
    supported = sum(bool(response.get("nodes") or response.get("edges") or response.get("proof_trace")
                         or response.get("provenance")) for response in data.responses)
    return provenance, edges, proof, supported


def measure_evidence(variant: Variant, data: RunData,
                     baseline_totals: tuple[int, int, int, int],
                     output_variant: str | None = None) -> dict[str, Any]:
    current = evidence_totals(data)
    correctness = pct(current[3], len(data.responses))
    return {
        "System": "Plasmod", "Variant": output_variant or variant.name,
        "Query p95 (ms)": round(percentile(data.query_latencies, 0.95), 6),
        "Evidence Assembly Latency p95 (ms)": round(
            percentile(data.evidence_latencies, 0.95), 6),
        "Provenance Completeness (%)": round(pct(current[0], baseline_totals[0]), 6),
        "Edge Recall (%)": round(pct(current[1], baseline_totals[1]), 6),
        "Proof Completeness (%)": round(pct(current[2], baseline_totals[2]), 6),
        "Citation / Evidence Correctness (%)": round(correctness, 6),
        "Stale Evidence Rate (%)": round(100.0 - correctness, 6),
    }


def run_evidence(run_dir: Path, port: int, event_limit: int, query_limit: int,
                 baseline: SharedFullBaseline,
                 retention: RetentionManager) -> list[dict[str, Any]]:
    rows = [baseline.module_rows["evidence"]]
    for variant in evidence_variants()[1:]:
        checkpoint = retention.load_variant_row(
            variant, "evidence", EVIDENCE_FIELDS)
        if checkpoint is not None:
            log(f"reusing checkpointed evidence variant: {variant.name}")
            rows.append(checkpoint)
            continue
        log(f"starting evidence variant: {variant.name}")
        retention.prepare_variant(variant)
        data = run_once(
            variant, run_dir, port, event_limit, query_limit, retention)
        row = measure_evidence(variant, data, baseline.evidence_totals)
        retention.record_variant(variant, "evidence", EVIDENCE_FIELDS, row)
        rows.append(row)
    return rows


def governance_event(kind: str, event_id: str, owner: str, visibility: str,
                     policy_tags: list[str] | None = None, contract: str = "") -> dict[str, Any]:
    return {
        "schema_version": "plasmod_dynamic_event_v0.4",
        "identity": {"event_id": event_id, "tenant_id": "default", "workspace_id": "plasmod-ablation"},
        "actor": {"agent_id": owner, "session_id": f"governance-{kind}", "team_id": "team-a"},
        "time": {"event_time": int(time.time() * 1000)},
        "event": {"event_type": "memory", "action": "created"},
        "object": {"object_type": "memory", "lifecycle_state": "active"},
        "access": {
            "consistency": "strict", "visibility": visibility,
            "policy_tags": policy_tags or [], "share_contract_id": contract,
        },
        "retrieval": {"index_text": f"governance {kind} sentinel"},
        "payload": {"content": f"governance {kind} sentinel"},
    }


def governance_variants() -> list[Variant]:
    return [
        Variant("governance", "Full Plasmod", {"PLASMOD_GOVERNANCE_PROFILE": "full"}),
        Variant("governance", "No-access-policy", {"PLASMOD_GOVERNANCE_PROFILE": "no_access"}),
        Variant("governance", "Metadata-filter-only", {"PLASMOD_GOVERNANCE_PROFILE": "metadata_only"}),
        Variant("governance", "No-share-contract", {"PLASMOD_GOVERNANCE_PROFILE": "no_share_contract"}),
        Variant("governance", "No-quarantine", {"PLASMOD_GOVERNANCE_PROFILE": "no_quarantine"}),
        Variant("governance", "No-delete-propagation", {"PLASMOD_GOVERNANCE_PROFILE": "no_delete_propagation"}),
    ]


def measure_governance(server: PlasmodProcess, variant: Variant) -> dict[str, Any]:
    contract_id = f"contract-governance-{variant.slug}"
    http_json(server.base, "POST", "/v1/share-contracts", {
        "contract_id": contract_id, "tenant_id": "default",
        "workspace_id": "plasmod-ablation", "scope": "plasmod-ablation",
        "read_agents": ["agent-b"], "read_acl": "agent:agent-b",
    })
    specs = [
        ("private", "private", [], ""),
        ("shared", "restricted_shared", [], contract_id),
        ("quarantine", "workspace", ["quarantine"], ""),
        ("deleted", "workspace", [], ""),
    ]
    ids: dict[str, str] = {}
    for ordinal, (kind, visibility, tags, contract) in enumerate(specs, 1):
        source = governance_event(
            kind, f"{variant.slug}-{kind}", "agent-a", visibility, tags, contract)
        event, _, _ = prepare_event(source, ordinal, variant.slug)
        ack = http_json(server.base, "POST", "/v1/ingest/events", event)
        ids[kind] = str(ack["memory_id"])
    qvec = hash_vector("governance sentinel")
    latencies = []

    def visible(kind: str, requester: str) -> bool:
        response, latency = query(
            server.base, f"governance {kind} sentinel", qvec, [ids[kind]],
            "objects_only", requester=requester, workspace="plasmod-ablation",
            session=f"governance-{kind}",
        )
        latencies.append(latency)
        return ids[kind] in response.get("objects", [])

    private_leak = visible("private", "agent-b")
    shared_hit = visible("shared", "agent-b")
    unauthorized = visible("shared", "agent-c")
    quarantine_excluded = not visible("quarantine", "agent-a")
    http_json(server.base, "POST", "/v1/admin/rollback", {
        "memory_id": ids["deleted"], "action": "deactivate", "reason": "governance measurement",
    })
    delete_started = time.perf_counter()
    delete_timeout_ms = float(os.getenv("PLASMOD_ABLATION_DELETE_TIMEOUT_MS", "500"))
    deleted_visible = True
    while deleted_visible:
        deleted_visible = visible("deleted", "agent-a")
        elapsed_ms = (time.perf_counter() - delete_started) * 1000
        if not deleted_visible or elapsed_ms >= delete_timeout_ms:
            break
        time.sleep(0.02)
    delete_delay = min((time.perf_counter() - delete_started) * 1000, delete_timeout_ms)
    return {
        "private": private_leak, "shared": shared_hit, "unauthorized": unauthorized,
        "quarantine": quarantine_excluded, "deleted_visible": deleted_visible,
        "delete_delay": delete_delay, "query_latency": sum(latencies) / max(len(latencies), 1),
    }


def governance_row(variant_name: str, value: dict[str, Any],
                   no_access_latency: float) -> dict[str, Any]:
    return {
        "System": "Plasmod", "Variant": variant_name,
        "Private Memory Leakage Rate (%)": 100.0 if value["private"] else 0.0,
        "Authorized Hit Rate (%)": 100.0 if value["shared"] else 0.0,
        "Unauthorized Hit Rate (%)": 100.0 if value["unauthorized"] else 0.0,
        "Delete Visibility Delay (ms)": round(float(value["delete_delay"]), 6),
        "Quarantine Exclusion Rate (%)": 100.0 if value["quarantine"] else 0.0,
        "Policy Overhead (ms)": round(max(
            0.0, float(value["query_latency"]) - no_access_latency), 6),
    }


def run_governance(run_dir: Path, port: int, event_limit: int,
                   query_limit: int, baseline: SharedFullBaseline,
                   retention: RetentionManager) -> list[dict[str, Any]]:
    variants = governance_variants()[1:]
    no_access_variant = variants[0]
    no_access_measurement_path = (
        run_dir / "variants" / no_access_variant.slug / "governance_measurement.json")
    no_access_row = retention.load_variant_row(
        no_access_variant, "governance", GOVERNANCE_FIELDS)
    if no_access_row is not None:
        if not no_access_measurement_path.exists():
            raise RuntimeError(
                f"missing governance measurement for checkpointed {no_access_variant.name}")
        no_access_measurement = json.loads(
            no_access_measurement_path.read_text(encoding="utf-8"))
        log(f"reusing checkpointed governance variant: {no_access_variant.name}")
    else:
        log(f"starting governance variant: {no_access_variant.name}")
        retention.prepare_variant(no_access_variant)
        server = PlasmodProcess(no_access_variant, run_dir, port)
        try:
            server.start(fresh=True)
            ingest_workload(
                server, no_access_variant, event_limit, query_limit, retention)
            no_access_measurement = measure_governance(server, no_access_variant)
            atomic_write_json(no_access_measurement_path, no_access_measurement)
        finally:
            server.stop()
        no_access_row = governance_row(
            no_access_variant.name,
            no_access_measurement,
            float(no_access_measurement["query_latency"]),
        )
        retention.record_variant(
            no_access_variant, "governance", GOVERNANCE_FIELDS, no_access_row)

    no_access_latency = float(no_access_measurement["query_latency"])
    rows = [
        governance_row(
            "Full Plasmod", baseline.governance_measurement, no_access_latency),
        no_access_row,
    ]
    for variant in variants[1:]:
        checkpoint = retention.load_variant_row(
            variant, "governance", GOVERNANCE_FIELDS)
        if checkpoint is not None:
            log(f"reusing checkpointed governance variant: {variant.name}")
            rows.append(checkpoint)
            continue
        log(f"starting governance variant: {variant.name}")
        retention.prepare_variant(variant)
        server = PlasmodProcess(variant, run_dir, port)
        measurement: dict[str, Any] | None = None
        try:
            server.start(fresh=True)
            ingest_workload(
                server, variant, event_limit, query_limit, retention)
            measurement = measure_governance(server, variant)
            atomic_write_json(
                run_dir / "variants" / variant.slug / "governance_measurement.json",
                measurement,
            )
        finally:
            server.stop()
        if measurement is None:
            raise RuntimeError(
                f"{variant.name} did not produce a governance measurement")
        row = governance_row(variant.name, measurement, no_access_latency)
        retention.record_variant(
            variant, "governance", GOVERNANCE_FIELDS, row)
        rows.append(row)
    return rows


def tier_variants() -> list[Variant]:
    return [
        Variant("tier", "Full Tiering", {"PLASMOD_TIER_PROFILE": "full"}, 2000),
        Variant("tier", "No-hot-cache", {"PLASMOD_TIER_PROFILE": "no_hot"}, 2000),
        Variant("tier", "Warm-only", {"PLASMOD_TIER_PROFILE": "warm_only"}, 2000),
        Variant("tier", "No-cold", {"PLASMOD_TIER_PROFILE": "no_cold"}, 2000),
        Variant("tier", "No-promotion", {"PLASMOD_TIER_PROFILE": "no_promotion"}, 2000),
        Variant("tier", "Hot-size-64", {"PLASMOD_TIER_PROFILE": "full"}, 64),
        Variant("tier", "Hot-size-512", {"PLASMOD_TIER_PROFILE": "full"}, 512),
        Variant("tier", "Hot-size-2000", {"PLASMOD_TIER_PROFILE": "full"}, 2000),
    ]


def measure_tier(server: PlasmodProcess, variant: Variant, data: RunData,
                 output_variant: str | None = None) -> dict[str, Any]:
    samples = data.latest_query_samples or data.query_samples
    sampled_id = samples[0][2] if samples else ""
    archive_id = sampled_id if sampled_id in data.memory_ids else next(iter(data.memory_ids), "")
    archive_latency = 0.0
    if archive_id and variant.env.get("PLASMOD_TIER_PROFILE", "full") not in ("warm_only", "no_cold"):
        archived = http_json(server.base, "POST", "/v1/admin/tier/archive", {"memory_id": archive_id})
        archive_latency = require_number(archived.get("archive_latency_ms", 0), "archive latency")
    tier_query_latencies = []
    hot = warm = cold = stale = total = 0
    promotions = [archive_latency]
    for text, vector, expected, requester, workspace, session in samples:
        response, latency = query(
            server.base, text, vector, requester=requester, include_cold=True,
            workspace=workspace, session=session)
        tier_query_latencies.append(latency)
        retrieval = response.get("retrieval") or {}
        hot += int(retrieval.get("hot_candidate_count", 0))
        warm += int(retrieval.get("warm_candidate_count", 0))
        cold += int(retrieval.get("cold_candidate_count", 0))
        promotions.append(float((response.get("diagnostics") or {}).get("promotion_latency_ms", 0)))
        canonical, _ = query(
            server.base, text, vector, [expected], "objects_only",
            requester=requester, workspace=workspace, session=session,
        )
        stale += int(expected not in canonical.get("objects", []))
        total += 1
    candidates = hot + warm + cold
    return {
        "System": "Plasmod", "Variant": output_variant or variant.name,
        "Hot Cache Size": variant.hot_size,
        "Query p50 (ms)": round(percentile(tier_query_latencies, 0.50), 6),
        "Query p95 (ms)": round(percentile(tier_query_latencies, 0.95), 6),
        "Query p99 (ms)": round(percentile(tier_query_latencies, 0.99), 6),
        "Hot Hit Rate (%)": round(pct(hot, candidates), 6),
        "Warm Hit Rate (%)": round(pct(warm, candidates), 6),
        "Cold Hit Rate (%)": round(pct(cold, candidates), 6),
        "Promotion Latency p95 (ms)": round(percentile(promotions, 0.95), 6),
        "Memory (MB)": round(server.rss_mb(), 6),
        "Stale Rate (%)": round(pct(stale, total), 6),
    }


def run_tier(run_dir: Path, port: int, event_limit: int, query_limit: int,
             baseline: SharedFullBaseline,
             retention: RetentionManager) -> list[dict[str, Any]]:
    rows = [baseline.module_rows["tier"]]
    for variant in tier_variants()[1:]:
        checkpoint = retention.load_variant_row(variant, "tier", TIER_FIELDS)
        if checkpoint is not None:
            log(f"reusing checkpointed tier variant: {variant.name}")
            rows.append(checkpoint)
            continue
        log(f"starting tier variant: {variant.name}")
        retention.prepare_variant(variant)
        server = PlasmodProcess(variant, run_dir, port)
        row: dict[str, Any] | None = None
        try:
            server.start(fresh=True)
            data = ingest_workload(
                server, variant, event_limit, query_limit, retention)
            row = measure_tier(server, variant, data)
        finally:
            server.stop()
        if row is None:
            raise RuntimeError(f"{variant.name} did not produce a tier result")
        retention.record_variant(variant, "tier", TIER_FIELDS, row)
        rows.append(row)
    return rows


def all_variants() -> list[Variant]:
    return [
        *recovery_variants(),
        *materialization_variants(),
        *evidence_variants(),
        *governance_variants(),
        *tier_variants(),
    ]


def shared_full_variant() -> Variant:
    return Variant("shared", "Full Plasmod", {
        "PLASMOD_WAL_MODE": "file",
        "PLASMOD_RECOVERY_REPLAY": "true",
        "PLASMOD_RECOVERY_PROJECTION": "full",
        "PLASMOD_MATERIALIZATION_PROFILE": "full",
        "PLASMOD_EVIDENCE_PROFILE": "full",
        "PLASMOD_GOVERNANCE_PROFILE": "full",
        "PLASMOD_TIER_PROFILE": "full",
    }, 2000)


def is_group_full_variant(variant: Variant) -> bool:
    labels = COMPARISON_LABELS.get(variant.group, {})
    return labels.get(variant.name, ("", ""))[0] == "Full"


def physical_variants() -> list[Variant]:
    return [shared_full_variant(), *(
        variant for variant in all_variants() if not is_group_full_variant(variant)
    )]


def common_metrics_path(run_dir: Path, variant: Variant) -> Path:
    metrics_variant = shared_full_variant() if is_group_full_variant(variant) else variant
    return run_dir / "variants" / metrics_variant.slug / "common_metrics.json"


def shared_full_baseline_path(run_dir: Path) -> Path:
    return run_dir / "variants" / shared_full_variant().slug / "shared_full_baseline.json"


def run_shared_full_baseline(run_dir: Path, port: int, event_limit: int,
                             query_limit: int,
                             retention: RetentionManager) -> SharedFullBaseline:
    variant = shared_full_variant()
    log("starting shared Full Plasmod baseline")
    retention.prepare_variant(variant)
    server = PlasmodProcess(variant, run_dir, port)
    baseline: SharedFullBaseline | None = None
    try:
        server.start(fresh=True)
        data = ingest_workload(
            server, variant, event_limit, query_limit, retention)
        reference_ids = data.state_ids | data.artifact_ids
        baseline = SharedFullBaseline(
            module_rows={},
            materialization_type_counts=materialization_type_counts(data),
            state_ids=set(data.state_ids),
            artifact_ids=set(data.artifact_ids),
            contexts={
                object_id: data.contexts[object_id]
                for object_id in reference_ids if object_id in data.contexts
            },
            evidence_totals=evidence_totals(data),
            governance_measurement={},
        )
        baseline.module_rows["materialization"] = measure_materialization(
            server, variant, data, baseline, "Full Plasmod")
        baseline.module_rows["evidence"] = measure_evidence(
            variant, data, baseline.evidence_totals, "Full Plasmod")
        baseline.module_rows["wal"] = measure_recovery(
            server, variant, data, "Full Plasmod")
        baseline.module_rows["tier"] = measure_tier(
            server, variant, data, "Full Tiering")
        baseline.governance_measurement = measure_governance(server, variant)
        baseline.write(shared_full_baseline_path(run_dir))
    finally:
        server.stop()
    if baseline is None:
        raise RuntimeError("shared Full Plasmod did not produce a baseline")
    SharedFullBaseline.read(shared_full_baseline_path(run_dir))
    retention.cleanup_variant(variant, (
        "capabilities.json",
        "measurements.json",
        "common_metrics.json",
        "server.log",
        "shared_full_baseline.json",
    ))
    return baseline


def load_or_run_shared_full_baseline(run_dir: Path, port: int, event_limit: int,
                                     query_limit: int, resume: bool,
                                     retention: RetentionManager) -> SharedFullBaseline:
    path = shared_full_baseline_path(run_dir)
    common_path = run_dir / "variants" / shared_full_variant().slug / "common_metrics.json"
    if resume and path.exists() and common_path.exists():
        log("reusing completed shared Full Plasmod baseline")
        baseline = SharedFullBaseline.read(path)
        retention.cleanup_variant(shared_full_variant(), (
            "capabilities.json",
            "measurements.json",
            "common_metrics.json",
            "server.log",
            "shared_full_baseline.json",
        ))
        return baseline
    return run_shared_full_baseline(
        run_dir, port, event_limit, query_limit, retention)


def variant_configuration(variant: Variant) -> dict[str, Any]:
    defaults = {
        "WAL Mode": "file",
        "Recovery Replay": "true",
        "Recovery Projection": "full",
        "Materialization Profile": "full",
        "Evidence Profile": "full",
        "Governance Profile": "full",
        "Tier Profile": "full",
    }
    env_to_column = {
        "PLASMOD_WAL_MODE": "WAL Mode",
        "PLASMOD_RECOVERY_REPLAY": "Recovery Replay",
        "PLASMOD_RECOVERY_PROJECTION": "Recovery Projection",
        "PLASMOD_MATERIALIZATION_PROFILE": "Materialization Profile",
        "PLASMOD_EVIDENCE_PROFILE": "Evidence Profile",
        "PLASMOD_GOVERNANCE_PROFILE": "Governance Profile",
        "PLASMOD_TIER_PROFILE": "Tier Profile",
    }
    for env_name, column in env_to_column.items():
        if env_name in variant.env:
            defaults[column] = variant.env[env_name]
    defaults["Hot Cache Size"] = variant.hot_size
    return defaults


def build_master_table(run_dir: Path,
                       tables: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    module_rows = {
        (group, str(row["Variant"])): row
        for group, rows in tables.items()
        for row in rows
    }
    rows: list[dict[str, Any]] = []
    for variant in all_variants():
        module_row = module_rows.get((variant.group, variant.name))
        if module_row is None:
            raise RuntimeError(f"missing module result for {variant.group}/{variant.name}")
        common_path = common_metrics_path(run_dir, variant)
        if not common_path.exists():
            raise RuntimeError(f"missing common metrics for {variant.group}/{variant.name}: {common_path}")
        common_payload = json.loads(common_path.read_text(encoding="utf-8"))
        common = common_payload.get("metrics") or {}
        missing_common = [field for field in COMMON_FIELDS if common.get(field) in (None, "")]
        if missing_common:
            raise RuntimeError(
                f"common metrics incomplete for {variant.group}/{variant.name}: {missing_common}")
        comparison_label, ablated_capability = COMPARISON_LABELS[variant.group][variant.name]
        row: dict[str, Any] = {
            "System": "Plasmod",
            "Module": variant.group,
            "Original Variant": variant.name,
            "Comparison Label": comparison_label,
            "Ablated Capability": ablated_capability,
            "Parameter Set": COMMON_PARAMETER_SET,
            "Write Consistency": "strict",
            "Query Consistency": "eventual",
            "Storage Backend": "Badger (disk)",
            "Cold Store": "MinIO S3",
            **variant_configuration(variant),
            **common,
        }
        for group, fields in MODULE_FIELDS.items():
            for field in fields:
                output_field = f"{group.upper()} | {field}"
                row[output_field] = module_row[field] if group == variant.group else NOT_APPLICABLE
        rows.append(row)
    if len(rows) != 34:
        raise RuntimeError(f"master ablation table must have 34 variants, got {len(rows)}")
    return rows


def write_master_style_manifest(run_dir: Path) -> None:
    manifest = {
        "purpose": "Header and cell colors identify metric scope; colors are not the sole data encoding.",
        "groups": {
            "identity_and_parameters": {"color": "#334155", "columns": MASTER_IDENTITY_FIELDS},
            "common": {"color": "#DBEAFE", "columns": COMMON_FIELDS},
            "wal": {"color": "#FFEDD5", "columns": [f"WAL | {field}" for field in MODULE_FIELDS["wal"]]},
            "materialization": {"color": "#DCFCE7", "columns": [
                f"MATERIALIZATION | {field}" for field in MODULE_FIELDS["materialization"]]},
            "evidence": {"color": "#F3E8FF", "columns": [
                f"EVIDENCE | {field}" for field in MODULE_FIELDS["evidence"]]},
            "governance": {"color": "#FEE2E2", "columns": [
                f"GOVERNANCE | {field}" for field in MODULE_FIELDS["governance"]]},
            "tier": {"color": "#CCFBF1", "columns": [f"TIER | {field}" for field in MODULE_FIELDS["tier"]]},
        },
        "not_applicable_value": NOT_APPLICABLE,
    }
    (run_dir / "ablation_master_style.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def build_full_baseline(tables: dict[str, list[dict[str, Any]]]) -> tuple[list[str], list[dict[str, Any]]]:
    row: dict[str, Any] = {
        "System": "Plasmod", "Variant": "Full database (disk WAL + canonical graph + retrieval + MinIO)",
    }
    for group, rows in tables.items():
        full = rows[0]
        for key, value in full.items():
            if key in ("System", "Variant"):
                continue
            row[f"{group}: {key}"] = value
    fields = list(row.keys())
    return fields, [row]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("smoke", "run"))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--event-limit", type=int, default=None,
                        help="events per general variant; 0 means all input")
    parser.add_argument("--query-limit", type=int, default=None)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="reuse completed variant checkpoints and CSV groups")
    parser.add_argument(
        "--retention", choices=("full", "metrics-only"), default="full",
        help="retain all database data or checkpoint metrics and remove per-variant data")
    parser.add_argument(
        "--disk-floor-gb", type=float, default=10,
        help="abort metrics-only ingestion before free disk drops below this value")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    global EMBEDDING_CACHE
    args = parse_args()
    event_limit = args.event_limit if args.event_limit is not None else (8 if args.mode == "smoke" else 1000)
    query_limit = args.query_limit if args.query_limit is not None else (3 if args.mode == "smoke" else 100)
    if event_limit < 0:
        raise ValueError("--event-limit must be 0 (all input) or a positive integer")
    if query_limit <= 0:
        raise ValueError("--query-limit must be a positive integer")
    run_id = args.run_id or f"agent_native_ablation_{args.mode}_{utc_id()}"
    run_dir = ROOT / "results" / "agent_native_ablation" / run_id
    run_dir.mkdir(parents=True, exist_ok=args.resume)
    retention = RetentionManager(
        run_dir, args.retention, disk_floor_gb=args.disk_floor_gb)
    retention.ensure_capacity("run start")
    mark_run_started(run_dir)
    (run_dir / "run.pid").write_text(str(os.getpid()), encoding="utf-8")
    metadata = {
        "run_id": run_id, "mode": args.mode, "event_limit": event_limit,
        "query_limit": query_limit, "top_k": 20, "embedding_dimension": 384,
        "retention_mode": args.retention,
        "disk_floor_gb": args.disk_floor_gb,
        "parameter_set": COMMON_PARAMETER_SET,
        "write_consistency": "strict", "query_consistency": "eventual",
        "storage_backend": "Badger (disk)", "cold_store": "MinIO S3",
        "logical_variant_rows": len(all_variants()),
        "physical_variant_runs": len(physical_variants()),
        "shared_full_runs": 1,
        "host": {
            "platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor(), "logical_cpu_count": os.cpu_count(),
        },
        "core_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=CORE, text=True).strip(),
        "experiment_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "data_sources": [str(path.relative_to(ROOT)) for path in event_files()],
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    minio = MinioManager(run_dir)
    EMBEDDING_CACHE = EmbeddingCache(EMBEDDING_CACHE_PATH)
    try:
        if not args.skip_build:
            log("building current Plasmod core")
            subprocess.run(["make", "build"], cwd=CORE, check=True)
        minio.start()
        baseline = load_or_run_shared_full_baseline(
            run_dir, args.port, event_limit, query_limit, args.resume, retention)
        tables: dict[str, list[dict[str, Any]]] = {}
        group_specs = [
            ("wal", "wal_event_log_ablation.csv", WAL_FIELDS,
             lambda: run_recovery(
                 run_dir, args.port, event_limit, query_limit, baseline, retention)),
            ("materialization", "materialization_ablation.csv", MATERIALIZATION_FIELDS,
             lambda: run_materialization(
                 run_dir, args.port, event_limit, query_limit, baseline, retention)),
            ("evidence", "evidence_provenance_ablation.csv", EVIDENCE_FIELDS,
             lambda: run_evidence(
                 run_dir, args.port, event_limit, query_limit, baseline, retention)),
            ("governance", "governance_ablation.csv", GOVERNANCE_FIELDS,
             lambda: run_governance(
                 run_dir, args.port, event_limit, query_limit, baseline, retention)),
            ("tier", "tiered_storage_ablation.csv", TIER_FIELDS,
             lambda: run_tier(
                 run_dir, args.port, event_limit, query_limit, baseline, retention)),
        ]
        for group, filename, fields, execute in group_specs:
            path = run_dir / filename
            if args.resume and path.exists():
                log(f"reusing completed group: {group}")
                tables[group] = read_csv(path)
                continue
            tables[group] = execute()
            write_csv(path, fields, tables[group])
            log(f"wrote {path.name}")
        full_fields, full_rows = build_full_baseline(tables)
        write_csv(run_dir / "full_database_baseline.csv", full_fields, full_rows)
        master_rows = build_master_table(run_dir, tables)
        write_csv(run_dir / "ablation_master_table.csv", MASTER_FIELDS, master_rows)
        write_master_style_manifest(run_dir)
        validate_service_logs(run_dir)
        summary = {
            "status": "complete", "row_counts": {name: len(rows) for name, rows in tables.items()},
            "master_row_count": len(master_rows),
            "physical_variant_runs": len(physical_variants()),
            "shared_full_runs": 1,
            "common_parameter_set": COMMON_PARAMETER_SET,
            "all_metrics_present": True, "minio_s3_exercised": True,
            **retention.summary(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        atomic_write_json(run_dir / "COMPLETE", {
            "status": "complete",
            "completed_at": summary["completed_at"],
        })
        try:
            (run_dir / "RUNNING").unlink()
        except FileNotFoundError:
            pass
        log(f"complete: {run_dir}")
        return 0
    except BaseException as exc:
        (run_dir / "FAILED").write_text(
            f"{type(exc).__name__}: {exc}", encoding="utf-8")
        try:
            (run_dir / "RUNNING").unlink()
        except FileNotFoundError:
            pass
        log(f"FAILED: {exc}")
        raise
    finally:
        if EMBEDDING_CACHE is not None:
            EMBEDDING_CACHE.close()
            EMBEDDING_CACHE = None
        minio.stop()
        try:
            (run_dir / "run.pid").unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    sys.exit(main())
