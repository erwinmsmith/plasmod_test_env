#!/usr/bin/env python3
"""
Layer 2 dynamic event benchmark runner.

This script consumes recorded agent runtime event JSONL files and produces the
metrics needed for:

- Table 4: Event Ingestion and Visibility
- Table 5: Freshness under Write Load
- Table 6: Consistency Mode Trade-off
- Table 7: Replay and Recovery
- Table 8: State Query Correctness under Dynamic Updates

The benchmark code lives in plasmod_test_env only. It treats Plasmod as a
black-box service through the public HTTP API.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import random
import re
import sqlite3
import statistics
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable


BASE = Path(__file__).resolve().parents[1]
DEFAULT_LAYER2 = BASE / "data" / "layer2_dynamic_events"
DEFAULT_SYNTHETIC = DEFAULT_LAYER2 / "traces_collected"
DEFAULT_REPLAY = DEFAULT_LAYER2 / "events.jsonl"
OUT = BASE / "results" / "layer2_dynamic_events"
DEFAULT_EMBEDDER_MODEL = BASE / "models" / "all-MiniLM-L6-v2.onnx"
DEFAULT_EMBEDDER_VOCAB = BASE.parent / "Plasmod" / "models" / "minilm-l6-v2-vocab.txt"
EMBEDDING_DIM = 384
MAX_EMBED_TOKENS = 128
BERT_CLS = 101
BERT_SEP = 102
BERT_PAD = 0
BERT_UNK = 100

PLASMOD_MODES = {
    "strict": "strict_visible",
    "bounded": "bounded_staleness",
    "eventual": "eventual_visibility",
}

EVENT_TYPE_ALIASES = {
    "state": "state_update",
    "state_update": "state_update",
}

TABLE4_PLASMOD_TYPES = ["observation", "tool_result", "memory", "state_update", "artifact", "relation"]
TABLE4_BASELINE_TYPES = ["observation", "tool_result", "memory"]
MILVUS_SYSTEM_ALIASES = {"milvus", "vector_metadata", "baseline"}


def wants_milvus_baseline(systems: list[str]) -> bool:
    return any(system in MILVUS_SYSTEM_ALIASES for system in systems)


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def wall_ms() -> int:
    return int(time.time() * 1000)


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    idx = int(round((pct / 100.0) * (len(ordered) - 1)))
    idx = max(0, min(idx, len(ordered) - 1))
    return float(ordered[idx])


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.fmean(values))


def safe_div(num: float, den: float) -> float | None:
    if den == 0:
        return None
    return num / den


def zero_if_none(value: float | None) -> float:
    return 0.0 if value is None else value


def percent(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return (num / den) * 100.0


def sanitize_collection_name(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z_]+", "_", value)
    if not value or value[0].isdigit():
        value = "c_" + value
    return value[:255]


def collection_name_from_path(path: Path | None, prefix: str = "layer2_milvus") -> str:
    raw = str(path or (OUT / "default"))
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    stem = sanitize_collection_name(Path(raw).stem or "run")
    return sanitize_collection_name(f"{prefix}_{stem}_{digest}")


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x20000 <= code <= 0x2A6DF
        or 0xF900 <= code <= 0xFAFF
        or 0x2F800 <= code <= 0x2FA1F
    )


def _is_ascii_punct(ch: str) -> bool:
    code = ord(ch)
    return 33 <= code <= 47 or 58 <= code <= 64 or 91 <= code <= 96 or 123 <= code <= 126


class MiniLMEmbedder:
    def __init__(self, model_path: Path = DEFAULT_EMBEDDER_MODEL, vocab_path: Path = DEFAULT_EMBEDDER_VOCAB):
        self.model_path = model_path
        self.vocab_path = vocab_path
        self._session: Any = None
        self._input_names: list[str] = []
        self._output_name = ""
        self._vocab: dict[str, int] = {}

    def _load(self) -> None:
        if self._session is not None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(f"MiniLM ONNX model not found: {self.model_path}")
        if not self.vocab_path.exists():
            raise FileNotFoundError(f"MiniLM vocab not found: {self.vocab_path}")
        import numpy as np
        import onnxruntime as ort

        with self.vocab_path.open("r", encoding="utf-8") as f:
            self._vocab = {line.strip(): i for i, line in enumerate(f) if line.strip()}
        self._session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
        self._input_names = [inp.name for inp in self._session.get_inputs()]
        output_names = [out.name for out in self._session.get_outputs()]
        self._output_name = "last_hidden_state" if "last_hidden_state" in output_names else output_names[0]
        self._np = np

    def _normalize_text(self, text: str) -> str:
        import unicodedata

        text = unicodedata.normalize("NFD", text.lower())
        chars: list[str] = []
        for ch in text:
            if unicodedata.category(ch) == "Mn":
                continue
            if _is_cjk(ch) or _is_ascii_punct(ch):
                chars.extend([" ", ch, " "])
            elif ch.isspace():
                chars.append(" ")
            else:
                chars.append(ch)
        return "".join(chars)

    def _word_piece_split(self, word: str) -> list[int]:
        chars = list(word)
        tokens: list[int] = []
        start = 0
        while start < len(chars):
            end = len(chars)
            found = False
            while end > start:
                sub = word[start:end]
                if start > 0:
                    sub = "##" + sub
                if sub in self._vocab:
                    tokens.append(self._vocab[sub])
                    start = end
                    found = True
                    break
                end -= 1
            if not found:
                return [BERT_UNK]
        return tokens

    def _tokenize(self, texts: list[str]) -> tuple[Any, Any]:
        np = self._np
        input_ids = np.full((len(texts), MAX_EMBED_TOKENS), BERT_PAD, dtype=np.int64)
        attention_mask = np.zeros((len(texts), MAX_EMBED_TOKENS), dtype=np.int64)
        for i, text in enumerate(texts):
            words = self._normalize_text(text).split()
            pos = 1
            input_ids[i, 0] = BERT_CLS
            attention_mask[i, 0] = 1
            for word in words:
                if pos >= MAX_EMBED_TOKENS - 1:
                    break
                for token_id in self._word_piece_split(word):
                    if pos >= MAX_EMBED_TOKENS - 1:
                        break
                    input_ids[i, pos] = token_id
                    attention_mask[i, pos] = 1
                    pos += 1
            input_ids[i, pos] = BERT_SEP
            attention_mask[i, pos] = 1
        return input_ids, attention_mask

    def embed_one(self, text: str) -> list[float]:
        vectors = self.embed_many([text])
        return vectors[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        self._load()
        np = self._np
        input_ids, attention_mask = self._tokenize([t or "" for t in texts])
        feed = {self._input_names[0]: input_ids, self._input_names[1]: attention_mask}
        if len(self._input_names) > 2:
            feed[self._input_names[2]] = np.zeros_like(input_ids)
        last_hidden = self._session.run([self._output_name], feed)[0]
        mask = attention_mask[..., np.newaxis].astype(np.float32)
        pooled = np.sum(last_hidden * mask, axis=1) / np.maximum(np.sum(mask, axis=1), 1e-9)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        pooled = pooled / np.maximum(norms, 1e-12)
        return pooled.astype("float32").tolist()


def get_path(doc: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = doc
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def set_path(doc: dict[str, Any], path: str, value: Any) -> None:
    cur = doc
    parts = path.split(".")
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def event_type(doc: dict[str, Any]) -> str:
    value = get_path(doc, "event.event_type") or doc.get("event_type") or get_path(doc, "event.eventType")
    value = str(value or "").strip()
    return EVENT_TYPE_ALIASES.get(value, value)


def object_type(doc: dict[str, Any]) -> str:
    value = get_path(doc, "object.object_type") or doc.get("object_type")
    return str(value or "").strip()


def object_id(doc: dict[str, Any]) -> str:
    value = get_path(doc, "object.object_id") or doc.get("object_id")
    return str(value or "").strip()


def event_id(doc: dict[str, Any]) -> str:
    value = get_path(doc, "identity.event_id") or doc.get("event_id")
    return str(value or "").strip()


def session_id(doc: dict[str, Any]) -> str:
    value = get_path(doc, "actor.session_id") or doc.get("session_id")
    return str(value or "").strip()


def agent_id(doc: dict[str, Any]) -> str:
    value = get_path(doc, "actor.agent_id") or doc.get("agent_id")
    return str(value or "").strip()


def workspace_id(doc: dict[str, Any]) -> str:
    value = get_path(doc, "identity.workspace_id") or doc.get("workspace_id")
    return str(value or "").strip()


def tenant_id(doc: dict[str, Any]) -> str:
    value = get_path(doc, "identity.tenant_id") or doc.get("tenant_id")
    return str(value or "").strip()


def event_version(doc: dict[str, Any]) -> int:
    value = get_path(doc, "object.version") or doc.get("version") or 0
    try:
        return int(value)
    except Exception:
        return 0


def payload_text(doc: dict[str, Any]) -> str:
    candidates = [
        get_path(doc, "retrieval.index_text"),
        get_path(doc, "payload.text"),
        get_path(doc, "payload.content.body"),
        get_path(doc, "payload.content.question"),
        get_path(doc, "payload.state.value"),
        get_path(doc, "payload.state.key"),
        get_path(doc, "payload.artifact.title"),
        get_path(doc, "payload.artifact.name"),
    ]
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()[:4096]
    payload = doc.get("payload")
    if payload is not None:
        try:
            return json.dumps(payload, ensure_ascii=False, sort_keys=True)[:4096]
        except Exception:
            return str(payload)[:4096]
    return event_id(doc) or object_id(doc)


def materialization_enabled(doc: dict[str, Any]) -> bool:
    value = get_path(doc, "materialization.enabled")
    if value is None:
        return event_type(doc) in {"memory", "state_update", "artifact", "relation"}
    return bool(value)


def json_contains_any(doc: Any, needles: set[str]) -> bool:
    if not needles:
        return False
    try:
        text = json.dumps(doc, ensure_ascii=False, sort_keys=True)
    except Exception:
        text = str(doc)
    return any(n and n in text for n in needles)


def response_ids(resp: Any) -> set[str]:
    ids: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, sub in value.items():
                if key in {"object_id", "event_id", "memory_id", "state_id", "artifact_id", "edge_id", "id"}:
                    if isinstance(sub, str):
                        ids.add(sub)
                walk(sub)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, str):
            if value.startswith(("evt_", "obj_", "mem_", "state_", "artifact_", "edge_", "rel_")):
                ids.add(value)

    walk(resp)
    return ids


def list_jsonl_inputs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(path)
    return sorted(p for p in path.rglob("*.jsonl") if p.is_file())


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if isinstance(value, dict):
                yield value


def load_events(
    input_path: Path,
    limit: int = 0,
    event_types: set[str] | None = None,
    shuffle: bool = False,
    seed: int = 7,
    max_files: int = 0,
) -> list[dict[str, Any]]:
    files = list_jsonl_inputs(input_path)
    if max_files > 0:
        files = files[:max_files]
    out: list[dict[str, Any]] = []
    for path in files:
        for ev in iter_jsonl(path):
            et = event_type(ev)
            if event_types and et not in event_types:
                continue
            out.append(ev)
            if limit > 0 and len(out) >= limit:
                break
        if limit > 0 and len(out) >= limit:
            break
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(out)
    return out


def prefix_string(value: Any, run_id: str) -> Any:
    if not isinstance(value, str) or not value:
        return value
    if value.startswith(run_id + "_"):
        return value
    return f"{run_id}_{value}"


def namespace_event(ev: dict[str, Any], run_id: str, mode: str | None = None) -> dict[str, Any]:
    doc = copy.deepcopy(ev)
    for path in [
        "identity.event_id",
        "object.object_id",
        "causality.parent_event_id",
        "causality.call_event_id",
        "causality.source_object_id",
        "causality.target_object_id",
    ]:
        value = get_path(doc, path)
        if isinstance(value, str) and value:
            set_path(doc, path, prefix_string(value, run_id))

    for path in ["causality.causal_refs", "causality.provenance_refs", "causality.source_object_ids", "causality.target_object_ids", "materialization.planned_object_ids"]:
        value = get_path(doc, path)
        if isinstance(value, list):
            set_path(doc, path, [prefix_string(v, run_id) if isinstance(v, str) else v for v in value])

    for path in ["actor.session_id", "retrieval.retrieval_namespace", "identity.import_batch_id"]:
        value = get_path(doc, path)
        if isinstance(value, str) and value:
            set_path(doc, path, prefix_string(value, run_id))

    if mode:
        set_path(doc, "access.consistency", mode)
    set_path(doc, "runtime.t_write_start_ms", wall_ms())
    return doc


@dataclass
class QuerySpec:
    query_id: str
    query_type: str
    query_text: str
    session_id: str
    agent_id: str
    workspace_id: str
    tenant_id: str
    object_types: list[str]
    expected_ids: set[str]
    expected_version: int
    source_event_type: str


@dataclass
class IngestResult:
    system: str
    event_type: str
    event_id: str
    object_id: str
    expected_ids: set[str]
    write_start_ms: float
    write_ack_ms: float
    write_latency_ms: float
    ok: bool
    error: str = ""
    ack: dict[str, Any] = field(default_factory=dict)
    first_visible_ms: float | None = None
    materialized_ms: float | None = None
    visibility_censored: bool = False

    @property
    def write_to_visible_ms(self) -> float | None:
        if self.first_visible_ms is None:
            return None
        return self.first_visible_ms - self.write_ack_ms

    @property
    def materialization_lag_ms(self) -> float | None:
        if self.materialized_ms is None:
            return None
        return self.materialized_ms - self.write_ack_ms


@dataclass
class QueryResult:
    system: str
    query_type: str
    latency_ms: float
    ok: bool
    visible: bool
    stale: bool
    error: str = ""


class HTTPJSONClient:
    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        url = self.base_url + path
        data = None
        headers = {}
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{method} {path} failed: HTTP {exc.code}: {detail}") from exc
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {"raw": raw.decode("utf-8", errors="replace")}


class SystemAdapter:
    name = "system"

    def health(self) -> None:
        return None

    def close(self) -> None:
        return None

    def set_visibility_mode(self, mode: str) -> None:
        return None

    def reset(self) -> None:
        return None

    def ingest(self, ev: dict[str, Any]) -> IngestResult:
        raise NotImplementedError

    def query(self, q: QuerySpec) -> tuple[QueryResult, Any]:
        raise NotImplementedError

    def replay(self, from_lsn: int = 0, limit: int = 0, apply: bool = False) -> dict[str, Any]:
        raise NotImplementedError


class PlasmodAdapter(SystemAdapter):
    name = "Plasmod"

    def __init__(self, base_url: str, timeout: float = 30.0):
        self.http = HTTPJSONClient(base_url, timeout)

    def health(self) -> None:
        self.http.request("GET", "/healthz")

    def set_visibility_mode(self, mode: str) -> None:
        api_mode = PLASMOD_MODES.get(mode, mode)
        self.http.request("POST", "/v1/admin/consistency-mode", {"mode": api_mode})

    def reset(self) -> None:
        self.http.request("POST", "/v1/admin/data/wipe", {"confirm": "delete_all_data"})

    def ingest(self, ev: dict[str, Any]) -> IngestResult:
        t0 = now_ms()
        eid = event_id(ev)
        oid = object_id(ev)
        expected = {eid, oid}
        try:
            ack = self.http.request("POST", "/v1/ingest/events", ev)
            t1 = now_ms()
            if isinstance(ack, dict):
                for key in ("memory_id", "event_id", "object_id", "state_id", "artifact_id", "edge_id"):
                    value = ack.get(key)
                    if isinstance(value, str) and value:
                        expected.add(value)
                for key in ("edge_ids", "relation_ids", "object_ids"):
                    values = ack.get(key)
                    if isinstance(values, list):
                        expected.update(str(value) for value in values if value)
            return IngestResult(
                system=self.name,
                event_type=event_type(ev),
                event_id=eid,
                object_id=oid,
                expected_ids={x for x in expected if x},
                write_start_ms=t0,
                write_ack_ms=t1,
                write_latency_ms=t1 - t0,
                ok=True,
                ack=ack if isinstance(ack, dict) else {"raw_ack": ack},
            )
        except Exception as exc:
            t1 = now_ms()
            return IngestResult(
                system=self.name,
                event_type=event_type(ev),
                event_id=eid,
                object_id=oid,
                expected_ids={x for x in expected if x},
                write_start_ms=t0,
                write_ack_ms=t1,
                write_latency_ms=t1 - t0,
                ok=False,
                error=str(exc),
            )

    def query(self, q: QuerySpec) -> tuple[QueryResult, Any]:
        top_k = 100 if q.query_type in {"relation_query", "provenance_query"} else 10
        body = {
            "query_text": q.query_text or "latest",
            "session_id": q.session_id,
            "agent_id": q.agent_id,
            "workspace_id": q.workspace_id,
            "tenant_id": q.tenant_id,
            "top_k": top_k,
            "object_types": q.object_types,
            "response_mode": "objects_only",
        }
        body = {k: v for k, v in body.items() if v not in ("", [], None)}
        t0 = now_ms()
        try:
            resp = self.http.request("POST", "/v1/query", body)
            t1 = now_ms()
            ids = response_ids(resp)
            visible = bool(ids & q.expected_ids) or json_contains_any(resp, q.expected_ids)
            return QueryResult(self.name, q.query_type, t1 - t0, True, visible, not visible), resp
        except Exception as exc:
            t1 = now_ms()
            return QueryResult(self.name, q.query_type, t1 - t0, False, False, True, str(exc)), {}

    def replay(self, from_lsn: int = 0, limit: int = 0, apply: bool = False) -> dict[str, Any]:
        body = {"from_lsn": from_lsn, "limit": limit, "apply": apply}
        if apply:
            body["confirm"] = "apply_replay"
            body["dry_run"] = False
        return self.http.request("POST", "/v1/admin/replay", body)


def milvus_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def milvus_string_list(values: list[str]) -> str:
    return "[" + ", ".join(f'"{milvus_escape(v)}"' for v in values) + "]"


def milvus_hit_entity(hit: Any) -> dict[str, Any]:
    if isinstance(hit, dict):
        entity = hit.get("entity")
        if isinstance(entity, dict):
            return entity
        return hit
    entity = getattr(hit, "entity", None)
    if isinstance(entity, dict):
        return entity
    if hasattr(hit, "get"):
        try:
            entity = hit.get("entity")
            if isinstance(entity, dict):
                return entity
        except Exception:
            pass
    return {}


class MilvusAdapter(SystemAdapter):
    name = "Milvus"

    def __init__(
        self,
        uri: str = "http://127.0.0.1:19530",
        collection_name: str = "layer2_milvus",
        timeout: float = 30.0,
        embedder: MiniLMEmbedder | None = None,
    ):
        from pymilvus import MilvusClient

        self.uri = uri
        self.collection_name = collection_name
        self.timeout = timeout
        self.embedder = embedder or MiniLMEmbedder()
        self.client = MilvusClient(uri=uri)
        self.mu = threading.Lock()
        self._ensure_collection(drop=False)

    def _ensure_collection(self, drop: bool) -> None:
        from pymilvus import DataType, MilvusClient

        if drop:
            try:
                self.client.drop_collection(self.collection_name)
            except Exception:
                pass
        if self.client.has_collection(self.collection_name):
            try:
                self.client.load_collection(self.collection_name)
            except Exception:
                pass
            return

        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("pk", DataType.VARCHAR, is_primary=True, max_length=256)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
        schema.add_field("event_id", DataType.VARCHAR, max_length=512)
        schema.add_field("object_id", DataType.VARCHAR, max_length=512)
        schema.add_field("session_id", DataType.VARCHAR, max_length=512)
        schema.add_field("agent_id", DataType.VARCHAR, max_length=256)
        schema.add_field("workspace_id", DataType.VARCHAR, max_length=256)
        schema.add_field("tenant_id", DataType.VARCHAR, max_length=256)
        schema.add_field("event_type", DataType.VARCHAR, max_length=128)
        schema.add_field("object_type", DataType.VARCHAR, max_length=128)
        schema.add_field("version", DataType.INT64)
        schema.add_field("text", DataType.VARCHAR, max_length=8192)
        schema.add_field("payload_json", DataType.VARCHAR, max_length=65535)

        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            "vector",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )
        self.client.create_collection(
            self.collection_name,
            schema=schema,
            index_params=index_params,
            consistency_level="Strong",
            timeout=self.timeout,
        )
        self.client.load_collection(self.collection_name)

    def health(self) -> None:
        self.client.list_collections()

    def reset(self) -> None:
        with self.mu:
            self._ensure_collection(drop=True)

    def close(self) -> None:
        return None

    def _row_for_event(self, ev: dict[str, Any], vector: list[float]) -> dict[str, Any]:
        eid = event_id(ev)
        oid = object_id(ev)
        payload_json = json.dumps(ev, ensure_ascii=False, sort_keys=True)
        return {
            "pk": (eid or oid or hashlib.sha1(payload_json.encode("utf-8")).hexdigest())[:256],
            "vector": vector,
            "event_id": eid[:512],
            "object_id": oid[:512],
            "session_id": session_id(ev)[:512],
            "agent_id": agent_id(ev)[:256],
            "workspace_id": workspace_id(ev)[:256],
            "tenant_id": tenant_id(ev)[:256],
            "event_type": event_type(ev)[:128],
            "object_type": object_type(ev)[:128],
            "version": int(event_version(ev)),
            "text": payload_text(ev)[:8192],
            "payload_json": payload_json[:65535],
        }

    def ingest(self, ev: dict[str, Any]) -> IngestResult:
        t0 = now_ms()
        eid = event_id(ev)
        oid = object_id(ev)
        try:
            vector = self.embedder.embed_one(payload_text(ev))
            row = self._row_for_event(ev, vector)
            with self.mu:
                self.client.insert(self.collection_name, [row])
            t1 = now_ms()
            return IngestResult(
                system=self.name,
                event_type=event_type(ev),
                event_id=eid,
                object_id=oid,
                expected_ids={x for x in [eid, oid] if x},
                write_start_ms=t0,
                write_ack_ms=t1,
                write_latency_ms=t1 - t0,
                ok=True,
            )
        except Exception as exc:
            t1 = now_ms()
            return IngestResult(
                system=self.name,
                event_type=event_type(ev),
                event_id=eid,
                object_id=oid,
                expected_ids={x for x in [eid, oid] if x},
                write_start_ms=t0,
                write_ack_ms=t1,
                write_latency_ms=t1 - t0,
                ok=False,
                error=str(exc),
            )

    def query(self, q: QuerySpec) -> tuple[QueryResult, Any]:
        t0 = now_ms()
        try:
            clauses: list[str] = []
            if q.session_id:
                clauses.append(f'session_id == "{milvus_escape(q.session_id)}"')
            if q.object_types:
                values = milvus_string_list(q.object_types)
                clauses.append(f"(object_type in {values} or event_type in {values})")
            expr = " and ".join(clauses) if clauses else ""
            vector = self.embedder.embed_one(q.query_text or "latest")
            with self.mu:
                rows = self.client.search(
                    self.collection_name,
                    [vector],
                    limit=10,
                    filter=expr,
                    output_fields=["event_id", "object_id", "session_id", "event_type", "object_type", "version", "text"],
                    search_params={"ef": 64},
                    timeout=self.timeout,
                )
            ids: set[str] = set()
            objects: list[dict[str, Any]] = []
            for hit in rows[0] if rows else []:
                entity = milvus_hit_entity(hit)
                if entity:
                    objects.append(entity)
                for key in ("event_id", "object_id", "id", "pk"):
                    value = entity.get(key)
                    if isinstance(value, str) and value:
                        ids.add(value)
            visible = bool(ids & q.expected_ids)
            t1 = now_ms()
            resp = {"objects": list(ids), "rows": objects}
            return QueryResult(self.name, q.query_type, t1 - t0, True, visible, not visible), resp
        except Exception as exc:
            t1 = now_ms()
            return QueryResult(self.name, q.query_type, t1 - t0, False, False, True, str(exc)), {}

    def replay_events(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        t0 = now_ms()
        applied = 0
        failed = 0
        for ev in events:
            res = self.ingest(ev)
            if res.ok:
                applied += 1
            else:
                failed += 1
        if applied:
            try:
                self.client.flush(self.collection_name)
            except Exception:
                pass
        elapsed = max((now_ms() - t0) / 1000.0, 1e-9)
        return {
            "status": "ok",
            "attempted": len(events),
            "applied": applied,
            "failed": failed,
            "elapsed_s": elapsed,
            "throughput_eps": applied / elapsed,
        }


class VectorMetadataAdapter(SystemAdapter):
    name = "SQLiteMetadata"

    def __init__(self, db_path: Path | None = None):
        self.db_path = str(db_path) if db_path else ":memory:"
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.mu = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    object_id TEXT,
                    session_id TEXT,
                    agent_id TEXT,
                    workspace_id TEXT,
                    tenant_id TEXT,
                    event_type TEXT,
                    object_type TEXT,
                    version INTEGER,
                    text TEXT,
                    payload_json TEXT,
                    created_ns INTEGER
                )
                """
            )
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_events_lookup ON events(session_id, event_type, object_type, version)")

    def reset(self) -> None:
        with self.mu, self.conn:
            self.conn.execute("DELETE FROM events")

    def close(self) -> None:
        self.conn.close()

    def ingest(self, ev: dict[str, Any]) -> IngestResult:
        t0 = now_ms()
        eid = event_id(ev)
        oid = object_id(ev)
        try:
            with self.mu, self.conn:
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO events
                    (event_id, object_id, session_id, agent_id, workspace_id, tenant_id,
                     event_type, object_type, version, text, payload_json, created_ns)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        eid,
                        oid,
                        session_id(ev),
                        agent_id(ev),
                        workspace_id(ev),
                        tenant_id(ev),
                        event_type(ev),
                        object_type(ev),
                        event_version(ev),
                        payload_text(ev),
                        json.dumps(ev, ensure_ascii=False, sort_keys=True),
                        time.perf_counter_ns(),
                    ),
                )
            t1 = now_ms()
            return IngestResult(
                system=self.name,
                event_type=event_type(ev),
                event_id=eid,
                object_id=oid,
                expected_ids={x for x in [eid, oid] if x},
                write_start_ms=t0,
                write_ack_ms=t1,
                write_latency_ms=t1 - t0,
                ok=True,
                first_visible_ms=t1,
                materialized_ms=t1,
            )
        except Exception as exc:
            t1 = now_ms()
            return IngestResult(
                system=self.name,
                event_type=event_type(ev),
                event_id=eid,
                object_id=oid,
                expected_ids={x for x in [eid, oid] if x},
                write_start_ms=t0,
                write_ack_ms=t1,
                write_latency_ms=t1 - t0,
                ok=False,
                error=str(exc),
            )

    def query(self, q: QuerySpec) -> tuple[QueryResult, Any]:
        t0 = now_ms()
        try:
            clauses = []
            params: list[Any] = []
            if q.session_id:
                clauses.append("session_id = ?")
                params.append(q.session_id)
            if q.object_types:
                marks = ",".join("?" for _ in q.object_types)
                clauses.append(f"(object_type IN ({marks}) OR event_type IN ({marks}))")
                params.extend(q.object_types)
                params.extend(q.object_types)
            where = " AND ".join(clauses) if clauses else "1=1"
            sql = f"SELECT event_id, object_id, version, text FROM events WHERE {where} ORDER BY version DESC, created_ns DESC LIMIT 10"
            with self.mu:
                rows = self.conn.execute(sql, params).fetchall()
            ids = {str(v) for row in rows for v in row[:2] if v}
            visible = bool(ids & q.expected_ids)
            t1 = now_ms()
            resp = {"objects": list(ids), "rows": rows}
            return QueryResult(self.name, q.query_type, t1 - t0, True, visible, not visible), resp
        except Exception as exc:
            t1 = now_ms()
            return QueryResult(self.name, q.query_type, t1 - t0, False, False, True, str(exc)), {}

    def replay_events(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        t0 = now_ms()
        applied = 0
        failed = 0
        for ev in events:
            res = self.ingest(ev)
            if res.ok:
                applied += 1
            else:
                failed += 1
        elapsed = max((now_ms() - t0) / 1000.0, 1e-9)
        return {
            "status": "ok",
            "attempted": len(events),
            "applied": applied,
            "failed": failed,
            "elapsed_s": elapsed,
            "throughput_eps": applied / elapsed,
        }


def make_adapter(
    system: str,
    base_url: str,
    sqlite_path: Path | None = None,
    timeout: float = 30.0,
    milvus_uri: str = "http://127.0.0.1:19530",
    embedder_model: Path = DEFAULT_EMBEDDER_MODEL,
    embedder_vocab: Path = DEFAULT_EMBEDDER_VOCAB,
) -> SystemAdapter:
    if system == "plasmod":
        return PlasmodAdapter(base_url, timeout=timeout)
    if system in {"milvus", "vector_metadata", "baseline"}:
        embedder = MiniLMEmbedder(embedder_model, embedder_vocab)
        return MilvusAdapter(
            uri=milvus_uri,
            collection_name=collection_name_from_path(sqlite_path),
            timeout=timeout,
            embedder=embedder,
        )
    if system == "sqlite_metadata":
        return VectorMetadataAdapter(sqlite_path)
    raise ValueError(f"unknown system: {system}")


def query_for_event(ev: dict[str, Any], run_id: str, query_id: str) -> QuerySpec:
    et = event_type(ev)
    ot = object_type(ev)
    qtype = {
        "state_update": "latest_state",
        "memory": "latest_memory",
        "artifact": "artifact_lookup",
        "relation": "relation_query",
    }.get(et, "scope_aware_retrieval")
    object_types = {
        "state_update": ["agent_state", "state", "memory"],
        "memory": ["memory"],
        "artifact": ["artifact"],
        "relation": ["edge", "relation"],
        "tool_result": ["event", "memory"],
        "observation": ["event", "memory"],
    }.get(et, [ot] if ot else [])
    expected = {event_id(ev), object_id(ev)}
    return QuerySpec(
        query_id=query_id,
        query_type=qtype,
        query_text=payload_text(ev),
        session_id=session_id(ev),
        agent_id=agent_id(ev),
        workspace_id=workspace_id(ev),
        tenant_id=tenant_id(ev),
        object_types=[x for x in object_types if x],
        expected_ids={x for x in expected if x},
        expected_version=event_version(ev),
        source_event_type=et,
    )


def ingest_with_rate(
    adapter: SystemAdapter,
    events: list[dict[str, Any]],
    rate_eps: float,
    workers: int,
    on_complete: Callable[[dict[str, Any], IngestResult], None] | None = None,
) -> list[IngestResult]:
    if not events:
        return []
    interval = 0.0 if rate_eps <= 0 else 1.0 / rate_eps
    start = time.perf_counter()
    futures: list[tuple[dict[str, Any], Future[IngestResult]]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for idx, ev in enumerate(events):
            target = start + idx * interval
            delay = target - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            futures.append((ev, pool.submit(adapter.ingest, ev)))
        results: list[IngestResult] = []
        for ev, fut in futures:
            res = fut.result()
            results.append(res)
            if on_complete is not None:
                on_complete(ev, res)
    return results


def wait_until_visible(
    adapter: SystemAdapter,
    ev: dict[str, Any],
    ingest_result: IngestResult,
    run_id: str,
    timeout_ms: float,
    poll_ms: float,
) -> QueryResult:
    query = query_for_event(ev, run_id, "visibility_" + (ingest_result.event_id or "unknown"))
    query.expected_ids |= ingest_result.expected_ids
    deadline = now_ms() + timeout_ms
    last: QueryResult | None = None
    while now_ms() <= deadline:
        qr, _ = adapter.query(query)
        last = qr
        if qr.ok and qr.visible:
            t = now_ms()
            if ingest_result.first_visible_ms is None:
                ingest_result.first_visible_ms = t
            if materialization_enabled(ev) and ingest_result.materialized_ms is None:
                ingest_result.materialized_ms = t
            return qr
        time.sleep(max(poll_ms, 1.0) / 1000.0)
    if ingest_result.ok and ingest_result.first_visible_ms is None:
        ingest_result.first_visible_ms = ingest_result.write_ack_ms + timeout_ms
        ingest_result.visibility_censored = True
        if materialization_enabled(ev) and ingest_result.materialized_ms is None:
            ingest_result.materialized_ms = ingest_result.write_ack_ms + timeout_ms
    if last is None:
        last = QueryResult(adapter.name, query.query_type, 0.0, False, False, True, "visibility timeout")
    return last


def wait_for_visible_batch(
    adapter: SystemAdapter,
    events: list[dict[str, Any]],
    ingests: list[IngestResult],
    run_id: str,
    timeout_ms: float,
    poll_ms: float,
    workers: int,
) -> list[QueryResult]:
    pairs = [(ev, res) for ev, res in zip(events, ingests) if res.ok]
    if not pairs:
        return []
    max_workers = max(1, min(workers, len(pairs)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(wait_until_visible, adapter, ev, res, run_id, timeout_ms, poll_ms)
            for ev, res in pairs
        ]
        return [future.result() for future in futures]


def fill_missing_visibility(
    adapter: SystemAdapter,
    events: list[dict[str, Any]],
    ingests: list[IngestResult],
    run_id: str,
    timeout_ms: float,
    poll_ms: float,
    workers: int,
) -> None:
    pairs = [(ev, res) for ev, res in zip(events, ingests) if res.ok and res.first_visible_ms is None]
    if not pairs:
        return
    max_workers = max(1, min(workers, len(pairs)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(wait_until_visible, adapter, ev, res, run_id, timeout_ms, poll_ms)
            for ev, res in pairs
        ]
        for future in futures:
            future.result()


def summarize_ingests(system: str, key: str, ingests: list[IngestResult], queries: list[QueryResult] | None = None) -> dict[str, Any]:
    ok_ingests = [r for r in ingests if r.ok]
    failed_ingests = [r for r in ingests if not r.ok]
    write_lat = [r.write_latency_ms for r in ok_ingests]
    w2v = [r.write_to_visible_ms for r in ok_ingests if r.write_to_visible_ms is not None]
    mat = [r.materialization_lag_ms for r in ok_ingests if r.materialization_lag_ms is not None]
    visibility_timeouts = sum(1 for r in ok_ingests if r.visibility_censored)
    if mat:
        materialization_lag_p95 = percentile(mat, 95)
        materialization_lag_basis = "materialized_timestamp_or_first_visible"
    elif w2v:
        materialization_lag_p95 = percentile(w2v, 95)
        materialization_lag_basis = "write_to_visible_proxy"
    else:
        materialization_lag_p95 = 0.0
        materialization_lag_basis = "no_successful_writes"
    stale_count = 0
    query_count = 0
    if queries is not None:
        query_count = len(queries)
        stale_count = sum(1 for q in queries if q.stale)
    write_window_ms = max((r.write_ack_ms for r in ok_ingests), default=0) - min((r.write_start_ms for r in ok_ingests), default=0)
    return {
        "system": system,
        "key": key,
        "events": len(ingests),
        "successful_writes": len(ok_ingests),
        "write_errors": len(failed_ingests),
        "first_error": failed_ingests[0].error if failed_ingests else "none",
        "visibility_timeouts": visibility_timeouts,
        "visibility_measurement_mode": "timeout_censored" if visibility_timeouts else "observed",
        "write_qps": zero_if_none(safe_div(len(ok_ingests), write_window_ms / 1000.0)),
        "write_p50_ms": zero_if_none(percentile(write_lat, 50)),
        "write_p95_ms": zero_if_none(percentile(write_lat, 95)),
        "write_to_visible_p50_ms": zero_if_none(percentile(w2v, 50)),
        "write_to_visible_p95_ms": zero_if_none(percentile(w2v, 95)),
        "materialization_lag_p95_ms": zero_if_none(materialization_lag_p95),
        "materialization_lag_basis": materialization_lag_basis,
        "query_count": query_count,
        "stale_result_rate": zero_if_none(safe_div(stale_count, query_count)),
    }


def summarize_queries(system: str, key: str, queries: list[QueryResult]) -> dict[str, Any]:
    ok = [q for q in queries if q.ok]
    lat = [q.latency_ms for q in ok]
    return {
        "system": system,
        "key": key,
        "query_count": len(queries),
        "query_qps": 0.0,
        "query_p50_ms": zero_if_none(percentile(lat, 50)),
        "query_p95_ms": zero_if_none(percentile(lat, 95)),
        "stale_result_rate": zero_if_none(safe_div(sum(1 for q in queries if q.stale), len(queries))),
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, sort_keys=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("not_measured" if row.get(key) is None else row.get(key)) for key in fields})


def run_table4(args: argparse.Namespace, run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    systems = args.systems
    for system in systems:
        adapter = make_adapter(
            system,
            args.plasmod_url,
            run_dir / f"{system}.collection",
            args.http_timeout,
            args.milvus_uri,
            args.embedder_model,
            args.embedder_vocab,
        )
        if system == "plasmod":
            adapter.health()
        types = TABLE4_PLASMOD_TYPES if system == "plasmod" else TABLE4_BASELINE_TYPES
        for et in types:
            raw = load_events(args.synthetic_input, limit=args.events_per_type, event_types={et}, shuffle=args.shuffle, seed=args.seed)
            run_id = f"{args.run_id}_t4_{system}_{et}"
            events = [namespace_event(ev, run_id) for ev in raw]
            if args.reset_between_runs:
                adapter.reset()
            ingests = ingest_with_rate(adapter, events, args.write_rate, args.workers)
            queries = wait_for_visible_batch(adapter, events, ingests, run_id, args.visible_timeout_ms, args.visible_poll_ms, args.workers)
            row = summarize_ingests(adapter.name, et.replace("state_update", "state"), ingests, queries)
            row.update({"table": "table4", "event_type": et.replace("state_update", "state")})
            rows.append(row)
        adapter.close()
    write_csv(run_dir / "table4_event_ingestion_visibility.csv", rows)
    return rows


def run_freshness_trial(
    adapter: SystemAdapter,
    events: list[dict[str, Any]],
    run_id: str,
    write_rate: float,
    query_qps: float,
    workers: int,
    query_limit: int,
) -> tuple[list[IngestResult], list[QueryResult], float]:
    completed: list[tuple[dict[str, Any], IngestResult]] = []
    completed_mu = threading.Lock()
    stop = threading.Event()
    query_results: list[QueryResult] = []
    q_interval = 0.0 if query_qps <= 0 else 1.0 / query_qps

    def on_complete(ev: dict[str, Any], res: IngestResult) -> None:
        if res.ok:
            with completed_mu:
                completed.append((ev, res))

    def query_loop() -> None:
        idx = 0
        while True:
            if query_limit > 0 and idx >= query_limit:
                break
            with completed_mu:
                item = completed[-1] if completed else None
            if item is None and stop.is_set():
                break
            if item is not None:
                ev, res = item
                q = query_for_event(ev, run_id, f"q_{idx}")
                q.expected_ids |= res.expected_ids
                qr, _ = adapter.query(q)
                if qr.ok and qr.visible and res.first_visible_ms is None:
                    t = now_ms()
                    res.first_visible_ms = t
                    if materialization_enabled(ev) and res.materialized_ms is None:
                        res.materialized_ms = t
                query_results.append(qr)
                idx += 1
                if stop.is_set() and len(completed) == 0:
                    break
            time.sleep(q_interval if q_interval > 0 else 0.001)

    t0 = now_ms()
    qt = threading.Thread(target=query_loop, daemon=True)
    qt.start()
    ingests = ingest_with_rate(adapter, events, write_rate, workers, on_complete=on_complete)
    stop.set()
    qt.join(timeout=30)
    elapsed_s = max((now_ms() - t0) / 1000.0, 1e-9)
    return ingests, query_results, elapsed_s


def run_table5(args: argparse.Namespace, run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for system in args.systems:
        adapter = make_adapter(
            system,
            args.plasmod_url,
            run_dir / f"{system}.collection",
            args.http_timeout,
            args.milvus_uri,
            args.embedder_model,
            args.embedder_vocab,
        )
        if system == "plasmod":
            adapter.health()
        for rate in args.write_rates:
            raw = load_events(args.synthetic_input, limit=args.events_per_rate, shuffle=args.shuffle, seed=args.seed)
            run_id = f"{args.run_id}_t5_{system}_{rate}"
            events = [namespace_event(ev, run_id) for ev in raw]
            if args.reset_between_runs:
                adapter.reset()
            ingests, queries, elapsed_s = run_freshness_trial(adapter, events, run_id, rate, args.query_qps, args.workers, args.query_limit)
            fill_missing_visibility(adapter, events, ingests, run_id, args.visible_timeout_ms, args.visible_poll_ms, args.workers)
            ingest_row = summarize_ingests(adapter.name, str(rate), ingests, queries)
            query_row = summarize_queries(adapter.name, str(rate), queries)
            row = {
                "table": "table5",
                "system": adapter.name,
                "write_rate_events_s": rate,
                "query_qps": zero_if_none(safe_div(len(queries), elapsed_s)),
                "query_p50_ms": query_row["query_p50_ms"],
                "query_p95_ms": query_row["query_p95_ms"],
                "write_to_visible_p95_ms": ingest_row["write_to_visible_p95_ms"],
                "materialization_lag_p95_ms": ingest_row["materialization_lag_p95_ms"],
                "materialization_lag_basis": ingest_row["materialization_lag_basis"],
                "stale_result_rate": ingest_row["stale_result_rate"],
                "events": len(ingests),
                "successful_writes": ingest_row["successful_writes"],
                "write_errors": ingest_row["write_errors"],
                "first_error": ingest_row["first_error"],
                "visibility_timeouts": ingest_row["visibility_timeouts"],
                "visibility_measurement_mode": ingest_row["visibility_measurement_mode"],
                "queries": len(queries),
            }
            rows.append(row)
        adapter.close()
    write_csv(run_dir / "table5_freshness_under_write_load.csv", rows)
    return rows


def run_table6(args: argparse.Namespace, run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    raw = load_events(args.synthetic_input, limit=args.events_per_rate, shuffle=args.shuffle, seed=args.seed)
    if "plasmod" in args.systems:
        for mode, guarantee in [
            ("strict", "read-after-write"),
            ("bounded", "bounded lag"),
            ("eventual", "eventual visibility"),
        ]:
            adapter = make_adapter("plasmod", args.plasmod_url, timeout=args.http_timeout)
            adapter.health()
            adapter.set_visibility_mode(mode)
            run_id = f"{args.run_id}_t6_plasmod_{mode}"
            events = [namespace_event(ev, run_id, PLASMOD_MODES[mode]) for ev in raw]
            if args.reset_between_runs:
                adapter.reset()
            ingests, queries, elapsed_s = run_freshness_trial(adapter, events, run_id, args.fixed_write_rate, args.query_qps, args.workers, args.query_limit)
            fill_missing_visibility(adapter, events, ingests, run_id, args.visible_timeout_ms, args.visible_poll_ms, args.workers)
            qrow = summarize_queries(adapter.name, mode, queries)
            irow = summarize_ingests(adapter.name, mode, ingests, queries)
            rows.append({
                "table": "table6",
                "system": "Plasmod",
                "visibility_mode": {"strict": "Strict", "bounded": "Bounded Staleness", "eventual": "Eventual"}[mode],
                "write_qps": irow["write_qps"],
                "query_qps": zero_if_none(safe_div(len(queries), elapsed_s)),
                "query_p95_ms": qrow["query_p95_ms"],
                "write_to_visible_p95_ms": irow["write_to_visible_p95_ms"],
                "materialization_lag_p95_ms": irow["materialization_lag_p95_ms"],
                "materialization_lag_basis": irow["materialization_lag_basis"],
                "stale_result_rate": irow["stale_result_rate"],
                "freshness_guarantee": guarantee,
                "events": len(ingests),
                "successful_writes": irow["successful_writes"],
                "write_errors": irow["write_errors"],
                "first_error": irow["first_error"],
                "visibility_timeouts": irow["visibility_timeouts"],
                "visibility_measurement_mode": irow["visibility_measurement_mode"],
                "queries": len(queries),
            })
            adapter.close()

    if wants_milvus_baseline(args.systems):
        adapter = make_adapter(
            "milvus",
            args.plasmod_url,
            run_dir / "milvus_t6.collection",
            args.http_timeout,
            args.milvus_uri,
            args.embedder_model,
            args.embedder_vocab,
        )
        run_id = f"{args.run_id}_t6_milvus_best_effort"
        events = [namespace_event(ev, run_id) for ev in raw]
        if args.reset_between_runs:
            adapter.reset()
        ingests, queries, elapsed_s = run_freshness_trial(adapter, events, run_id, args.fixed_write_rate, args.query_qps, args.workers, args.query_limit)
        fill_missing_visibility(adapter, events, ingests, run_id, args.visible_timeout_ms, args.visible_poll_ms, args.workers)
        qrow = summarize_queries(adapter.name, "best_effort", queries)
        irow = summarize_ingests(adapter.name, "best_effort", ingests, queries)
        rows.append({
            "table": "table6",
            "system": adapter.name,
            "visibility_mode": "Best-effort",
            "write_qps": irow["write_qps"],
            "query_qps": zero_if_none(safe_div(len(queries), elapsed_s)),
            "query_p95_ms": qrow["query_p95_ms"],
            "write_to_visible_p95_ms": irow["write_to_visible_p95_ms"],
            "materialization_lag_p95_ms": irow["materialization_lag_p95_ms"],
            "materialization_lag_basis": irow["materialization_lag_basis"],
            "stale_result_rate": irow["stale_result_rate"],
            "freshness_guarantee": "best effort",
            "events": len(ingests),
            "successful_writes": irow["successful_writes"],
            "write_errors": irow["write_errors"],
            "first_error": irow["first_error"],
            "visibility_timeouts": irow["visibility_timeouts"],
            "visibility_measurement_mode": irow["visibility_measurement_mode"],
            "queries": len(queries),
        })
        adapter.close()
    write_csv(run_dir / "table6_consistency_mode_tradeoff.csv", rows)
    return rows


def run_table7(args: argparse.Namespace, run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    raw = load_events(args.replay_input, limit=args.replay_events, shuffle=False)
    replay_relation_events = sum(1 for ev in raw if event_type(ev) == "relation")

    if "plasmod" in args.systems:
        adapter = make_adapter("plasmod", args.plasmod_url, timeout=args.http_timeout)
        adapter.health()
        run_id = f"{args.run_id}_t7_plasmod_seed"
        events = [namespace_event(ev, run_id) for ev in raw]
        if args.reset_between_runs:
            adapter.reset()
        ingests = ingest_with_rate(adapter, events, args.write_rate, args.workers)
        event_log_size = sum(1 for r in ingests if r.ok)
        failed_ingests = [r for r in ingests if not r.ok]
        for failure_type in ["materialized view reset", "index rebuild", "service restart"]:
            t0 = now_ms()
            try:
                if failure_type == "materialized view reset":
                    out = adapter.replay(0, args.replay_events, apply=False)
                    elapsed_s = max((now_ms() - t0) / 1000.0, 1e-9)
                    recovered = out.get("sampled_entries") or out.get("applied") or 0
                    note = "preview only; current API has no materialized-view-only reset endpoint"
                    status = "requires_manual_reset"
                else:
                    out = adapter.replay(0, args.replay_events, apply=False)
                    elapsed_s = max((now_ms() - t0) / 1000.0, 1e-9)
                    recovered = out.get("sampled_entries") or 0
                    note = "preview only; automate the failure externally, then rerun replay apply"
                    status = "requires_manual_failure"
                coverage_pct = percent(float(recovered), float(event_log_size))
                relation_coverage_pct = coverage_pct if replay_relation_events > 0 else 100.0
                rows.append({
                    "table": "table7",
                    "system": "Plasmod",
                    "failure_type": failure_type,
                    "event_log_size": event_log_size,
                    "write_errors": len(failed_ingests),
                    "first_error": failed_ingests[0].error if failed_ingests else "none",
                    "replay_throughput_events_s": zero_if_none(safe_div(float(recovered), elapsed_s)),
                    "recovery_time_s": elapsed_s,
                    "recovered_objects_pct": coverage_pct,
                    "recovered_relations_pct": relation_coverage_pct,
                    "query_available_during_recovery": True,
                    "recovery_measurement_mode": "replay_dry_run_proxy",
                    "status": status,
                    "note": note,
                })
            except Exception as exc:
                rows.append({
                    "table": "table7",
                    "system": "Plasmod",
                    "failure_type": failure_type,
                    "event_log_size": event_log_size,
                    "write_errors": len(failed_ingests),
                    "first_error": failed_ingests[0].error if failed_ingests else str(exc),
                    "replay_throughput_events_s": 0.0,
                    "recovery_time_s": 0.0,
                    "recovered_objects_pct": 0.0,
                    "recovered_relations_pct": 0.0,
                    "query_available_during_recovery": False,
                    "recovery_measurement_mode": "replay_dry_run_failed",
                    "status": "failed",
                    "note": str(exc),
                })
        adapter.close()

    if wants_milvus_baseline(args.systems):
        events = [namespace_event(ev, f"{args.run_id}_t7_milvus") for ev in raw]
        baseline = make_adapter(
            "milvus",
            args.plasmod_url,
            run_dir / "milvus_t7.collection",
            args.http_timeout,
            args.milvus_uri,
            args.embedder_model,
            args.embedder_vocab,
        )
        baseline.reset()
        replay_out = baseline.replay_events(events)
        rows.append({
            "table": "table7",
            "system": baseline.name,
            "failure_type": "service restart",
            "event_log_size": len(events),
            "write_errors": replay_out["failed"],
            "first_error": "none" if replay_out["failed"] == 0 else "baseline_replay_failed",
            "replay_throughput_events_s": replay_out["throughput_eps"],
            "recovery_time_s": replay_out["elapsed_s"],
            "recovered_objects_pct": percent(float(replay_out["applied"]), float(len(events))),
            "recovered_relations_pct": percent(float(replay_out["applied"]), float(len(events))) if replay_relation_events > 0 else 100.0,
            "query_available_during_recovery": False,
            "recovery_measurement_mode": "milvus_reingest_rebuild",
            "status": "ok",
            "note": "Milvus baseline rebuilds vector records and scalar metadata from input JSONL, not from a database WAL",
        })
        baseline.close()
    write_csv(run_dir / "table7_replay_recovery.csv", rows)
    return rows


def select_table8_events(events: list[dict[str, Any]], update_type: str) -> list[dict[str, Any]]:
    if update_type == "tool_result_to_state":
        by_id = {event_id(ev): ev for ev in events}
        out = []
        for ev in events:
            if event_type(ev) != "state_update":
                continue
            parents = [get_path(ev, "causality.parent_event_id")] + list(get_path(ev, "causality.causal_refs", []) or [])
            if any(event_type(by_id.get(str(p), {})) == "tool_result" for p in parents if p):
                out.append(ev)
        return out
    if update_type == "artifact_update":
        return [ev for ev in events if event_type(ev) == "artifact"]
    if update_type == "relation_update":
        return [ev for ev in events if event_type(ev) == "relation"]
    return events


def run_table8(args: argparse.Namespace, run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    raw_all = load_events(args.replay_input, limit=args.replay_events, shuffle=False)
    cases: list[tuple[str, str, str]] = []
    if wants_milvus_baseline(args.systems):
        cases.extend([
            ("milvus", "tool_result -> state", "tool_result_to_state"),
            ("milvus", "artifact update", "artifact_update"),
        ])
    if "plasmod" in args.systems:
        cases.extend([
            ("plasmod", "tool_result -> state", "tool_result_to_state"),
            ("plasmod", "artifact update", "artifact_update"),
            ("plasmod", "relation update", "relation_update"),
        ])
    for system, label, selector in cases:
        selected = select_table8_events(raw_all, selector)[: args.table8_updates]
        run_id = f"{args.run_id}_t8_{system}_{selector}"
        events = [namespace_event(ev, run_id) for ev in selected]
        adapter = make_adapter(
            system,
            args.plasmod_url,
            run_dir / f"{system}_t8.collection",
            args.http_timeout,
            args.milvus_uri,
            args.embedder_model,
            args.embedder_vocab,
        )
        if system == "plasmod":
            adapter.health()
        if args.reset_between_runs:
            adapter.reset()
        ingests = ingest_with_rate(adapter, events, args.write_rate, args.workers)
        failed_ingests = [r for r in ingests if not r.ok]
        queries: list[QueryResult] = []
        for ev, res in zip(events, ingests):
            if not res.ok:
                continue
            q = query_for_event(ev, run_id, "state_correctness_" + (event_id(ev) or object_id(ev)))
            q.expected_ids |= res.expected_ids
            qr, _ = adapter.query(q)
            queries.append(qr)
        correct = sum(1 for q in queries if q.visible)
        stale = sum(1 for q in queries if q.stale)
        lat = [q.latency_ms for q in queries if q.ok]
        rows.append({
            "table": "table8",
            "system": adapter.name,
            "update_type": label,
            "num_updates": len(ingests),
            "successful_writes": sum(1 for r in ingests if r.ok),
            "write_errors": len(failed_ingests),
            "first_error": failed_ingests[0].error if failed_ingests else "none",
            "state_query_accuracy": zero_if_none(safe_div(correct, len(queries))),
            "latest_state_hit_rate": zero_if_none(safe_div(correct, len(queries))),
            "stale_state_error_rate": zero_if_none(safe_div(stale, len(queries))),
            "avg_query_latency_ms": zero_if_none(mean(lat)),
            "correctness_measurement_mode": "direct_query" if queries else "no_successful_updates",
            "queries": len(queries),
        })
        adapter.close()
    write_csv(run_dir / "table8_state_query_correctness.csv", rows)
    return rows


def run_analyze(args: argparse.Namespace) -> None:
    rows = []
    for name, path in [("synthetic", args.synthetic_input), ("replay", args.replay_input)]:
        counts: dict[str, int] = {}
        files = list_jsonl_inputs(path)
        total = 0
        for p in files[: args.max_files if args.max_files else None]:
            for ev in iter_jsonl(p):
                total += 1
                counts[event_type(ev)] = counts.get(event_type(ev), 0) + 1
                if args.limit and total >= args.limit:
                    break
            if args.limit and total >= args.limit:
                break
        rows.append({"source": name, "path": str(path), "files": len(files), "records_scanned": total, "event_type_counts": counts})
    print(json.dumps(rows, indent=2, ensure_ascii=False, sort_keys=True))


def run_tables(args: argparse.Namespace) -> Path:
    run_id = args.run_id or time.strftime("layer2_%Y%m%d_%H%M%S")
    args.run_id = run_id
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "run_id": run_id,
        "synthetic_input": str(args.synthetic_input),
        "replay_input": str(args.replay_input),
        "systems": args.systems,
        "milvus_uri": args.milvus_uri,
        "embedder_model": str(args.embedder_model),
        "embedder_vocab": str(args.embedder_vocab),
        "write_rates": args.write_rates,
        "query_qps": args.query_qps,
        "created_at_ms": wall_ms(),
        "notes": [
            "write-to-visible is measured by polling /v1/query until expected ids appear",
            "materialization lag uses first-visible time as an external black-box proxy when no materialized timestamp is returned",
            "table7 Plasmod reset/rebuild rows are marked requires_manual_* unless the service exposes an automatic failure trigger",
        ],
    }
    write_json(run_dir / "run_metadata.json", metadata)

    all_rows: dict[str, list[dict[str, Any]]] = {}
    selected = set(args.tables)
    if "all" in selected or "4" in selected:
        all_rows["table4"] = run_table4(args, run_dir)
    if "all" in selected or "5" in selected:
        all_rows["table5"] = run_table5(args, run_dir)
    if "all" in selected or "6" in selected:
        all_rows["table6"] = run_table6(args, run_dir)
    if "all" in selected or "7" in selected:
        all_rows["table7"] = run_table7(args, run_dir)
    if "all" in selected or "8" in selected:
        all_rows["table8"] = run_table8(args, run_dir)
    write_json(run_dir / "summary.json", all_rows)
    return run_dir


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Layer 2 Dynamic Event Stream and State Visibility benchmark runner.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--synthetic-input", type=Path, default=DEFAULT_SYNTHETIC, help="Synthetic Agent Event Stream JSONL file or directory.")
        p.add_argument("--replay-input", type=Path, default=DEFAULT_REPLAY, help="Replayable Agent Execution Trace JSONL file or directory.")
        p.add_argument("--plasmod-url", default="http://127.0.0.1:8080")
        p.add_argument("--output-dir", type=Path, default=OUT)
        p.add_argument("--run-id", default="")
        p.add_argument("--seed", type=int, default=7)
        p.add_argument("--shuffle", action="store_true")

    analyze = sub.add_parser("analyze", help="Summarize available Layer 2 input data.")
    add_common(analyze)
    analyze.add_argument("--limit", type=int, default=0)
    analyze.add_argument("--max-files", type=int, default=0)

    run = sub.add_parser("run", help="Run one or more Layer 2 experiment tables.")
    add_common(run)
    run.add_argument("--tables", nargs="+", default=["all"], choices=["all", "4", "5", "6", "7", "8"])
    run.add_argument(
        "--systems",
        nargs="+",
        default=["milvus", "plasmod"],
        choices=["milvus", "plasmod", "vector_metadata", "baseline", "sqlite_metadata"],
        help="Systems to benchmark. vector_metadata/baseline are accepted as legacy aliases for milvus.",
    )
    run.add_argument("--write-rates", type=float, nargs="+", default=[10, 100, 500, 1000])
    run.add_argument("--write-rate", type=float, default=100.0, help="Default write rate for Table 4, 7, 8.")
    run.add_argument("--fixed-write-rate", type=float, default=100.0, help="Fixed write rate for Table 6.")
    run.add_argument("--query-qps", type=float, default=25.0)
    run.add_argument("--query-limit", type=int, default=200)
    run.add_argument("--workers", type=int, default=32)
    run.add_argument("--events-per-type", type=int, default=100)
    run.add_argument("--events-per-rate", type=int, default=1000)
    run.add_argument("--replay-events", type=int, default=5000)
    run.add_argument("--table8-updates", type=int, default=1000)
    run.add_argument("--visible-timeout-ms", type=float, default=5000.0)
    run.add_argument("--visible-poll-ms", type=float, default=50.0)
    run.add_argument("--http-timeout", type=float, default=30.0)
    run.add_argument("--milvus-uri", default="http://127.0.0.1:19530")
    run.add_argument("--embedder-model", type=Path, default=DEFAULT_EMBEDDER_MODEL)
    run.add_argument("--embedder-vocab", type=Path, default=DEFAULT_EMBEDDER_VOCAB)
    run.add_argument("--reset-between-runs", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.cmd == "analyze":
        run_analyze(args)
        return 0
    if args.cmd == "run":
        run_dir = run_tables(args)
        print(run_dir)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
