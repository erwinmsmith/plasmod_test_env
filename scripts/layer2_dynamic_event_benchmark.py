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
import csv
import hashlib
import json
import os
import random
import re
import shutil
import sqlite3
import statistics
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable


BASE = Path(__file__).resolve().parents[1]
DEFAULT_LAYER2 = BASE / "data" / "layer2_dynamic_events"
DEFAULT_SYNTHETIC = DEFAULT_LAYER2 / "traces_collected"
DEFAULT_REPLAY = DEFAULT_LAYER2 / "events.jsonl"
OUT = BASE / "results" / "layer2_dynamic_events"
DEFAULT_EMBEDDING_CACHE = OUT / "embedding_cache.sqlite3"
DEFAULT_EMBEDDER_MODEL = BASE / "models" / "all-MiniLM-L6-v2.onnx"
DEFAULT_EMBEDDER_VOCAB = BASE.parent / "Plasmod" / "models" / "minilm-l6-v2-vocab.txt"
EMBEDDING_DIM = 384
MAX_EMBED_TOKENS = 128
DEFAULT_MAX_HOT_EVENT_BYTES = 1024 * 1024
DEFAULT_HOT_PAYLOAD_PREVIEW_BYTES = 4096
HOT_EVENT_MAX_BYTES = DEFAULT_MAX_HOT_EVENT_BYTES
HOT_PAYLOAD_PREVIEW_BYTES = DEFAULT_HOT_PAYLOAD_PREVIEW_BYTES
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
PAYLOAD_EXTERNALIZATION_STATS: dict[str, int] = {
    "events_externalized": 0,
    "original_payload_bytes": 0,
    "original_record_bytes": 0,
}
PAYLOAD_EXTERNALIZATION_LOCK = threading.Lock()


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


class HashEmbedder:
    """Deterministic vectors for Layer 2 visibility experiments.

    The dynamic-event tables query by agent object ids, not ANN semantic recall.
    This embedder keeps Milvus' vector schema realistic without charging MiniLM
    inference to the database write path.
    """

    def embed_one(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        import numpy as np

        vectors: list[list[float]] = []
        for text in texts:
            seed = (text or "").encode("utf-8")
            raw = bytearray()
            counter = 0
            while len(raw) < EMBEDDING_DIM * 4:
                raw.extend(hashlib.sha256(seed + counter.to_bytes(4, "little")).digest())
                counter += 1
            ints = np.frombuffer(bytes(raw[: EMBEDDING_DIM * 4]), dtype=np.uint32).astype(np.float32)
            vec = (ints / np.float32(4294967295.0)) * np.float32(2.0) - np.float32(1.0)
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec = vec / np.float32(norm)
            vectors.append(vec.astype("float32").tolist())
        return vectors

    def prewarm_texts(self, texts: Iterable[str]) -> dict[str, Any]:
        unique = len([text for text in dict.fromkeys((t or "") for t in texts) if text])
        return {
            "embedding_provider": "hash",
            "texts": unique,
            "new_embeddings": 0,
            "cached_embeddings": 0,
            "elapsed_s": 0.0,
        }


def embedding_model_signature(model_path: Path, vocab_path: Path) -> str:
    h = hashlib.sha256()
    for path in [model_path, vocab_path]:
        resolved = path.resolve()
        stat = resolved.stat()
        h.update(str(resolved).encode("utf-8"))
        h.update(str(stat.st_size).encode("ascii"))
        h.update(str(stat.st_mtime_ns).encode("ascii"))
    h.update(str(EMBEDDING_DIM).encode("ascii"))
    h.update(str(MAX_EMBED_TOKENS).encode("ascii"))
    return h.hexdigest()


class CachedEmbedder:
    def __init__(
        self,
        embedder: MiniLMEmbedder,
        cache_path: Path,
        model_path: Path,
        vocab_path: Path,
        batch_size: int = 64,
    ):
        self.embedder = embedder
        self.cache_path = cache_path
        self.model_sig = embedding_model_signature(model_path, vocab_path)
        self.batch_size = max(1, batch_size)
        self.mu = threading.Lock()
        self._ready = False

    def _ensure_db(self) -> None:
        if self._ready:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute(
                """
                create table if not exists embeddings (
                    model_sig text not null,
                    text_hash text not null,
                    text text not null,
                    dim integer not null,
                    vector blob not null,
                    created_at_ms integer not null,
                    primary key (model_sig, text_hash)
                )
                """
            )
            conn.execute("create index if not exists embeddings_text_hash_idx on embeddings(text_hash)")
        self._ready = True

    @staticmethod
    def _text_hash(text: str) -> str:
        return hashlib.sha256((text or "").encode("utf-8")).hexdigest()

    @staticmethod
    def _vector_to_blob(vector: list[float]) -> bytes:
        import numpy as np

        return np.asarray(vector, dtype=np.float32).tobytes()

    @staticmethod
    def _blob_to_vector(blob: bytes, dim: int) -> list[float]:
        import numpy as np

        return np.frombuffer(blob, dtype=np.float32, count=dim).astype(np.float32).tolist()

    def embed_one(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        texts = [t or "" for t in texts]
        if not texts:
            return []
        with self.mu:
            self._ensure_db()
            results: dict[str, list[float]] = {}
            unique_texts = list(dict.fromkeys(texts))
            hashes = {text: self._text_hash(text) for text in unique_texts}
            with sqlite3.connect(self.cache_path) as conn:
                for text in unique_texts:
                    row = conn.execute(
                        "select text, dim, vector from embeddings where model_sig = ? and text_hash = ?",
                        (self.model_sig, hashes[text]),
                    ).fetchone()
                    if row is None:
                        continue
                    cached_text, dim, blob = row
                    if cached_text == text and int(dim) == EMBEDDING_DIM:
                        results[text] = self._blob_to_vector(blob, EMBEDDING_DIM)

                missing = [text for text in unique_texts if text not in results]
                for start in range(0, len(missing), self.batch_size):
                    chunk = missing[start : start + self.batch_size]
                    vectors = self.embedder.embed_many(chunk)
                    rows = [
                        (
                            self.model_sig,
                            hashes[text],
                            text,
                            EMBEDDING_DIM,
                            self._vector_to_blob(vector),
                            wall_ms(),
                        )
                        for text, vector in zip(chunk, vectors)
                    ]
                    conn.executemany(
                        """
                        insert or replace into embeddings
                        (model_sig, text_hash, text, dim, vector, created_at_ms)
                        values (?, ?, ?, ?, ?, ?)
                        """,
                        rows,
                    )
                    conn.commit()
                    for text, vector in zip(chunk, vectors):
                        results[text] = vector
        return [results[text] for text in texts]

    def prewarm_texts(self, texts: Iterable[str]) -> dict[str, Any]:
        unique = [text for text in dict.fromkeys((t or "") for t in texts) if text]
        before = self.count_cached()
        started = now_ms()
        for start in range(0, len(unique), self.batch_size):
            self.embed_many(unique[start : start + self.batch_size])
        after = self.count_cached()
        return {
            "cache_path": str(self.cache_path),
            "model_sig": self.model_sig,
            "texts": len(unique),
            "new_embeddings": max(0, after - before),
            "cached_embeddings": after,
            "elapsed_s": (now_ms() - started) / 1000.0,
        }

    def count_cached(self) -> int:
        with self.mu:
            self._ensure_db()
            with sqlite3.connect(self.cache_path) as conn:
                row = conn.execute(
                    "select count(*) from embeddings where model_sig = ?",
                    (self.model_sig,),
                ).fetchone()
        return int(row[0] if row else 0)


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


def set_path_copy(doc: dict[str, Any], path: str, value: Any) -> dict[str, Any]:
    copied = dict(doc)
    cur_dst: dict[str, Any] = copied
    cur_src: Any = doc
    parts = path.split(".")
    for part in parts[:-1]:
        src_child = cur_src.get(part) if isinstance(cur_src, dict) else None
        dst_child = dict(src_child) if isinstance(src_child, dict) else {}
        cur_dst[part] = dst_child
        cur_dst = dst_child
        cur_src = src_child
    cur_dst[parts[-1]] = value
    return copied


def int_path(doc: dict[str, Any], path: str) -> int:
    value = get_path(doc, path)
    try:
        return int(value)
    except Exception:
        return 0


def utf8_len(value: str) -> int:
    return len(value.encode("utf-8"))


def truncate_text_bytes(value: str, max_bytes: int) -> str:
    if max_bytes <= 0:
        return ""
    raw = value.encode("utf-8")
    if len(raw) <= max_bytes:
        return value
    return raw[:max_bytes].decode("utf-8", errors="ignore")


def summarize_externalized_value(value: Any, preview_bytes: int) -> Any:
    if isinstance(value, str):
        return {
            "externalized": True,
            "type": "str",
            "original_bytes": utf8_len(value),
            "preview": truncate_text_bytes(value, preview_bytes),
        }
    if isinstance(value, dict):
        out: dict[str, Any] = {"externalized": True, "type": "dict", "fields": {}}
        fields: dict[str, Any] = {}
        for key, sub in value.items():
            if isinstance(sub, str):
                fields[key] = {
                    "type": "str",
                    "original_bytes": utf8_len(sub),
                    "preview": truncate_text_bytes(sub, preview_bytes),
                }
            elif isinstance(sub, (int, float, bool)) or sub is None:
                fields[key] = sub
            elif isinstance(sub, list):
                fields[key] = {
                    "type": "list",
                    "length": len(sub),
                    "preview": sub[:10],
                }
            elif isinstance(sub, dict):
                fields[key] = {
                    "type": "dict",
                    "keys": sorted(str(k) for k in sub.keys())[:50],
                }
            else:
                fields[key] = {"type": type(sub).__name__}
        out["fields"] = fields
        return out
    if isinstance(value, list):
        return {
            "externalized": True,
            "type": "list",
            "length": len(value),
            "preview": value[:10],
        }
    return {"externalized": True, "type": type(value).__name__}


def maybe_externalize_large_event(doc: dict[str, Any]) -> dict[str, Any]:
    """Keep agent DB hot-path writes bounded while preserving object semantics.

    Recorded traces can contain full artifacts such as 100MB+ patches. Those
    bytes belong to the cold artifact body, while the Layer 2 experiment measures
    event/object/state visibility. We keep identity, provenance, hashes, size
    metadata, and a small preview in the online write.
    """

    max_hot_bytes = HOT_EVENT_MAX_BYTES
    if max_hot_bytes <= 0:
        return doc
    payload_bytes = int_path(doc, "data.payload_size_bytes")
    record_bytes = int_path(doc, "data.record_size_bytes")
    if max(payload_bytes, record_bytes) <= max_hot_bytes:
        return doc

    eid = event_id(doc) or object_id(doc) or "unknown"
    original_hash = get_path(doc, "data.payload_hash") or ""
    external_ref = f"agent-artifact://{eid}/{original_hash or 'payload'}"
    metadata = {
        "enabled": True,
        "reason": "hot_path_payload_exceeds_limit",
        "max_hot_event_bytes": max_hot_bytes,
        "preview_bytes": HOT_PAYLOAD_PREVIEW_BYTES,
        "original_payload_size_bytes": payload_bytes,
        "original_record_size_bytes": record_bytes,
        "payload_hash": original_hash,
        "external_ref": external_ref,
    }

    out = set_path_copy(doc, "extensions.benchmark.payload_externalization", metadata)
    payload_content = get_path(out, "payload.content")
    if payload_content is not None:
        out = set_path_copy(out, "payload.content", summarize_externalized_value(payload_content, HOT_PAYLOAD_PREVIEW_BYTES))
    elif out.get("payload") is not None:
        out = set_path_copy(out, "payload", summarize_externalized_value(out.get("payload"), HOT_PAYLOAD_PREVIEW_BYTES))

    retrieval_text = get_path(out, "retrieval.index_text")
    if retrieval_text is not None:
        out = set_path_copy(out, "retrieval.index_text", summarize_externalized_value(retrieval_text, HOT_PAYLOAD_PREVIEW_BYTES))

    data = dict(out.get("data") or {})
    data["hot_path_externalized"] = True
    data["hot_path_external_ref"] = external_ref
    data["hot_path_original_payload_size_bytes"] = payload_bytes
    data["hot_path_original_record_size_bytes"] = record_bytes
    out["data"] = data

    with PAYLOAD_EXTERNALIZATION_LOCK:
        PAYLOAD_EXTERNALIZATION_STATS["events_externalized"] += 1
        PAYLOAD_EXTERNALIZATION_STATS["original_payload_bytes"] += payload_bytes
        PAYLOAD_EXTERNALIZATION_STATS["original_record_bytes"] += record_bytes
    return out


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


def event_type_cache_dir(input_path: Path) -> Path:
    resolved = str(input_path.resolve())
    digest = hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:16]
    return OUT / "event_type_cache" / digest


def event_type_cache_fingerprint(input_path: Path, files: list[Path]) -> dict[str, Any]:
    total_size = 0
    newest_mtime_ns = 0
    for p in files:
        st = p.stat()
        total_size += st.st_size
        newest_mtime_ns = max(newest_mtime_ns, st.st_mtime_ns)
    return {
        "input_path": str(input_path.resolve()),
        "file_count": len(files),
        "total_size": total_size,
        "newest_mtime_ns": newest_mtime_ns,
    }


def ensure_event_type_cache(input_path: Path, files: list[Path]) -> Path:
    cache_dir = event_type_cache_dir(input_path)
    manifest_path = cache_dir / "manifest.json"
    fingerprint = event_type_cache_fingerprint(input_path, files)
    if manifest_path.exists():
        try:
            with manifest_path.open("r", encoding="utf-8") as f:
                manifest = json.load(f)
            if manifest.get("fingerprint") == fingerprint:
                return cache_dir
        except Exception:
            pass

    tmp_dir = cache_dir.with_name(cache_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    handles: dict[str, Any] = {}
    counts: dict[str, int] = {}
    started = time.perf_counter()
    next_progress = started + 30.0
    scanned = 0
    try:
        for path in files:
            for ev in iter_jsonl(path):
                scanned += 1
                et = event_type(ev) or "unknown"
                safe_et = sanitize_collection_name(et)
                handle = handles.get(safe_et)
                if handle is None:
                    handle = (tmp_dir / f"{safe_et}.jsonl").open("w", encoding="utf-8")
                    handles[safe_et] = handle
                handle.write(json.dumps(ev, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
                counts[et] = counts.get(et, 0) + 1
                now = time.perf_counter()
                if now >= next_progress:
                    elapsed = max(now - started, 1e-9)
                    print(
                        f"[cache] indexed {scanned} events from {len(files)} files "
                        f"({scanned / elapsed:.1f} events/s)",
                        file=sys.stderr,
                        flush=True,
                    )
                    next_progress = now + 30.0
        for handle in handles.values():
            handle.close()
        handles.clear()
        write_json(tmp_dir / "manifest.json", {"fingerprint": fingerprint, "counts": counts})
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        tmp_dir.rename(cache_dir)
        print(
            f"[cache] built event-type cache at {cache_dir} counts={counts}",
            file=sys.stderr,
            flush=True,
        )
        return cache_dir
    finally:
        for handle in handles.values():
            handle.close()


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
    if input_path.is_dir() and event_types and len(event_types) == 1 and max_files == 0:
        cache_dir = ensure_event_type_cache(input_path, files)
        only_type = next(iter(event_types))
        cache_file = cache_dir / f"{sanitize_collection_name(only_type)}.jsonl"
        if not cache_file.exists():
            return []
        files = [cache_file]
        event_types = None
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


def iter_events(
    input_path: Path,
    limit: int = 0,
    event_types: set[str] | None = None,
    max_files: int = 0,
) -> Iterable[dict[str, Any]]:
    files = list_jsonl_inputs(input_path)
    if max_files > 0:
        files = files[:max_files]
    if input_path.is_dir() and event_types and len(event_types) == 1 and max_files == 0:
        cache_dir = ensure_event_type_cache(input_path, files)
        only_type = next(iter(event_types))
        cache_file = cache_dir / f"{sanitize_collection_name(only_type)}.jsonl"
        files = [cache_file] if cache_file.exists() else []
        event_types = None
    emitted = 0
    for path in files:
        for ev in iter_jsonl(path):
            et = event_type(ev)
            if event_types and et not in event_types:
                continue
            yield ev
            emitted += 1
            if limit > 0 and emitted >= limit:
                return


def count_events(
    input_path: Path,
    limit: int = 0,
    event_types: set[str] | None = None,
    max_files: int = 0,
) -> int:
    if input_path.is_dir() and event_types and len(event_types) == 1 and max_files == 0:
        files = list_jsonl_inputs(input_path)
        cache_dir = ensure_event_type_cache(input_path, files)
        manifest_path = cache_dir / "manifest.json"
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as f:
                manifest = json.load(f)
            counts = manifest.get("counts") or {}
            total = int(counts.get(next(iter(event_types)), 0))
            return min(total, limit) if limit > 0 else total
    return sum(1 for _ in iter_events(input_path, limit=limit, event_types=event_types, max_files=max_files))


def prefix_string(value: Any, run_id: str) -> Any:
    if not isinstance(value, str) or not value:
        return value
    if value.startswith(run_id + "_"):
        return value
    return f"{run_id}_{value}"


def namespace_event(ev: dict[str, Any], run_id: str, mode: str | None = None) -> dict[str, Any]:
    doc = dict(ev)
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
            doc = set_path_copy(doc, path, prefix_string(value, run_id))

    for path in ["causality.causal_refs", "causality.provenance_refs", "causality.source_object_ids", "causality.target_object_ids", "materialization.planned_object_ids"]:
        value = get_path(doc, path)
        if isinstance(value, list):
            doc = set_path_copy(doc, path, [prefix_string(v, run_id) if isinstance(v, str) else v for v in value])

    for path in ["actor.session_id", "retrieval.retrieval_namespace", "identity.import_batch_id"]:
        value = get_path(doc, path)
        if isinstance(value, str) and value:
            doc = set_path_copy(doc, path, prefix_string(value, run_id))

    if mode:
        doc = set_path_copy(doc, "access.consistency", mode)
    doc = set_path_copy(doc, "runtime.t_write_start_ms", wall_ms())
    return maybe_externalize_large_event(doc)


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
    target_object_ids: set[str]
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
    start_ms: float = 0.0
    end_ms: float = 0.0


class HTTPJSONClient:
    def __init__(self, base_url: str, timeout: float = 30.0, retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = max(0, retries)

    def request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        url = self.base_url + path
        data = None
        headers = {}
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        last_error: BaseException | None = None
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(url, data=data, headers=headers, method=method)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read()
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"{method} {path} failed: HTTP {exc.code}: {detail}") from exc
            except (TimeoutError, urllib.error.URLError, OSError) as exc:
                last_error = exc
                if attempt >= self.retries:
                    raise
                time.sleep(min(0.25 * (2 ** attempt), 2.0))
        else:
            raise RuntimeError(f"{method} {path} failed: {last_error}")
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
            "target_object_ids": sorted(q.target_object_ids),
            "response_mode": "objects_only",
        }
        body = {k: v for k, v in body.items() if v not in ("", [], None)}
        t0 = now_ms()
        try:
            resp = self.http.request("POST", "/v1/query", body)
            t1 = now_ms()
            ids = response_ids(resp)
            visible = bool(ids & q.expected_ids) or json_contains_any(resp, q.expected_ids)
            return QueryResult(self.name, q.query_type, t1 - t0, True, visible, not visible, start_ms=t0, end_ms=t1), resp
        except Exception as exc:
            t1 = now_ms()
            return QueryResult(self.name, q.query_type, t1 - t0, False, False, True, str(exc), start_ms=t0, end_ms=t1), {}

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


def truncate_utf8(value: str, max_bytes: int) -> str:
    raw = (value or "").encode("utf-8")
    if len(raw) <= max_bytes:
        return value or ""
    return raw[:max_bytes].decode("utf-8", errors="ignore")


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
        visibility_policy: str = "flush_on_ack",
        index_type: str = "HNSW",
        payload_json_bytes: int = 65535,
    ):
        from pymilvus import MilvusClient

        self.uri = uri
        self.collection_name = collection_name
        self.timeout = timeout
        self.embedder = embedder or MiniLMEmbedder()
        self.visibility_policy = visibility_policy
        self.index_type = index_type.upper()
        self.payload_json_bytes = payload_json_bytes
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
        if self.index_type == "FLAT":
            index_params.add_index("vector", index_type="FLAT", metric_type="COSINE", params={})
        else:
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
            "pk": truncate_utf8(eid or oid or hashlib.sha1(payload_json.encode("utf-8")).hexdigest(), 256),
            "vector": vector,
            "event_id": truncate_utf8(eid, 512),
            "object_id": truncate_utf8(oid, 512),
            "session_id": truncate_utf8(session_id(ev), 512),
            "agent_id": truncate_utf8(agent_id(ev), 256),
            "workspace_id": truncate_utf8(workspace_id(ev), 256),
            "tenant_id": truncate_utf8(tenant_id(ev), 256),
            "event_type": truncate_utf8(event_type(ev), 128),
            "object_type": truncate_utf8(object_type(ev), 128),
            "version": int(event_version(ev)),
            "text": truncate_utf8(payload_text(ev), 8192),
            "payload_json": truncate_utf8(payload_json, self.payload_json_bytes) if self.payload_json_bytes > 0 else "",
        }

    def ingest(self, ev: dict[str, Any]) -> IngestResult:
        return self._ingest(ev, enforce_visibility=True)

    def _ingest(self, ev: dict[str, Any], enforce_visibility: bool) -> IngestResult:
        t0 = now_ms()
        eid = event_id(ev)
        oid = object_id(ev)
        try:
            vector = self.embedder.embed_one(payload_text(ev))
            row = self._row_for_event(ev, vector)
            with self.mu:
                self.client.insert(self.collection_name, [row], timeout=self.timeout)
                if enforce_visibility and self.visibility_policy == "flush_on_ack":
                    self.client.flush(self.collection_name, timeout=self.timeout)
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
            if q.target_object_ids:
                target_values = milvus_string_list(sorted(q.target_object_ids))
                clauses.append(f"(event_id in {target_values} or object_id in {target_values} or pk in {target_values})")
            expr = " and ".join(clauses) if clauses else ""
            if q.target_object_ids:
                rows = self.client.query(
                    self.collection_name,
                    filter=expr,
                    output_fields=["pk", "event_id", "object_id", "session_id", "event_type", "object_type", "version", "text"],
                    timeout=self.timeout,
                )
                rows = [rows]
            else:
                vector = self.embedder.embed_one(q.query_text or "latest")
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
            return QueryResult(self.name, q.query_type, t1 - t0, True, visible, not visible, start_ms=t0, end_ms=t1), resp
        except Exception as exc:
            t1 = now_ms()
            return QueryResult(self.name, q.query_type, t1 - t0, False, False, True, str(exc), start_ms=t0, end_ms=t1), {}

    def replay_events(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        t0 = now_ms()
        applied = 0
        failed = 0
        for ev in events:
            res = self._ingest(ev, enforce_visibility=False)
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
            return QueryResult(self.name, q.query_type, t1 - t0, True, visible, not visible, start_ms=t0, end_ms=t1), resp
        except Exception as exc:
            t1 = now_ms()
            return QueryResult(self.name, q.query_type, t1 - t0, False, False, True, str(exc), start_ms=t0, end_ms=t1), {}

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
    embedding_cache: Path | None = DEFAULT_EMBEDDING_CACHE,
    embedding_batch_size: int = 64,
    embedding_provider: str = "minilm",
    milvus_visibility_policy: str = "flush_on_ack",
    milvus_index_type: str = "HNSW",
    milvus_payload_json_bytes: int = 65535,
) -> SystemAdapter:
    if system == "plasmod":
        return PlasmodAdapter(base_url, timeout=timeout)
    if system in {"milvus", "vector_metadata", "baseline"}:
        if embedding_provider == "hash":
            embedder = HashEmbedder()
        elif embedding_provider == "minilm":
            embedder = MiniLMEmbedder(embedder_model, embedder_vocab)
        else:
            raise ValueError(f"unknown embedding provider: {embedding_provider}")
        if embedding_provider == "minilm" and embedding_cache is not None:
            embedder = CachedEmbedder(
                embedder,
                embedding_cache,
                embedder_model,
                embedder_vocab,
                batch_size=embedding_batch_size,
            )
        return MilvusAdapter(
            uri=milvus_uri,
            collection_name=collection_name_from_path(sqlite_path),
            timeout=timeout,
            embedder=embedder,
            visibility_policy=milvus_visibility_policy,
            index_type=milvus_index_type,
            payload_json_bytes=milvus_payload_json_bytes,
        )
    if system == "sqlite_metadata":
        return VectorMetadataAdapter(sqlite_path)
    raise ValueError(f"unknown system: {system}")


def prewarm_adapter_embeddings(adapter: SystemAdapter, events: list[dict[str, Any]]) -> dict[str, Any] | None:
    embedder = getattr(adapter, "embedder", None)
    prewarm = getattr(embedder, "prewarm_texts", None)
    if prewarm is None:
        return None
    return prewarm(payload_text(ev) for ev in events)


def embedding_cache_arg(args: argparse.Namespace) -> Path | None:
    if getattr(args, "no_embedding_cache", False):
        return None
    return getattr(args, "embedding_cache", DEFAULT_EMBEDDING_CACHE)


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
    if et and et not in object_types:
        object_types.append(et)
    expected = {event_id(ev), object_id(ev)}
    expected = {x for x in expected if x}
    return QuerySpec(
        query_id=query_id,
        query_type=qtype,
        query_text=payload_text(ev),
        session_id=session_id(ev),
        agent_id=agent_id(ev),
        workspace_id=workspace_id(ev),
        tenant_id=tenant_id(ev),
        object_types=[x for x in object_types if x],
        target_object_ids=set(expected),
        expected_ids=set(expected),
        expected_version=event_version(ev),
        source_event_type=et,
    )


def ingest_with_rate(
    adapter: SystemAdapter,
    events: Iterable[dict[str, Any]],
    rate_eps: float,
    workers: int,
    on_complete: Callable[[dict[str, Any], IngestResult], None] | None = None,
    progress_label: str = "",
    total_events: int | None = None,
    fail_fast: bool = True,
) -> list[IngestResult]:
    if total_events is None:
        try:
            total_events = len(events)  # type: ignore[arg-type]
        except TypeError:
            total_events = None
    interval = 0.0 if rate_eps <= 0 else 1.0 / rate_eps
    start = time.perf_counter()
    pending: list[tuple[int, dict[str, Any], Future[IngestResult]]] = []
    results: list[IngestResult | None] = [None] * total_events if total_events is not None else []
    unordered_results: dict[int, IngestResult] = {}
    effective_workers = max(1, workers)
    if adapter.name == "Milvus" and getattr(adapter, "visibility_policy", "") == "flush_on_ack":
        effective_workers = 1
    max_pending = max(effective_workers * 8, effective_workers)
    completed = 0
    submitted = 0
    next_progress = start + 30.0

    def collect_one(block: bool = True) -> bool:
        nonlocal completed, next_progress
        if not pending:
            return False
        idx, _ev, fut = pending[0]
        if not block and not fut.done():
            return False
        pending.pop(0)
        res = fut.result()
        if total_events is not None:
            results[idx] = res
        else:
            unordered_results[idx] = res
        if fail_fast and not res.ok:
            raise RuntimeError(f"{progress_label or adapter.name}: write failed for event_id={res.event_id}: {res.error}")
        completed += 1
        now = time.perf_counter()
        if progress_label and now >= next_progress:
            elapsed = max(now - start, 1e-9)
            denominator = str(total_events) if total_events is not None else str(submitted)
            print(
                f"[progress] {progress_label}: {completed}/{denominator} ingests completed "
                f"({completed / elapsed:.1f} events/s observed)",
                file=sys.stderr,
                flush=True,
            )
            next_progress = now + 30.0
        return True

    with ThreadPoolExecutor(max_workers=effective_workers) as pool:
        for idx, ev in enumerate(events):
            submitted = idx + 1
            target = start + idx * interval
            delay = target - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            fut = pool.submit(adapter.ingest, ev)
            if on_complete is not None:
                fut.add_done_callback(lambda done, event=ev: on_complete(event, done.result()))
            pending.append((idx, ev, fut))
            while len(pending) >= max_pending:
                collect_one(block=True)
        while pending:
            collect_one(block=True)
    if total_events is not None:
        return [r for r in results[:submitted] if r is not None]
    return [unordered_results[idx] for idx in sorted(unordered_results)]


def ingest_with_visibility_probe(
    adapter: SystemAdapter,
    events: Iterable[dict[str, Any]],
    run_id: str,
    rate_eps: float,
    workers: int,
    timeout_ms: float,
    poll_ms: float,
    total_events: int | None = None,
) -> tuple[list[IngestResult], list[QueryResult]]:
    visibility_futures: list[Future[QueryResult]] = []
    visibility_mu = threading.Lock()
    visibility_pool = ThreadPoolExecutor(max_workers=max(1, workers))

    def on_complete(ev: dict[str, Any], res: IngestResult) -> None:
        if not res.ok:
            return
        with visibility_mu:
            visibility_futures.append(
                visibility_pool.submit(wait_until_visible, adapter, ev, res, run_id, timeout_ms, poll_ms)
            )

    try:
        ingests = ingest_with_rate(
            adapter,
            events,
            rate_eps,
            workers,
            on_complete=on_complete,
            progress_label=f"{run_id}/{adapter.name}",
            total_events=total_events,
            fail_fast=True,
        )
        with visibility_mu:
            pending = list(visibility_futures)
        queries = [future.result() for future in pending]
        return ingests, queries
    finally:
        visibility_pool.shutdown(wait=True)


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
    query.target_object_ids |= ingest_result.expected_ids
    deadline = now_ms() + timeout_ms
    last: QueryResult | None = None
    while now_ms() <= deadline:
        qr, _ = adapter.query(query)
        last = qr
        if not qr.ok:
            return qr
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
        t = now_ms()
        last = QueryResult(adapter.name, query.query_type, 0.0, False, False, True, "visibility timeout", start_ms=t, end_ms=t)
    return last


def validate_measurements_or_raise(
    args: argparse.Namespace,
    table: str,
    context: str,
    ingests: list[IngestResult],
    queries: list[QueryResult] | None = None,
) -> None:
    failed_ingests = [r for r in ingests if not r.ok]
    if failed_ingests and not args.allow_write_errors:
        first = failed_ingests[0]
        raise RuntimeError(f"{table} {context}: write failed for event_id={first.event_id}: {first.error}")
    if queries is not None:
        failed_queries = [q for q in queries if not q.ok]
        if failed_queries and not args.allow_query_errors:
            first_q = failed_queries[0]
            raise RuntimeError(f"{table} {context}: query failed type={first_q.query_type}: {first_q.error}")
    visibility_timeouts = [r for r in ingests if r.visibility_censored]
    if visibility_timeouts and not args.allow_visibility_timeouts:
        first = visibility_timeouts[0]
        raise RuntimeError(
            f"{table} {context}: visibility timeout for event_id={first.event_id}; "
            "increase --visible-timeout-ms only if timeout-censored metrics are intentional"
        )


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
    visibility_probe_count = len(w2v)
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
        "visibility_probe_count": visibility_probe_count,
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


def query_window_qps(queries: list[QueryResult]) -> float:
    ok = [q for q in queries if q.ok]
    if not ok:
        return 0.0
    start = min(q.start_ms for q in ok)
    end = max(q.end_ms for q in ok)
    window_ms = end - start
    if window_ms <= 0:
        window_ms = sum(max(q.latency_ms, 0.0) for q in ok)
    return zero_if_none(safe_div(float(len(ok)), max(window_ms / 1000.0, 1e-9)))


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
            embedding_cache_arg(args),
            args.embedding_batch_size,
            args.embedding_provider,
            args.milvus_visibility_policy,
            args.milvus_index_type,
            args.milvus_payload_json_bytes,
        )
        if system == "plasmod":
            adapter.health()
        types = TABLE4_PLASMOD_TYPES if system == "plasmod" else TABLE4_BASELINE_TYPES
        for et in types:
            run_id = f"{args.run_id}_t4_{system}_{et}"
            event_count = count_events(args.synthetic_input, limit=args.events_per_type, event_types={et})
            print(
                f"[table4] start system={system} event_type={et} events={event_count}",
                file=sys.stderr,
                flush=True,
            )
            if args.shuffle:
                raw = load_events(
                    args.synthetic_input,
                    limit=args.events_per_type,
                    event_types={et},
                    shuffle=args.shuffle,
                    seed=args.seed,
                )
                events: Iterable[dict[str, Any]] = [namespace_event(ev, run_id) for ev in raw]
            else:
                events = (
                    namespace_event(ev, run_id)
                    for ev in iter_events(args.synthetic_input, limit=args.events_per_type, event_types={et})
                )
            if args.shuffle:
                prewarm_adapter_embeddings(adapter, events)  # type: ignore[arg-type]
            if args.reset_between_runs:
                adapter.reset()
            ingests, queries = ingest_with_visibility_probe(
                adapter,
                events,
                run_id,
                args.write_rate,
                args.workers,
                args.visible_timeout_ms,
                args.visible_poll_ms,
                total_events=event_count,
            )
            validate_measurements_or_raise(args, "table4", f"{system}/{et}", ingests, queries)
            row = summarize_ingests(adapter.name, et.replace("state_update", "state"), ingests, queries)
            row.update({"table": "table4", "event_type": et.replace("state_update", "state")})
            rows.append(row)
            write_csv(run_dir / "table4_event_ingestion_visibility.csv", rows)
            print(
                f"[table4] done system={system} event_type={et} "
                f"write_p95_ms={row['write_p95_ms']:.3f} "
                f"w2v_p95_ms={row['write_to_visible_p95_ms']:.3f} "
                f"timeouts={row['visibility_timeouts']} errors={row['write_errors']}",
                file=sys.stderr,
                flush=True,
            )
        adapter.close()
    write_csv(run_dir / "table4_event_ingestion_visibility.csv", rows)
    return rows


def run_freshness_trial(
    adapter: SystemAdapter,
    events: Iterable[dict[str, Any]],
    run_id: str,
    write_rate: float,
    query_qps: float,
    workers: int,
    query_limit: int,
    visible_timeout_ms: float,
    visible_poll_ms: float,
    total_events: int | None = None,
    visibility_probe_limit: int = 5000,
) -> tuple[list[IngestResult], list[QueryResult], float]:
    completed: deque[tuple[dict[str, Any], IngestResult]] = deque()
    completed_mu = threading.Lock()
    ingest_done = threading.Event()
    query_done = threading.Event()
    query_results: list[QueryResult] = []
    q_interval = 0.0 if query_qps <= 0 else 1.0 / query_qps
    visibility_futures: list[Future[QueryResult]] = []
    visibility_mu = threading.Lock()
    probe_mu = threading.Lock()
    visibility_pool = ThreadPoolExecutor(max_workers=max(1, workers))
    if total_events is None:
        try:
            total_events = len(events)  # type: ignore[arg-type]
        except TypeError:
            total_events = None
    if query_limit <= 0:
        query_goal = total_events
    elif total_events is None:
        query_goal = query_limit
    else:
        query_goal = min(query_limit, total_events)
    enqueued_queries = 0
    completed_writes = 0
    submitted_visibility = 0
    visibility_stride = 1
    if total_events is not None and visibility_probe_limit > 0:
        visibility_stride = max(1, (total_events + visibility_probe_limit - 1) // visibility_probe_limit)

    def should_probe_visibility(write_index: int) -> bool:
        if visibility_probe_limit <= 0:
            return True
        if write_index == 1:
            return True
        if submitted_visibility >= visibility_probe_limit:
            return False
        return write_index % visibility_stride == 0

    def on_complete(ev: dict[str, Any], res: IngestResult) -> None:
        nonlocal completed_writes, enqueued_queries, submitted_visibility
        if res.ok:
            with probe_mu:
                completed_writes += 1
                write_index = completed_writes
            with completed_mu:
                if not query_done.is_set() and (query_goal is None or enqueued_queries < query_goal):
                    completed.append((ev, res))
                    enqueued_queries += 1
            with probe_mu:
                probe_visibility = should_probe_visibility(write_index)
                if probe_visibility:
                    submitted_visibility += 1
            if probe_visibility:
                with visibility_mu:
                    visibility_futures.append(
                        visibility_pool.submit(wait_until_visible, adapter, ev, res, run_id, visible_timeout_ms, visible_poll_ms)
                    )

    def query_loop() -> None:
        idx = 0
        next_query_at = time.perf_counter()
        next_progress = time.perf_counter() + 30.0
        while True:
            if query_goal is not None and idx >= query_goal:
                query_done.set()
                break
            with completed_mu:
                item = completed.popleft() if completed else None
                enqueued = enqueued_queries
            if item is None:
                if ingest_done.is_set() and idx >= enqueued:
                    break
                next_query_at = time.perf_counter()
                time.sleep(0.001)
                continue
            if q_interval > 0:
                delay = next_query_at - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                next_query_at = max(next_query_at + q_interval, time.perf_counter() + q_interval)
            ev, res = item
            q = query_for_event(ev, run_id, f"q_{idx}")
            q.expected_ids |= res.expected_ids
            q.target_object_ids |= res.expected_ids
            qr, _ = adapter.query(q)
            query_results.append(qr)
            idx += 1
            now = time.perf_counter()
            if now >= next_progress:
                denominator = query_goal if query_goal is not None else enqueued
                print(
                    f"[progress] {run_id}/{adapter.name}: {idx}/{denominator} queries completed",
                    file=sys.stderr,
                    flush=True,
                )
                next_progress = now + 30.0
            if not qr.ok:
                query_done.set()
                break
        query_done.set()

    t0 = now_ms()
    qt = threading.Thread(target=query_loop, daemon=True)
    qt.start()
    ingests = ingest_with_rate(
        adapter,
        events,
        write_rate,
        workers,
        on_complete=on_complete,
        progress_label=f"{run_id}/{adapter.name}",
        total_events=total_events,
        fail_fast=True,
    )
    ingest_done.set()
    qt.join()
    if query_goal is not None and len(query_results) < query_goal:
        raise RuntimeError(
            f"{run_id}/{adapter.name}: query workload incomplete "
            f"({len(query_results)}/{query_goal}); first query error="
            f"{next((q.error for q in query_results if not q.ok), 'none')}"
        )
    with visibility_mu:
        pending_visibility = list(visibility_futures)
    for fut in pending_visibility:
        fut.result()
    visibility_pool.shutdown(wait=True)
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
            embedding_cache_arg(args),
            args.embedding_batch_size,
            args.embedding_provider,
            args.milvus_visibility_policy,
            args.milvus_index_type,
            args.milvus_payload_json_bytes,
        )
        if system == "plasmod":
            adapter.health()
        for rate in args.write_rates:
            run_id = f"{args.run_id}_t5_{system}_{rate}"
            event_count = count_events(args.synthetic_input, limit=args.events_per_rate)
            print(
                f"[table5] start system={system} write_rate={rate} events={event_count}",
                file=sys.stderr,
                flush=True,
            )
            if args.shuffle:
                raw = load_events(args.synthetic_input, limit=args.events_per_rate, shuffle=args.shuffle, seed=args.seed)
                events: Iterable[dict[str, Any]] = [namespace_event(ev, run_id) for ev in raw]
                prewarm_adapter_embeddings(adapter, events)  # type: ignore[arg-type]
            else:
                events = (namespace_event(ev, run_id) for ev in iter_events(args.synthetic_input, limit=args.events_per_rate))
            if args.reset_between_runs:
                adapter.reset()
            ingests, queries, _elapsed_s = run_freshness_trial(
                adapter,
                events,
                run_id,
                rate,
                args.query_qps,
                args.workers,
                args.query_limit,
                args.visible_timeout_ms,
                args.visible_poll_ms,
                total_events=event_count,
                visibility_probe_limit=args.visibility_probe_limit,
            )
            if args.shuffle:
                fill_missing_visibility(adapter, events, ingests, run_id, args.visible_timeout_ms, args.visible_poll_ms, args.workers)  # type: ignore[arg-type]
            validate_measurements_or_raise(args, "table5", f"{system}/rate={rate}", ingests, queries)
            ingest_row = summarize_ingests(adapter.name, str(rate), ingests, queries)
            query_row = summarize_queries(adapter.name, str(rate), queries)
            row = {
                "table": "table5",
                "system": adapter.name,
                "write_rate_events_s": rate,
                "query_qps": query_window_qps(queries),
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
                "visibility_probe_count": ingest_row["visibility_probe_count"],
                "visibility_timeouts": ingest_row["visibility_timeouts"],
                "visibility_measurement_mode": ingest_row["visibility_measurement_mode"],
                "queries": len(queries),
            }
            rows.append(row)
            write_csv(run_dir / "table5_freshness_under_write_load.csv", rows)
            print(
                f"[table5] done system={adapter.name} write_rate={rate} "
                f"events={len(ingests)} query_p95_ms={row['query_p95_ms']:.3f} "
                f"w2v_p95_ms={row['write_to_visible_p95_ms']:.3f} "
                f"stale={row['stale_result_rate']:.6f} timeouts={row['visibility_timeouts']} errors={row['write_errors']}",
                file=sys.stderr,
                flush=True,
            )
        adapter.close()
    write_csv(run_dir / "table5_freshness_under_write_load.csv", rows)
    return rows


def run_table6(args: argparse.Namespace, run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    event_count = count_events(args.synthetic_input, limit=args.events_per_rate)
    raw: list[dict[str, Any]] | None = None
    if args.shuffle:
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
            print(
                f"[table6] start system=plasmod visibility_mode={mode} events={event_count}",
                file=sys.stderr,
                flush=True,
            )
            if args.shuffle:
                assert raw is not None
                events: Iterable[dict[str, Any]] = [namespace_event(ev, run_id, PLASMOD_MODES[mode]) for ev in raw]
            else:
                events = (
                    namespace_event(ev, run_id, PLASMOD_MODES[mode])
                    for ev in iter_events(args.synthetic_input, limit=args.events_per_rate)
                )
            if args.reset_between_runs:
                adapter.reset()
            ingests, queries, _elapsed_s = run_freshness_trial(
                adapter,
                events,
                run_id,
                args.fixed_write_rate,
                args.query_qps,
                args.workers,
                args.query_limit,
                args.visible_timeout_ms,
                args.visible_poll_ms,
                total_events=event_count,
                visibility_probe_limit=args.visibility_probe_limit,
            )
            if args.shuffle:
                fill_missing_visibility(adapter, events, ingests, run_id, args.visible_timeout_ms, args.visible_poll_ms, args.workers)  # type: ignore[arg-type]
            validate_measurements_or_raise(args, "table6", f"plasmod/{mode}", ingests, queries)
            qrow = summarize_queries(adapter.name, mode, queries)
            irow = summarize_ingests(adapter.name, mode, ingests, queries)
            row = {
                "table": "table6",
                "system": "Plasmod",
                "visibility_mode": {"strict": "Strict", "bounded": "Bounded Staleness", "eventual": "Eventual"}[mode],
                "write_qps": irow["write_qps"],
                "query_qps": query_window_qps(queries),
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
                "visibility_probe_count": irow["visibility_probe_count"],
                "visibility_timeouts": irow["visibility_timeouts"],
                "visibility_measurement_mode": irow["visibility_measurement_mode"],
                "queries": len(queries),
            }
            rows.append(row)
            write_csv(run_dir / "table6_consistency_mode_tradeoff.csv", rows)
            print(
                f"[table6] done system=Plasmod visibility_mode={mode} "
                f"query_p95_ms={row['query_p95_ms']:.3f} w2v_p95_ms={row['write_to_visible_p95_ms']:.3f} "
                f"stale={row['stale_result_rate']:.6f} timeouts={row['visibility_timeouts']} errors={row['write_errors']}",
                file=sys.stderr,
                flush=True,
            )
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
            embedding_cache_arg(args),
            args.embedding_batch_size,
            args.embedding_provider,
            args.milvus_visibility_policy,
            args.milvus_index_type,
            args.milvus_payload_json_bytes,
        )
        run_id = f"{args.run_id}_t6_milvus_best_effort"
        print(
            f"[table6] start system=milvus visibility_mode=best_effort events={event_count}",
            file=sys.stderr,
            flush=True,
        )
        if args.shuffle:
            assert raw is not None
            events = [namespace_event(ev, run_id) for ev in raw]
            prewarm_adapter_embeddings(adapter, events)
        else:
            events = (namespace_event(ev, run_id) for ev in iter_events(args.synthetic_input, limit=args.events_per_rate))
        if args.reset_between_runs:
            adapter.reset()
        ingests, queries, _elapsed_s = run_freshness_trial(
            adapter,
            events,
            run_id,
            args.fixed_write_rate,
            args.query_qps,
            args.workers,
            args.query_limit,
            args.visible_timeout_ms,
            args.visible_poll_ms,
            total_events=event_count,
            visibility_probe_limit=args.visibility_probe_limit,
        )
        if args.shuffle:
            fill_missing_visibility(adapter, events, ingests, run_id, args.visible_timeout_ms, args.visible_poll_ms, args.workers)  # type: ignore[arg-type]
        validate_measurements_or_raise(args, "table6", "milvus/best_effort", ingests, queries)
        qrow = summarize_queries(adapter.name, "best_effort", queries)
        irow = summarize_ingests(adapter.name, "best_effort", ingests, queries)
        row = {
            "table": "table6",
            "system": adapter.name,
            "visibility_mode": "Best-effort",
            "write_qps": irow["write_qps"],
            "query_qps": query_window_qps(queries),
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
            "visibility_probe_count": irow["visibility_probe_count"],
            "visibility_timeouts": irow["visibility_timeouts"],
            "visibility_measurement_mode": irow["visibility_measurement_mode"],
            "queries": len(queries),
        }
        rows.append(row)
        write_csv(run_dir / "table6_consistency_mode_tradeoff.csv", rows)
        print(
            f"[table6] done system={adapter.name} visibility_mode=best_effort "
            f"query_p95_ms={row['query_p95_ms']:.3f} w2v_p95_ms={row['write_to_visible_p95_ms']:.3f} "
            f"stale={row['stale_result_rate']:.6f} timeouts={row['visibility_timeouts']} errors={row['write_errors']}",
            file=sys.stderr,
            flush=True,
        )
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
        validate_measurements_or_raise(args, "table7", "plasmod/seed", ingests)
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
            embedding_cache_arg(args),
            args.embedding_batch_size,
            args.embedding_provider,
            args.milvus_visibility_policy,
            args.milvus_index_type,
            args.milvus_payload_json_bytes,
        )
        prewarm_adapter_embeddings(baseline, events)
        baseline.reset()
        replay_out = baseline.replay_events(events)
        if replay_out["failed"] and not args.allow_write_errors:
            raise RuntimeError(f"table7 milvus/service_restart: replay failed for {replay_out['failed']} events")
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
    raw_full: list[dict[str, Any]] | None = None
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
        if len(selected) < args.table8_updates and args.replay_events > 0:
            if raw_full is None:
                raw_full = load_events(args.replay_input, limit=0, shuffle=False)
            selected = select_table8_events(raw_full, selector)[: args.table8_updates]
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
            embedding_cache_arg(args),
            args.embedding_batch_size,
            args.embedding_provider,
            args.milvus_visibility_policy,
            args.milvus_index_type,
            args.milvus_payload_json_bytes,
        )
        if system == "plasmod":
            adapter.health()
        prewarm_adapter_embeddings(adapter, events)
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
            q.target_object_ids |= res.expected_ids
            qr, _ = adapter.query(q)
            queries.append(qr)
        validate_measurements_or_raise(args, "table8", f"{system}/{selector}", ingests, queries)
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


def run_prepare_embeddings(args: argparse.Namespace) -> None:
    if args.no_embedding_cache:
        raise ValueError("prepare-embeddings requires embedding cache to be enabled")
    synthetic = load_events(
        args.synthetic_input,
        limit=args.synthetic_limit,
        shuffle=args.shuffle,
        seed=args.seed,
        max_files=args.max_files,
    )
    replay = load_events(
        args.replay_input,
        limit=args.replay_limit,
        shuffle=False,
        max_files=args.max_files,
    )
    texts = [payload_text(ev) for ev in synthetic]
    texts.extend(payload_text(ev) for ev in replay)
    if args.embedding_provider == "hash":
        embedder = HashEmbedder()
    elif args.embedding_provider == "minilm":
        embedder = CachedEmbedder(
            MiniLMEmbedder(args.embedder_model, args.embedder_vocab),
            args.embedding_cache,
            args.embedder_model,
            args.embedder_vocab,
            batch_size=args.embedding_batch_size,
        )
    else:
        raise ValueError(f"unknown embedding provider: {args.embedding_provider}")
    summary = embedder.prewarm_texts(texts)
    summary.update({
        "synthetic_events": len(synthetic),
        "replay_events": len(replay),
        "embedding_cache_enabled": True,
    })
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


def run_tables(args: argparse.Namespace) -> Path:
    global HOT_EVENT_MAX_BYTES, HOT_PAYLOAD_PREVIEW_BYTES
    HOT_EVENT_MAX_BYTES = args.max_hot_event_bytes
    HOT_PAYLOAD_PREVIEW_BYTES = args.hot_payload_preview_bytes
    with PAYLOAD_EXTERNALIZATION_LOCK:
        for key in PAYLOAD_EXTERNALIZATION_STATS:
            PAYLOAD_EXTERNALIZATION_STATS[key] = 0

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
        "milvus_visibility_policy": args.milvus_visibility_policy,
        "milvus_index_type": args.milvus_index_type,
        "milvus_payload_json_bytes": args.milvus_payload_json_bytes,
        "embedder_model": str(args.embedder_model),
        "embedder_vocab": str(args.embedder_vocab),
        "embedding_cache": "disabled" if args.no_embedding_cache else str(args.embedding_cache),
        "embedding_batch_size": args.embedding_batch_size,
        "allow_write_errors": args.allow_write_errors,
        "allow_query_errors": args.allow_query_errors,
        "allow_visibility_timeouts": args.allow_visibility_timeouts,
        "write_rates": args.write_rates,
        "query_qps": args.query_qps,
        "max_hot_event_bytes": args.max_hot_event_bytes,
        "hot_payload_preview_bytes": args.hot_payload_preview_bytes,
        "created_at_ms": wall_ms(),
        "notes": [
            "write-to-visible is measured by polling /v1/query until expected ids appear",
            "materialization lag uses first-visible time as an external black-box proxy when no materialized timestamp is returned",
            "large artifact payloads are represented by hash/size/ref plus preview on the online write path",
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
    with PAYLOAD_EXTERNALIZATION_LOCK:
        metadata["payload_externalization_stats"] = dict(PAYLOAD_EXTERNALIZATION_STATS)
    write_json(run_dir / "run_metadata.json", metadata)
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

    def add_embedding_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--embedder-model", type=Path, default=DEFAULT_EMBEDDER_MODEL)
        p.add_argument("--embedder-vocab", type=Path, default=DEFAULT_EMBEDDER_VOCAB)
        p.add_argument(
            "--embedding-provider",
            choices=["minilm", "hash"],
            default="minilm",
            help="Milvus vector source. minilm uses cached ONNX embeddings; hash uses deterministic vectors for Layer 2 object-visibility experiments.",
        )
        p.add_argument("--embedding-cache", type=Path, default=DEFAULT_EMBEDDING_CACHE)
        p.add_argument("--embedding-batch-size", type=int, default=64)
        p.add_argument("--no-embedding-cache", action="store_true", help="Disable reusable embedding cache; useful only for debugging.")

    prepare = sub.add_parser("prepare-embeddings", help="Precompute and persist reusable MiniLM embeddings for Layer 2 data.")
    add_common(prepare)
    add_embedding_args(prepare)
    prepare.add_argument("--synthetic-limit", type=int, default=0)
    prepare.add_argument("--replay-limit", type=int, default=0)
    prepare.add_argument("--max-files", type=int, default=0)

    run = sub.add_parser("run", help="Run one or more Layer 2 experiment tables.")
    add_common(run)
    add_embedding_args(run)
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
    run.add_argument(
        "--visibility-probe-limit",
        type=int,
        default=5000,
        help="Maximum per-run write-to-visible probe count for Table 5/6; 0 probes every successful write.",
    )
    run.add_argument(
        "--max-hot-event-bytes",
        type=int,
        default=DEFAULT_MAX_HOT_EVENT_BYTES,
        help="Externalize agent artifact payload bodies above this byte size on the online write path; 0 disables.",
    )
    run.add_argument(
        "--hot-payload-preview-bytes",
        type=int,
        default=DEFAULT_HOT_PAYLOAD_PREVIEW_BYTES,
        help="UTF-8 preview bytes retained when a large agent payload is externalized.",
    )
    run.add_argument("--http-timeout", type=float, default=30.0)
    run.add_argument("--milvus-uri", default="http://127.0.0.1:19530")
    run.add_argument(
        "--milvus-visibility-policy",
        choices=["flush_on_ack", "deferred"],
        default="flush_on_ack",
        help="flush_on_ack makes Milvus writes read-visible before ACK; deferred measures best-effort async visibility.",
    )
    run.add_argument(
        "--milvus-index-type",
        choices=["HNSW", "FLAT"],
        default="HNSW",
        help="Milvus vector index for baseline collections. FLAT avoids background HNSW builds in visibility-only workloads.",
    )
    run.add_argument(
        "--milvus-payload-json-bytes",
        type=int,
        default=65535,
        help="Bytes of full event JSON to store in Milvus metadata; use 0 for vector+metadata pointer-style baselines.",
    )
    run.add_argument("--reset-between-runs", action="store_true")
    run.add_argument("--allow-write-errors", action="store_true", help="Record write failures instead of stopping the run.")
    run.add_argument("--allow-query-errors", action="store_true", help="Record query failures instead of stopping the run.")
    run.add_argument("--allow-visibility-timeouts", action="store_true", help="Record timeout-censored visibility metrics instead of stopping the run.")
    return ap.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.cmd == "analyze":
        run_analyze(args)
        return 0
    if args.cmd == "prepare-embeddings":
        run_prepare_embeddings(args)
        return 0
    if args.cmd == "run":
        run_dir = run_tables(args)
        print(run_dir)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
