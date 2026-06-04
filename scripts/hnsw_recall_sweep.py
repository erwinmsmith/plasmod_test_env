#!/usr/bin/env python3
from __future__ import annotations

"""
HNSW recall/QPS sweep for the 1M-index / 10k-query deep10M experiment.

Default run:
  python scripts/hnsw_recall_sweep.py

The script intentionally lives outside benchmark_all.py because Plasmod HNSW
search ef is currently process-scoped (PLASMOD_HNSW_EF_SEARCH), while Qdrant and
Milvus support per-query ef. Results record that distinction explicitly.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
BASE = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import benchmark_all as bench  # noqa: E402


DEFAULT_TARGETS = "0.8,0.85,0.9,0.95,1.0"
DEFAULT_EF_VALUES = "12,16,24,32,48,64,96,128,192,256,384,512"
DEFAULT_DBS = "qdrant,milvus,plasmod,chromadb,lancedb"


@dataclass
class SweepPoint:
    method: str
    db: str
    sweep_param: str
    sweep_value: int
    recall: float
    batch_ms: float
    batch_qps: float
    serial_ms: float | None
    serial_qps: float | None
    p50_ms: float | None
    p95_ms: float | None
    p99_ms: float | None
    n_indexed: int
    n_queries: int
    dim: int
    topk: int
    build_ms: float | None = None
    notes: str = ""


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_ints(raw: str) -> list[int]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    if not values:
        raise ValueError("empty integer list")
    return values


def parse_floats(raw: str) -> list[float]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    if not values:
        raise ValueError("empty float list")
    return values


def parse_dbs(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return DEFAULT_DBS.split(",")
    valid = {"qdrant", "milvus", "plasmod", "chromadb", "lancedb"}
    out = []
    for part in raw.split(","):
        db = part.strip().lower()
        if not db:
            continue
        if db not in valid:
            raise ValueError(f"unknown db {db!r}; valid={sorted(valid)}")
        out.append(db)
    if not out:
        raise ValueError("empty db list")
    return out


def valid_ef_values(raw_values: list[int], topk: int) -> list[int]:
    values = [ef for ef in raw_values if ef > topk]
    skipped = [ef for ef in raw_values if ef <= topk]
    if skipped:
        _log(f"Skipping HNSW ef values <= topk ({topk}): {skipped}")
    if not values:
        raise ValueError(f"no valid HNSW ef values; ef must be > topk ({topk})")
    return values


def rows_to_jsonable(points: Iterable[SweepPoint]) -> list[dict[str, Any]]:
    return [asdict(p) for p in points]


def load_deep_or_nfcorpus(dataset: str, index_count: int, queries_count: int):
    if dataset == "nfcorpus":
        base = bench.DATA / "nfcorpus"
        indexed, in_n, dim = bench.load_fbin(str(base / "corpus.fbin"), index_count or 0)
        queries, qn, _ = bench.load_fbin(str(base / "queries.fbin"), queries_count)
    else:
        base = bench.DATA / "deep"
        indexed, in_n, dim = bench.load_fbin(str(base / "base.10M.fbin"), index_count or 0)
        queries, qn, _ = bench.load_fbin(str(base / "query.public.10K.fbin"), queries_count)
    n_idx = index_count or in_n
    n_q = min(qn, queries_count)
    return indexed[: n_idx * dim], n_idx, dim, queries[: n_q * dim], n_q


def compute_ground_truth(indexed, n_idx: int, dim: int, queries, n_q: int, topk: int, cache_path: Path):
    if cache_path.exists():
        _log(f"Loading cached ground truth: {cache_path}")
        import numpy as np

        arr = np.load(cache_path)
        return arr[:, :topk].tolist()

    _log("Computing brute-force ground truth for the indexed subset...")
    gt = bench.brute_force_search(indexed, n_idx, dim, queries, n_q, topk)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import numpy as np

        np.save(cache_path, np.asarray(gt, dtype=np.int32))
        _log(f"Ground truth cached: {cache_path}")
    except Exception as e:
        _log(f"Ground truth cache skipped: {e}")
    return gt


def iter_query_chunks(queries, n_q: int, dim: int, chunk_size: int):
    for start in range(0, n_q, chunk_size):
        end = min(start + chunk_size, n_q)
        yield start, end, queries[start * dim : end * dim]


def _percentiles(latencies: list[float]) -> tuple[float | None, float | None, float | None]:
    if not latencies:
        return None, None, None
    return bench._percentiles_ms(latencies)


def _serial_stats_qdrant(queries, n_q: int, dim: int, topk: int, ef: int, sample_n: int, coll: str):
    if sample_n <= 0:
        return None, None, None, None, None
    sample_n = min(sample_n, n_q)
    latencies: list[float] = []
    base = "http://127.0.0.1:6333"
    for q in range(sample_n):
        body = bench._json_dumps({
            "vector": queries[q * dim : (q + 1) * dim],
            "top": topk,
            "with_vectors": False,
            "params": {"hnsw": {"ef": ef, "exact": False}},
        })
        t0 = time.time()
        urllib.request.urlopen(
            urllib.request.Request(
                f"{base}/collections/{coll}/points/search",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            ),
            timeout=120,
        ).read()
        latencies.append((time.time() - t0) * 1000)
    serial_ms = (sum(latencies) / sample_n) * n_q
    p50, p95, p99 = _percentiles(latencies)
    return serial_ms, n_q / (serial_ms / 1000), p50, p95, p99


def build_qdrant(indexed, n_idx: int, dim: int, ef_construction: int, coll: str) -> float:
    base = "http://127.0.0.1:6333"
    try:
        urllib.request.urlopen(
            urllib.request.Request(
                f"{base}/collections/{coll}",
                method="DELETE",
                headers={"Content-Type": "application/json"},
            ),
            timeout=30,
        ).read()
    except Exception:
        pass

    body = {
        "vectors": {"size": dim, "distance": "Cosine"},
        "hnsw_config": {
            "m": 16,
            "ef_construct": ef_construction,
            "full_scan_threshold": 10000,
        },
    }
    urllib.request.urlopen(
        urllib.request.Request(
            f"{base}/collections/{coll}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="PUT",
        ),
        timeout=60,
    ).read()

    _log(f"[Qdrant] ingesting {n_idx:,} vectors...")
    t0 = time.time()
    report_step = max(5000, n_idx // 10)
    next_report = report_step
    for start in range(0, n_idx, 500):
        end = min(start + 500, n_idx)
        points = [
            {"id": i + 1, "vector": indexed[i * dim : (i + 1) * dim]}
            for i in range(start, end)
        ]
        urllib.request.urlopen(
            urllib.request.Request(
                f"{base}/collections/{coll}/points",
                data=bench._json_dumps({"points": points}),
                headers={"Content-Type": "application/json"},
                method="PUT",
            ),
            timeout=120,
        ).read()
        if end >= next_report:
            _log(f"[Qdrant]   {end:,}/{n_idx:,} ingested")
            next_report += report_step
    try:
        urllib.request.urlopen(f"{base}/collections/{coll}/flush", timeout=60).read()
    except Exception:
        pass
    return (time.time() - t0) * 1000


def run_qdrant(indexed, n_idx: int, dim: int, queries, n_q: int, topk: int,
               gt: list[list[int]], ef_values: list[int], ef_construction: int,
               chunk_size: int, serial_samples: int) -> list[SweepPoint]:
    coll = "bench_hnsw_recall_sweep"
    build_ms = build_qdrant(indexed, n_idx, dim, ef_construction, coll)
    base = "http://127.0.0.1:6333"
    points: list[SweepPoint] = []
    for ef in ef_values:
        _log(f"[Qdrant] search ef={ef}")
        all_ids: list[list[int]] = []
        t0 = time.time()
        for start, end, qchunk in iter_query_chunks(queries, n_q, dim, chunk_size):
            chunk_n = end - start
            payload = {
                "searches": [
                    {
                        "vector": qchunk[(i * dim) : ((i + 1) * dim)],
                        "top": topk,
                        "with_vectors": False,
                        "params": {"hnsw": {"ef": ef, "exact": False}},
                    }
                    for i in range(chunk_n)
                ]
            }
            resp = urllib.request.urlopen(
                urllib.request.Request(
                    f"{base}/collections/{coll}/points/search/batch",
                    data=bench._json_dumps(payload),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                ),
                timeout=300,
            )
            data = json.loads(resp.read())
            all_ids.extend([[pt["id"] - 1 for pt in res] for res in data.get("result", [])])
        batch_ms = (time.time() - t0) * 1000
        recall = bench.recall_at_k(all_ids, gt, topk)
        serial_ms, serial_qps, p50, p95, p99 = _serial_stats_qdrant(
            queries, n_q, dim, topk, ef, serial_samples, coll
        )
        points.append(SweepPoint(
            method="Qdrant HNSW",
            db="Qdrant",
            sweep_param="query_ef",
            sweep_value=ef,
            recall=recall,
            batch_ms=batch_ms,
            batch_qps=n_q / (batch_ms / 1000) if batch_ms > 0 else 0.0,
            serial_ms=serial_ms,
            serial_qps=serial_qps,
            p50_ms=p50,
            p95_ms=p95,
            p99_ms=p99,
            n_indexed=n_idx,
            n_queries=n_q,
            dim=dim,
            topk=topk,
            build_ms=build_ms,
            notes="per-query ef is supported",
        ))
    return points


def build_milvus(indexed, n_idx: int, dim: int, ef_construction: int, coll: str):
    from pymilvus import MilvusClient
    from pymilvus.milvus_client.index import IndexParams

    client = MilvusClient(uri="http://127.0.0.1:19530")
    try:
        client.drop_collection(coll)
    except Exception:
        pass
    ip = IndexParams()
    ip.add_index("vector", "HNSW", m=16, efConstruction=ef_construction)
    client.create_collection(coll, dimension=dim, index_params=ip, auto_id=False)
    client.flush(coll)

    _log(f"[Milvus] ingesting {n_idx:,} vectors...")
    t0 = time.time()
    report_step = max(5000, n_idx // 10)
    next_report = report_step
    for start in range(0, n_idx, 500):
        end = min(start + 500, n_idx)
        rows = [{"id": i, "vector": indexed[i * dim : (i + 1) * dim]} for i in range(start, end)]
        client.insert(coll, rows)
        if end >= next_report:
            _log(f"[Milvus]   {end:,}/{n_idx:,} ingested")
            next_report += report_step
    client.flush(coll)
    try:
        client.load_collection(coll)
    except Exception:
        pass
    return client, (time.time() - t0) * 1000


def _serial_stats_milvus(client, coll: str, qvecs: list[Any], n_q: int, topk: int, ef: int, sample_n: int):
    if sample_n <= 0:
        return None, None, None, None, None
    sample_n = min(sample_n, n_q)
    latencies: list[float] = []
    for qvec in qvecs[:sample_n]:
        t0 = time.time()
        client.search(coll, [qvec], limit=topk, search_params={"ef": ef})
        latencies.append((time.time() - t0) * 1000)
    serial_ms = (sum(latencies) / sample_n) * n_q
    p50, p95, p99 = _percentiles(latencies)
    return serial_ms, n_q / (serial_ms / 1000), p50, p95, p99


def run_milvus(indexed, n_idx: int, dim: int, queries, n_q: int, topk: int,
               gt: list[list[int]], ef_values: list[int], ef_construction: int,
               chunk_size: int, serial_samples: int) -> list[SweepPoint]:
    coll = "bench_hnsw_recall_sweep"
    client, build_ms = build_milvus(indexed, n_idx, dim, ef_construction, coll)
    qvecs = [queries[q * dim : (q + 1) * dim] for q in range(n_q)]
    points: list[SweepPoint] = []
    for ef in ef_values:
        _log(f"[Milvus] search ef={ef}")
        all_ids: list[list[int]] = []
        t0 = time.time()
        for start in range(0, n_q, chunk_size):
            end = min(start + chunk_size, n_q)
            res = client.search(coll, qvecs[start:end], limit=topk, search_params={"ef": ef})
            all_ids.extend([[hit["id"] for hit in row] for row in res])
        batch_ms = (time.time() - t0) * 1000
        recall = bench.recall_at_k(all_ids, gt, topk)
        serial_ms, serial_qps, p50, p95, p99 = _serial_stats_milvus(
            client, coll, qvecs, n_q, topk, ef, serial_samples
        )
        points.append(SweepPoint(
            method="Milvus HNSW",
            db="Milvus",
            sweep_param="query_ef",
            sweep_value=ef,
            recall=recall,
            batch_ms=batch_ms,
            batch_qps=n_q / (batch_ms / 1000) if batch_ms > 0 else 0.0,
            serial_ms=serial_ms,
            serial_qps=serial_qps,
            p50_ms=p50,
            p95_ms=p95,
            p99_ms=p99,
            n_indexed=n_idx,
            n_queries=n_q,
            dim=dim,
            topk=topk,
            build_ms=build_ms,
            notes="per-query ef is supported",
        ))
    return points


def restart_plasmod_for_ef(ef: int) -> None:
    _log(f"[Plasmod] restarting server with PLASMOD_HNSW_EF_SEARCH={ef}")
    subprocess.run(["bash", "stop_server.sh"], cwd=BASE, check=True)
    env = os.environ.copy()
    env["PLASMOD_HNSW_EF_SEARCH"] = str(ef)
    subprocess.run(["bash", "start_server.sh"], cwd=BASE, env=env, check=True)
    bench.require_plasmod_server()


def build_plasmod(indexed, n_idx: int, dim: int, ef_construction: int, seg_id: str):
    http = bench._HTTPClient("http://127.0.0.1:8080", timeout=3600)
    http.unload(seg_id)
    t0 = time.time()
    ok, payload_ms, http_ms, code, body = http.ingest(
        seg_id,
        indexed[: n_idx * dim],
        n_idx,
        dim,
        index_type="HNSW",
        ef_construction=ef_construction,
    )
    if not ok:
        msg = body[:1000].decode(errors="replace")
        raise RuntimeError(f"Plasmod ingest failed: status={code} body={msg}")
    return http, (time.time() - t0) * 1000, payload_ms, http_ms


def _query_plasmod_chunks(http, seg_id: str, queries, n_q: int, dim: int, topk: int,
                          chunk_size: int, raw: bool):
    all_ids: list[int] = []
    t0 = time.time()
    for start, end, qchunk in iter_query_chunks(queries, n_q, dim, chunk_size):
        chunk_n = end - start
        if raw:
            ok, _, _, ids, _ = http.query_batch_raw(seg_id, qchunk, chunk_n, dim, topk)
        else:
            ok, _, _, ids, _ = http.query_batch(seg_id, qchunk, chunk_n, dim, topk)
        if not ok:
            mode = "raw" if raw else "optimized"
            raise RuntimeError(f"Plasmod {mode} batch query failed at chunk {start}:{end}")
        all_ids.extend(ids)
    batch_ms = (time.time() - t0) * 1000
    return [all_ids[i * topk : (i + 1) * topk] for i in range(n_q)], batch_ms


def _serial_stats_plasmod(http, seg_id: str, queries, n_q: int, dim: int, topk: int, sample_n: int):
    if sample_n <= 0:
        return None, None, None, None, None
    sample_n = min(sample_n, n_q)
    latencies: list[float] = []
    for q in range(sample_n):
        ok, _, lat_ms, _ = http.query_serial(seg_id, queries[q * dim : (q + 1) * dim], dim, topk)
        if ok:
            latencies.append(lat_ms)
    if not latencies:
        return None, None, None, None, None
    serial_ms = (sum(latencies) / len(latencies)) * n_q
    p50, p95, p99 = _percentiles(latencies)
    return serial_ms, n_q / (serial_ms / 1000), p50, p95, p99


def run_plasmod(indexed, n_idx: int, dim: int, queries, n_q: int, topk: int,
                gt: list[list[int]], ef_values: list[int], ef_construction: int,
                chunk_size: int, serial_samples: int, modes: list[str],
                restart_per_ef: bool) -> list[SweepPoint]:
    original_ef = int(os.getenv("PLASMOD_HNSW_EF_SEARCH", "96"))
    if not restart_per_ef:
        current_ef = original_ef
        if current_ef not in ef_values:
            _log(f"[Plasmod] adding current process ef assumption {current_ef} to sweep")
        ef_values = [current_ef]
        notes = (
            "Plasmod HNSW efSearch is process-scoped; this point assumes the "
            "running server uses PLASMOD_HNSW_EF_SEARCH from this environment. "
            "Use --restart-plasmod-per-ef for a true ef sweep."
        )
    else:
        notes = "Plasmod server restarted per ef; index rebuilt per point"

    points: list[SweepPoint] = []
    try:
        for ef in ef_values:
            if restart_per_ef:
                restart_plasmod_for_ef(ef)
            _log(f"[Plasmod] build HNSW and search ef={ef}")
            seg_id = "bench.hnsw_recall_sweep"
            http, build_ms, payload_ms, http_ms = build_plasmod(indexed, n_idx, dim, ef_construction, seg_id)
            try:
                for mode in modes:
                    raw = mode == "raw"
                    got_ids, batch_ms = _query_plasmod_chunks(
                        http, seg_id, queries, n_q, dim, topk, chunk_size, raw=raw
                    )
                    recall = bench.recall_at_k(got_ids, gt, topk)
                    if raw:
                        serial_ms = serial_qps = p50 = p95 = p99 = None
                    else:
                        serial_ms, serial_qps, p50, p95, p99 = _serial_stats_plasmod(
                            http, seg_id, queries, n_q, dim, topk, serial_samples
                        )
                    points.append(SweepPoint(
                        method=f"Plasmod HNSW {mode}",
                        db="Plasmod",
                        sweep_param="process_ef_search",
                        sweep_value=ef,
                        recall=recall,
                        batch_ms=batch_ms,
                        batch_qps=n_q / (batch_ms / 1000) if batch_ms > 0 else 0.0,
                        serial_ms=serial_ms,
                        serial_qps=serial_qps,
                        p50_ms=p50,
                        p95_ms=p95,
                        p99_ms=p99,
                        n_indexed=n_idx,
                        n_queries=n_q,
                        dim=dim,
                        topk=topk,
                        build_ms=build_ms,
                        notes=f"{notes}; ingest_payload_ms={payload_ms:.1f}; ingest_http_ms={http_ms:.1f}",
                    ))
            finally:
                http.unload(seg_id)
                http.close()
    finally:
        if restart_per_ef:
            restart_plasmod_for_ef(original_ef)
    return points


def build_chromadb(indexed, n_idx: int, dim: int, ef_construction: int, ef_search: int, coll_name: str):
    import chromadb

    client = chromadb.PersistentClient(path=str(BASE / "chromadb_hnsw_sweep_data"))
    try:
        client.delete_collection(coll_name)
    except Exception:
        pass
    time.sleep(1.0)
    coll = client.create_collection(
        coll_name,
        configuration={
            "hnsw": {
                "space": "cosine",
                "ef_construction": ef_construction,
                "ef_search": ef_search,
                "max_neighbors": 16,
            }
        },
    )

    _log(f"[ChromaDB] ingesting {n_idx:,} vectors...")
    t0 = time.time()
    report_step = max(5000, n_idx // 10)
    next_report = report_step
    for start in range(0, n_idx, 500):
        end = min(start + 500, n_idx)
        ids = [f"vec_{i}" for i in range(start, end)]
        vectors = [
            [float(indexed[i * dim + d]) for d in range(dim)]
            for i in range(start, end)
        ]
        coll.add(ids=ids, embeddings=vectors)
        if end >= next_report:
            _log(f"[ChromaDB]   {end:,}/{n_idx:,} ingested")
            next_report += report_step
    return client, coll, (time.time() - t0) * 1000


def _query_chromadb(coll, queries, n_q: int, dim: int, topk: int, chunk_size: int):
    got_ids: list[list[int]] = []
    t0 = time.time()
    for start, end, qchunk in iter_query_chunks(queries, n_q, dim, chunk_size):
        qvecs = [
            [float(qchunk[i * dim + d]) for d in range(dim)]
            for i in range(end - start)
        ]
        res = coll.query(query_embeddings=qvecs, n_results=topk, include=[])
        got_ids.extend([[int(id_.split("_", 1)[1]) for id_ in row] for row in res.get("ids", [])])
    return got_ids, (time.time() - t0) * 1000


def _serial_stats_chromadb(coll, queries, n_q: int, dim: int, topk: int, sample_n: int):
    if sample_n <= 0:
        return None, None, None, None, None
    sample_n = min(sample_n, n_q)
    latencies: list[float] = []
    for q in range(sample_n):
        qvec = [float(queries[q * dim + d]) for d in range(dim)]
        t0 = time.time()
        coll.query(query_embeddings=[qvec], n_results=topk, include=[])
        latencies.append((time.time() - t0) * 1000)
    serial_ms = (sum(latencies) / sample_n) * n_q
    p50, p95, p99 = _percentiles(latencies)
    return serial_ms, n_q / (serial_ms / 1000), p50, p95, p99


def run_chromadb(indexed, n_idx: int, dim: int, queries, n_q: int, topk: int,
                 gt: list[list[int]], ef_values: list[int], ef_construction: int,
                 chunk_size: int, serial_samples: int) -> list[SweepPoint]:
    coll_name = "bench_hnsw_recall_sweep"
    _, coll, build_ms = build_chromadb(indexed, n_idx, dim, ef_construction, ef_values[0], coll_name)
    points: list[SweepPoint] = []
    for ef in ef_values:
        _log(f"[ChromaDB] search ef={ef}")
        try:
            coll.modify(configuration={"hnsw": {"ef_search": ef}})
            time.sleep(0.2)
        except Exception as e:
            _log(f"[ChromaDB] ef_search modify failed for ef={ef}: {e}; continuing with collection default")
        got_ids, batch_ms = _query_chromadb(coll, queries, n_q, dim, topk, chunk_size)
        recall = bench.recall_at_k(got_ids, gt, topk)
        serial_ms, serial_qps, p50, p95, p99 = _serial_stats_chromadb(
            coll, queries, n_q, dim, topk, serial_samples
        )
        points.append(SweepPoint(
            method="ChromaDB HNSW",
            db="ChromaDB",
            sweep_param="collection_ef_search",
            sweep_value=ef,
            recall=recall,
            batch_ms=batch_ms,
            batch_qps=n_q / (batch_ms / 1000) if batch_ms > 0 else 0.0,
            serial_ms=serial_ms,
            serial_qps=serial_qps,
            p50_ms=p50,
            p95_ms=p95,
            p99_ms=p99,
            n_indexed=n_idx,
            n_queries=n_q,
            dim=dim,
            topk=topk,
            build_ms=build_ms,
            notes="ChromaDB HNSW ef_search is collection configuration, not per-query search params",
        ))
    return points


def build_lancedb(indexed, n_idx: int, dim: int, ef_construction: int, coll_name: str):
    import lancedb
    import pyarrow as pa

    db = lancedb.connect(str(BASE / "lancedb_hnsw_sweep_data"))
    try:
        db.drop_table(coll_name)
    except Exception:
        pass

    schema = pa.schema([
        ("id", pa.int64()),
        ("vector", pa.list_(pa.float32(), dim)),
    ])
    _log(f"[LanceDB] ingesting {n_idx:,} vectors...")
    t0 = time.time()
    report_step = max(5000, n_idx // 10)
    next_report = report_step
    table = None
    for start in range(0, n_idx, 500):
        end = min(start + 500, n_idx)
        ids = list(range(start, end))
        vectors = [
            [float(indexed[i * dim + d]) for d in range(dim)]
            for i in range(start, end)
        ]
        batch = pa.table({"id": ids, "vector": vectors}, schema=schema)
        if table is None:
            table = db.create_table(coll_name, data=batch)
        else:
            table.add(batch)
        if end >= next_report:
            _log(f"[LanceDB]   {end:,}/{n_idx:,} ingested")
            next_report += report_step
    table = db.open_table(coll_name)
    _log("[LanceDB] building HNSW index...")
    table.create_index(
        metric="cosine",
        vector_column_name="vector",
        index_type="IVF_HNSW_PQ",
        replace=True,
        num_partitions=1,
        num_sub_vectors=min(dim, 96),
        m=16,
        ef_construction=ef_construction,
    )
    try:
        table.wait_for_index()
    except Exception:
        pass
    return table, (time.time() - t0) * 1000


def _query_lancedb(table, queries, n_q: int, dim: int, topk: int, ef: int):
    got_ids: list[list[int]] = []
    t0 = time.time()
    for q in range(n_q):
        qvec = [float(queries[q * dim + d]) for d in range(dim)]
        rows = table.search(qvec).ef(ef).limit(topk).to_list()
        got_ids.append([int(row["id"]) for row in rows])
    return got_ids, (time.time() - t0) * 1000


def _serial_stats_lancedb(table, queries, n_q: int, dim: int, topk: int, ef: int, sample_n: int):
    if sample_n <= 0:
        return None, None, None, None, None
    sample_n = min(sample_n, n_q)
    latencies: list[float] = []
    for q in range(sample_n):
        qvec = [float(queries[q * dim + d]) for d in range(dim)]
        t0 = time.time()
        table.search(qvec).ef(ef).limit(topk).to_list()
        latencies.append((time.time() - t0) * 1000)
    serial_ms = (sum(latencies) / sample_n) * n_q
    p50, p95, p99 = _percentiles(latencies)
    return serial_ms, n_q / (serial_ms / 1000), p50, p95, p99


def run_lancedb(indexed, n_idx: int, dim: int, queries, n_q: int, topk: int,
                gt: list[list[int]], ef_values: list[int], ef_construction: int,
                serial_samples: int) -> list[SweepPoint]:
    coll_name = "bench_hnsw_recall_sweep"
    table, build_ms = build_lancedb(indexed, n_idx, dim, ef_construction, coll_name)
    points: list[SweepPoint] = []
    for ef in ef_values:
        _log(f"[LanceDB] search ef={ef}")
        got_ids, batch_ms = _query_lancedb(table, queries, n_q, dim, topk, ef)
        recall = bench.recall_at_k(got_ids, gt, topk)
        serial_ms, serial_qps, p50, p95, p99 = _serial_stats_lancedb(
            table, queries, n_q, dim, topk, ef, serial_samples
        )
        points.append(SweepPoint(
            method="LanceDB IVF_HNSW_PQ",
            db="LanceDB",
            sweep_param="query_ef",
            sweep_value=ef,
            recall=recall,
            batch_ms=batch_ms,
            batch_qps=n_q / (batch_ms / 1000) if batch_ms > 0 else 0.0,
            serial_ms=serial_ms,
            serial_qps=serial_qps,
            p50_ms=p50,
            p95_ms=p95,
            p99_ms=p99,
            n_indexed=n_idx,
            n_queries=n_q,
            dim=dim,
            topk=topk,
            build_ms=build_ms,
            notes="LanceDB uses IVF_HNSW_PQ as its HNSW-family vector index",
        ))
    return points


def _metric_value(point: SweepPoint, metric: str) -> float:
    value = getattr(point, metric)
    return float(value) if value is not None else -1.0


def best_points_by_target(points: list[SweepPoint], targets: list[float], metric: str, policy: str):
    grouped: dict[str, list[SweepPoint]] = {}
    for point in points:
        grouped.setdefault(point.method, []).append(point)

    rows = []
    for method, method_points in grouped.items():
        for target in targets:
            qualifying = [p for p in method_points if p.recall >= target]
            achieved = bool(qualifying)

            if achieved:
                if policy == "max_qps":
                    best = max(qualifying, key=lambda p: (_metric_value(p, metric), p.recall))
                else:
                    best = min(qualifying, key=lambda p: (p.recall, -_metric_value(p, metric)))
            else:
                best = max(method_points, key=lambda p: (p.recall, _metric_value(p, metric)))
            rows.append({
                "method": method,
                "target_recall": target,
                "achieved": achieved,
                "selected_recall": best.recall,
                "selected_param": best.sweep_param,
                "selected_value": best.sweep_value,
                "batch_qps": best.batch_qps,
                "serial_qps": best.serial_qps,
                "selection_metric": metric,
                "selection_policy": policy,
                "fallback_reason": "" if achieved else "target not reached; selected best available point",
            })
    return rows


def write_outputs(out_dir: Path, points: list[SweepPoint], targets: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "sweep_points.json", "w") as f:
        json.dump({"metadata": metadata, "points": rows_to_jsonable(points)}, f, indent=2)
    with open(out_dir / "target_summary.json", "w") as f:
        json.dump({"metadata": metadata, "targets": targets}, f, indent=2)
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"metadata": metadata, "target_summary": targets, "points": rows_to_jsonable(points)}, f, indent=2)

    point_rows = rows_to_jsonable(points)
    if point_rows:
        with open(out_dir / "sweep_points.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(point_rows[0].keys()))
            writer.writeheader()
            writer.writerows(point_rows)
    if targets:
        with open(out_dir / "target_summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(targets[0].keys()))
            writer.writeheader()
            writer.writerows(targets)


def print_target_table(target_rows: list[dict[str, Any]]) -> None:
    print("\nTarget recall summary")
    print(f"{'Method':<24} {'Target':>7} {'OK':>3} {'Recall':>8} {'Param':>18} {'Batch QPS':>11} {'Serial QPS':>11}")
    print("-" * 90)
    for row in target_rows:
        serial = row["serial_qps"]
        serial_s = "" if serial is None else f"{serial:.1f}"
        ok = "yes" if row["achieved"] else "no"
        print(
            f"{row['method']:<24} {row['target_recall']:>7.2f} {ok:>3} "
            f"{row['selected_recall']:>8.4f} {row['selected_param']}={row['selected_value']:<5} "
            f"{row['batch_qps']:>11.1f} {serial_s:>11}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Reproducible HNSW recall/QPS sweep for deep10M 1M-index / 10k-query experiments."
    )
    ap.add_argument("--dataset", default="deep10M", choices=["deep10M", "nfcorpus"])
    ap.add_argument("--index-count", type=int, default=1_000_000)
    ap.add_argument("--queries", type=int, default=10_000)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--db", default=DEFAULT_DBS,
                    help=f"Comma-separated db list, or all. Default: {DEFAULT_DBS}")
    ap.add_argument("--targets", default=DEFAULT_TARGETS,
                    help="Comma-separated recall targets. Default treats 0.85 as the intended 0.85 target.")
    ap.add_argument("--ef-search-values", default=DEFAULT_EF_VALUES)
    ap.add_argument("--ef-construction", type=int, default=256)
    ap.add_argument("--batch-chunk-size", type=int, default=1000)
    ap.add_argument("--serial-samples", type=int, default=1000,
                    help="nq=1 latency sample size per point; 0 disables serial stats.")
    ap.add_argument("--select-by", default="batch_qps", choices=["batch_qps", "serial_qps"],
                    help="Metric used to break ties for each recall target.")
    ap.add_argument("--selection-policy", default="closest_above", choices=["closest_above", "max_qps"],
                    help="closest_above selects the lowest recall point above target; max_qps selects fastest above target.")
    ap.add_argument("--output-dir", default="",
                    help="Default: results/hnsw_recall_sweep_<dataset>_n<N>_q<Q>_k<K>_<timestamp>")
    ap.add_argument("--groundtruth-cache", default="",
                    help="Default: results/groundtruth_<dataset>_n<N>_q<Q>_k<K>.npy")
    ap.add_argument("--restart-plasmod-per-ef", action="store_true",
                    help="Restart Plasmod with PLASMOD_HNSW_EF_SEARCH for each ef value.")
    ap.add_argument("--plasmod-modes", default="optimized",
                    help="Comma-separated Plasmod modes: optimized,raw")
    args = ap.parse_args()

    dbs = parse_dbs(args.db)
    targets = parse_floats(args.targets)
    ef_values = valid_ef_values(parse_ints(args.ef_search_values), args.topk)
    plasmod_modes = [m.strip().lower() for m in args.plasmod_modes.split(",") if m.strip()]
    for mode in plasmod_modes:
        if mode not in {"optimized", "raw"}:
            raise ValueError("--plasmod-modes only supports optimized,raw")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else (
        bench.OUT / f"hnsw_recall_sweep_{args.dataset}_n{args.index_count}_q{args.queries}_k{args.topk}_{timestamp}"
    )
    gt_cache = Path(args.groundtruth_cache) if args.groundtruth_cache else (
        bench.OUT / f"groundtruth_{args.dataset}_n{args.index_count}_q{args.queries}_k{args.topk}.npy"
    )

    _log(
        f"Starting HNSW recall sweep dataset={args.dataset} n={args.index_count:,} "
        f"queries={args.queries:,} topk={args.topk} dbs={','.join(dbs)}"
    )
    indexed, n_idx, dim, queries, n_q = load_deep_or_nfcorpus(args.dataset, args.index_count, args.queries)
    gt = compute_ground_truth(indexed, n_idx, dim, queries, n_q, args.topk, gt_cache)

    all_points: list[SweepPoint] = []
    if "qdrant" in dbs:
        all_points.extend(run_qdrant(
            indexed, n_idx, dim, queries, n_q, args.topk, gt, ef_values,
            args.ef_construction, args.batch_chunk_size, args.serial_samples,
        ))
    if "milvus" in dbs:
        all_points.extend(run_milvus(
            indexed, n_idx, dim, queries, n_q, args.topk, gt, ef_values,
            args.ef_construction, args.batch_chunk_size, args.serial_samples,
        ))
    if "plasmod" in dbs:
        bench.require_plasmod_server()
        all_points.extend(run_plasmod(
            indexed, n_idx, dim, queries, n_q, args.topk, gt, ef_values,
            args.ef_construction, args.batch_chunk_size, args.serial_samples,
            plasmod_modes, args.restart_plasmod_per_ef,
        ))
    if "chromadb" in dbs:
        all_points.extend(run_chromadb(
            indexed, n_idx, dim, queries, n_q, args.topk, gt, ef_values,
            args.ef_construction, args.batch_chunk_size, args.serial_samples,
        ))
    if "lancedb" in dbs:
        all_points.extend(run_lancedb(
            indexed, n_idx, dim, queries, n_q, args.topk, gt, ef_values,
            args.ef_construction, args.serial_samples,
        ))

    target_rows = best_points_by_target(all_points, targets, args.select_by, args.selection_policy)
    metadata = {
        "dataset": args.dataset,
        "n_indexed": n_idx,
        "n_queries": n_q,
        "dim": dim,
        "topk": args.topk,
        "targets": targets,
        "ef_search_values": ef_values,
        "ef_construction": args.ef_construction,
        "dbs": dbs,
        "select_by": args.select_by,
        "selection_policy": args.selection_policy,
        "serial_samples": args.serial_samples,
        "batch_chunk_size": args.batch_chunk_size,
        "plasmod_restart_per_ef": args.restart_plasmod_per_ef,
        "notes": [
            "Recall is computed against brute-force ground truth over the indexed subset.",
            "Target 0.85 is used for the user's 9.85/0.85 item.",
            "If a target is not reached, the summary selects the highest-recall available point.",
        ],
    }
    write_outputs(out_dir, all_points, target_rows, metadata)
    print_target_table(target_rows)
    _log(f"Saved results to {out_dir}")


if __name__ == "__main__":
    main()
