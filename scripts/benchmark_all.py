#!/usr/bin/env python3
"""
Benchmark: compare index performance across Qdrant, Milvus, LanceDB, Plasmod.

Datasets:
  nfcorpus: ~3.6K vectors, dim=384
  deep10M:  10M vectors, dim=96

Index types (all 4 DBs support):
  FLAT      — brute-force baseline
  IVF_FLAT   — IVF + flat vectors
  IVF_PQ     — IVF + product quantization

Metrics measured:
  Build(ms)  — Index build time
  Batch(ms)   — Single batch query (all N queries in one call)
  Batch QPS   — N / Batch(ms)
  Serial QPS  — N / sum(serial latencies)
  Recall@K    — Recall vs brute-force ground truth
  P50/P95/P99 — Percentile latencies (serial, per-query)
  Memory(GB)  — Process/resident memory of DB process

Usage:
  python3 scripts/benchmark_all.py --dataset nfcorpus --index flat
  python3 scripts/benchmark_all.py --dataset nfcorpus --index all
  python3 scripts/benchmark_all.py --dataset deep10M --index-count 100000 --index ivf_flat
"""

import argparse
import json
import os
import resource
import struct
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

BASE = Path("/Users/erwin/Downloads/codespace/Plasmodexp/plasmod_test_env")
DATA = BASE / "data"
OUT  = BASE / "results"
OUT.mkdir(exist_ok=True)


# ─── Data loading ──────────────────────────────────────────────────────────────

def load_fbin(path: str, limit: int = 0):
    """Read float32 .fbin: [n(uint32), dim(uint32), vectors...]"""
    with open(path, "rb") as f:
        n   = struct.unpack("<I", f.read(4))[0]
        dim = struct.unpack("<I", f.read(4))[0]
        avail = (os.path.getsize(path) - 8) // (dim * 4)
        n = min(n, avail, limit) if limit else min(n, avail)
        data = f.read(n * dim * 4)
    vecs = list(struct.unpack(f"<{n*dim}f", data))
    return vecs, n, dim


def load_ibin(path: str, nq: int, topk: int):
    """Read int32 .ibin: [nq(uint32), topk(uint32), ids[nq*topk*4bytes]"""
    with open(path, "rb") as f:
        f.read(8)
        data = f.read(nq * topk * 4)
    ids = list(struct.unpack(f"<{nq*topk}i", data))
    return ids


def brute_force_search(indexed: List[float], indexed_n: int, dim: int,
                       queries: List[float], q_n: int, topk: int):
    """Compute ground truth via brute force (cosine similarity)."""
    results = []
    for qi in range(q_n):
        q = queries[qi * dim:(qi + 1) * dim]
        q_norm = sum(x*x for x in q) ** 0.5 or 1
        scores = []
        for i in range(indexed_n):
            v = indexed[i * dim:(i + 1) * dim]
            score = sum(qj * vj for qj, vj in zip(q, v)) / (q_norm * (sum(x*x for x in v) ** 0.5) + 1e-9)
            scores.append((score, i))
        scores.sort(reverse=True)
        results.append([i for _, i in scores[:topk]])
    return results


def recall_at_k(got_ids: List[List[int]], gt_ids: List[List[int]], k: int) -> float:
    hits = 0
    total = 0
    for got, gt in zip(got_ids, gt_ids):
        gt_set = set(gt[:k])
        for gid in got[:k]:
            if gid in gt_set:
                hits += 1
        total += k
    return hits / total if total else 0.0


def _plasmod_recall(int_ids: List[int], indexed: List[float], indexed_n: int,
                    dim: int, queries: List[float], query_n: int, topk: int) -> float:
    """
    Compute recall for Plasmod results.
    Plasmod returns int_ids = [q0_top0, q0_top1, ..., q0_top(k-1), q1_top0, ...]
    These IDs are order-of-ingestion (0-based row indices in the .fbin).
    Map them to corpus indices and compare with brute-force ground truth.
    """
    if not int_ids:
        return 0.0
    got_ids = [int_ids[q * topk:(q + 1) * topk] for q in range(query_n)]
    gt = brute_force_search(indexed, indexed_n, dim, queries, query_n, topk)
    return recall_at_k(got_ids, gt, topk)


# ─── Result ────────────────────────────────────────────────────────────────────

@dataclass
class Result:
    db: str
    index_type: str
    n_indexed: int
    n_queries: int
    dim: int
    topk: int
    build_ms: float
    batch_ms: float       # single batch call for all queries
    batch_qps: float
    serial_ms: float
    serial_qps: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    recall: float          # Recall@K vs ground truth
    memory_mb: float       # Resident memory in MB
    # QPS at different recall levels
    qps_at_recall: dict = None  # {0.5: qps, 0.6: qps, ...}

    def __post_init__(self):
        if self.qps_at_recall is None:
            self.qps_at_recall = {}

    def save(self, out_dir: Path):
        p = out_dir / f"{self.db}_{self.index_type}.json"
        with open(p, "w") as f:
            json.dump(asdict(self), f, indent=2)
        print(f"  -> {p.name}")


def print_table(results: List[Result]):
    print(f"\n{'DB':<10} {'Index':<12} {'Build(s)':>8} {'Batch(ms)':>10} {'BatchQPS':>10} "
          f"{'SerialQPS':>11} {'Recall@K':>10} "
          f"{'Mean(ms)':>9} {'P50':>8} {'P95':>8} {'P99':>8} {'Mem(MB)':>9}")
    print("-" * 112)
    for r in results:
        mean = r.serial_ms / r.n_queries if r.n_queries else 0
        print(f"{r.db:<10} {r.index_type:<12} "
              f"{r.build_ms/1000:>8.3f} {r.batch_ms:>10.1f} {r.batch_qps:>10.1f} "
              f"{r.serial_qps:>11.1f} {r.recall:>10.4f} "
              f"{mean:>9.3f} {r.p50_ms:>8.3f} {r.p95_ms:>8.3f} {r.p99_ms:>8.3f} {r.memory_mb:>9.1f}")


def mem_mb(pid: int) -> float:
    """Return resident memory in MB for a process."""
    try:
        out = subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "rss="], text=True, timeout=5
        )
        return float(out.strip()) / 1024   # KB -> MB
    except Exception:
        return 0.0


def mem_mb_with_mmap(pid: int, data_dir: Path) -> float:
    """
    Return memory in MB for a process, including mmap'd vector storage.
    For Plasmod which uses mmap for vector storage (.mem files), this gives a fairer comparison.
    Only counts .mem files (actual vector data), not .vlog (WAL logs) or other metadata files.
    """
    # RSS (physical memory)
    rss_mb = mem_mb(pid)

    # Only count .mem files for vector storage (not .vlog which is WAL)
    mmap_mb = 0.0
    if data_dir.exists():
        for f in data_dir.iterdir():
            # Only .mem files contain vector data
            if f.suffix == ".mem":
                try:
                    mmap_mb += os.path.getsize(f) / (1024 * 1024)
                except Exception:
                    pass

    return rss_mb + mmap_mb


def compute_recall_qps_sweep(db_name, indexed, indexed_n, dim, queries, query_n, topk,
                             idx_type, recall_thresholds=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0)):
    """
    Compute QPS at different recall levels by sweeping search parameters.

    For IVF: sweep nprobe from 1 to nlist
    For HNSW: sweep ef from 4 to 256

    Returns tuple: (qps_at_recall_dict, sweep_points_list)
      - qps_at_recall_dict: {recall_level: qps_at_that_recall}
      - sweep_points_list: [{param: x, qps: y, recall: z}, ...]
    """

    # Rebuild collection for sweep
    print(f"      [{db_name}] Rebuilding collection for sweep...")
    rebuild_ok = _sweep_rebuild(db_name, indexed, indexed_n, dim, idx_type)
    if not rebuild_ok:
        print(f"      [{db_name}] Failed to rebuild collection")
        return {}, []

    # Compute ground truth once
    gt = brute_force_search(indexed, indexed_n, dim, queries, query_n, topk)

    # Define sweep ranges based on index type
    if idx_type == "hnsw":
        param_range = [4, 8, 16, 32, 64, 128, 256]
        param_name = "ef"
    elif idx_type == "ivf_pq":
        # For IVF-PQ: sweep nlist with more aggressive nprobe to achieve recall=1.0
        param_range = [16, 32, 64, 128, 256, 512, 1024, 2048]
        param_name = "nlist"
    else:
        if indexed_n < 10000:
            max_nprobe = 32
        elif indexed_n < 100000:
            max_nprobe = 128
        else:
            max_nprobe = 256
        param_range = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128]
        param_range = [p for p in param_range if p <= max_nprobe]
        if max_nprobe not in param_range:
            param_range.append(max_nprobe)
        param_name = "nprobe"

    sweep_points = []  # List of {param, qps, recall}

    for param_val in param_range:
        got_ids = _sweep_search(db_name, indexed, indexed_n, dim, queries, query_n, topk, idx_type, param_val, param_name)
        if not got_ids:
            continue

        recall = recall_at_k(got_ids, gt, topk)
        qps = query_n * 1000.0 / (serial_ms_for_param(db_name, indexed_n, dim, queries, query_n, topk, idx_type, param_val, param_name) or 1.0)

        sweep_points.append({"param": param_val, "param_name": param_name, "qps": qps, "recall": recall})
        print(f"      [{db_name}] {param_name}={param_val}: recall={recall:.4f}, QPS={qps:.1f}")

    # Interpolate QPS at each target recall
    result = {}
    for target in recall_thresholds:
        qps_at_target = _interpolate_qps(sweep_points, target)
        result[target] = qps_at_target

    return result, sweep_points


def _sweep_rebuild(db_name, indexed, indexed_n, dim, idx_type):
    """Rebuild collection for sweep."""
    if db_name == "Qdrant":
        return _sweep_rebuild_qdrant(indexed, indexed_n, dim, idx_type)
    elif db_name == "Milvus":
        return _sweep_rebuild_milvus(indexed, indexed_n, dim, idx_type)
    elif db_name == "ChromaDB":
        return _sweep_rebuild_chromadb(indexed, indexed_n, dim, idx_type)
    elif db_name == "Plasmod":
        return _sweep_rebuild_plasmod(indexed, indexed_n, dim, idx_type)
    elif db_name == "LanceDB":
        return _sweep_rebuild_lancedb(indexed, indexed_n, dim, idx_type)
    return False


def _sweep_rebuild_plasmod(indexed, indexed_n, dim, idx_type):
    """Rebuild Plasmod segment for sweep (initial build with default params)."""
    seg_id = "bench.sweep"
    server_url = "http://127.0.0.1:8080"

    # Unload any existing segment
    try:
        data = json.dumps({"segment_id": seg_id}).encode()
        req = urllib.request.Request(
            f"{server_url}/v1/internal/rpc/unload_segment",
            data=data, headers={"Content-Type": "application/json"}, method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass

    # Index type mapping for Plasmod (default params for initial build)
    INDEX_TYPE_MAP = {
        "ivf_flat": ("IVF_FLAT", 32, 8, 0, 0, ""),
        "ivf_pq":   ("IVF_PQ",   32, 8, 16, 8, ""),
        "ivf_sq8":  ("IVF_SQ8",  32, 8, 0, 0, "INT8"),
        "hnsw":     ("HNSW",     0, 0, 0, 0, ""),
        "flat":     ("IVF_FLAT", 32, 8, 0, 0, ""),
    }
    ptype, nlist, nprobe, m, nbits, sq_type = INDEX_TYPE_MAP.get(idx_type, ("HNSW", 0, 0, 0, 0, ""))

    try:
        http = _HTTPClient(server_url, timeout=120)
        # Ingest in chunks
        ingest_batch = min(500_000, indexed_n)
        for start in range(0, indexed_n, ingest_batch):
            end = min(start + ingest_batch, indexed_n)
            batch_n = end - start
            ok, _ = http.ingest(seg_id, indexed[start * dim : end * dim], batch_n, dim,
                                index_type=ptype, nlist=nlist, nprobe=nprobe,
                                m=m, nbits=nbits, sq_type=sq_type)
            if not ok:
                raise RuntimeError(f"ingest failed at batch {start}-{end}")

        # Register warm
        http.register_warm(seg_id, indexed_n)
        return True
    except Exception as e:
        print(f"      Plasmod rebuild error: {e}")
        return False


def _sweep_rebuild_chromadb(indexed, indexed_n, dim, idx_type):
    """Rebuild ChromaDB collection for sweep. Only HNSW is supported by ChromaDB."""
    import chromadb

    chroma_dir = str(BASE / "chromadb_data")
    coll_name = "bench_chroma_sweep"

    client = chromadb.PersistentClient(path=chroma_dir)

    # ChromaDB only supports HNSW, not IVF types
    if idx_type != "hnsw":
        return False

    try:
        try:
            client.delete_collection(coll_name)
        except Exception:
            pass

        # Wait a bit for deletion to complete
        time.sleep(0.5)

        ids = [f"sweep_{i}" for i in range(indexed_n)]
        # Convert numpy floats to plain Python floats for ChromaDB compatibility
        vectors = [[float(indexed[i * dim + d]) for d in range(dim)] for i in range(indexed_n)]
        coll = client.create_collection(name=coll_name, metadata={"hnsw:space": "cosine"})

        for bs in range(0, indexed_n, 500):
            be = min(bs + 500, indexed_n)
            coll.add(ids=ids[bs:be], embeddings=vectors[bs:be])

        return True
    except Exception as e:
        print(f"      ChromaDB rebuild error: {e}")
        return False


def _sweep_rebuild_lancedb(indexed, indexed_n, dim, idx_type):
    """Rebuild LanceDB table for sweep."""
    import lancedb
    import pyarrow as pa

    db = lancedb.connect(str(BASE / "lancedb_data"))
    table_name = "bench_lancedb_sweep"

    try:
        db.drop_table(table_name)
    except Exception:
        pass

    try:
        ids = list(range(indexed_n))
        vectors = [[indexed[i * dim + d] for d in range(dim)] for i in range(indexed_n)]

        schema = pa.schema([
            ("id", pa.int64()),
            ("vector", pa.list_(pa.float32(), dim)),
        ])

        # Build table
        for bs in range(0, indexed_n, 500):
            be = min(bs + 500, indexed_n)
            tbl = pa.table({"id": ids[bs:be], "vector": vectors[bs:be]}, schema=schema)
            if bs == 0:
                db.create_table(table_name, data=tbl)
            else:
                db.open_table(table_name).add(tbl)

        # Create index based on idx_type
        tbl_ref = db.open_table(table_name)
        if idx_type == "hnsw":
            tbl_ref.create_index(metric="cosine", vector_column_name="vector",
                                 index_type="IVF_HNSW_PQ",
                                 num_partitions=1, num_sub_vectors=min(96, dim),
                                 replace=True)
        elif idx_type == "ivf_flat":
            tbl_ref.create_index(metric="cosine", vector_column_name="vector",
                                 index_type="IVF_FLAT", num_partitions=1,
                                 replace=True)
        elif idx_type == "ivf_pq":
            tbl_ref.create_index(metric="cosine", vector_column_name="vector",
                                 index_type="IVF_PQ", num_partitions=1, num_sub_vectors=8,
                                 replace=True)
        elif idx_type == "ivf_sq8":
            tbl_ref.create_index(metric="cosine", vector_column_name="vector",
                                 index_type="IVF_SQ", num_partitions=1,
                                 replace=True)
        else:
            # flat/no index
            pass

        try:
            tbl_ref.wait_for_index()
        except Exception:
            pass

        return True
    except Exception as e:
        print(f"      LanceDB rebuild error: {e}")
        return False


def _sweep_rebuild_qdrant(indexed, indexed_n, dim, idx_type):
    """Rebuild Qdrant collection."""
    base = "http://127.0.0.1:6333"
    coll = "bench_test"

    try:
        # Delete collection
        try:
            urllib.request.urlopen(
                urllib.request.Request(f"{base}/collections/{coll}", method="DELETE",
                                     headers={"Content-Type": "application/json"}),
                timeout=10)
        except:
            pass

        # Create collection with appropriate index type
        if idx_type == "hnsw":
            req = {"vectors": {"size": dim, "distance": "Cosine"},
                   "hnsw_config": {"m": 16, "ef_construct": 256}}
        elif idx_type == "ivf_pq":
            # Qdrant uses HNSW + Product Quantization (scalar quantization with product mode)
            req = {"vectors": {"size": dim, "distance": "Cosine"},
                   "hnsw_config": {"m": 16, "ef_construct": 256},
                   "quantization_config": {"scalar": {"type": "int8", "quantization": "product"}}}
        elif idx_type == "ivf_sq8":
            # Qdrant uses HNSW + Scalar Quantization (int8)
            req = {"vectors": {"size": dim, "distance": "Cosine"},
                   "hnsw_config": {"m": 16, "ef_construct": 256},
                   "quantization_config": {"scalar": {"type": "int8"}}}
        else:  # ivf_flat
            req = {"vectors": {"size": dim, "distance": "Cosine"},
                   "hnsw_config": {"m": 16, "ef_construct": 256}}

        urllib.request.urlopen(
            urllib.request.Request(f"{base}/collections/{coll}",
                                  data=json.dumps(req).encode(),
                                  headers={"Content-Type": "application/json"}, method="PUT"),
            timeout=30).read()

        # Ingest
        for batch_start in range(0, indexed_n, 500):
            batch_end = min(batch_start + 500, indexed_n)
            points = [{"id": i + 1, "vector": indexed[i * dim:(i + 1) * dim]}
                      for i in range(batch_start, batch_end)]
            body = json.dumps({"points": points}).encode()
            urllib.request.urlopen(
                urllib.request.Request(f"{base}/collections/{coll}/points", data=body,
                                      headers={"Content-Type": "application/json"}, method="PUT"),
                timeout=60)

        # Flush
        try:
            urllib.request.urlopen(f"{base}/collections/{coll}/flush", timeout=30)
        except:
            pass

        return True
    except Exception as e:
        print(f"      Qdrant rebuild error: {e}")
        return False


def _sweep_rebuild_milvus(indexed, indexed_n, dim, idx_type):
    """Rebuild Milvus collection."""
    from pymilvus import MilvusClient
    from pymilvus.milvus_client.index import IndexParams

    try:
        client = MilvusClient(uri="http://127.0.0.1:19530")
        coll = "bench_milvus"

        try:
            client.drop_collection(coll)
        except:
            pass

        if indexed_n < 10000:
            nlist = 32
        else:
            nlist = 128

        build_params = {
            "ivf_flat": {"index_type": "IVF_FLAT", "nlist": nlist},
            "ivf_pq":   {"index_type": "IVF_PQ", "nlist": nlist, "m": 16, "nbits": 8},
            "ivf_sq8":  {"index_type": "IVF_SQ8", "nlist": nlist},
            "hnsw":     {"index_type": "HNSW", "M": 16, "efConstruction": 256},
        }
        bp = build_params.get(idx_type, build_params["ivf_flat"])

        ip = IndexParams()
        ip.add_index("vector", bp["index_type"],
                     nlist=bp.get("nlist", nlist),
                     m=bp.get("m", 16),
                     nbits=bp.get("nbits", 8),
                     efConstruction=bp.get("efConstruction", 256))
        client.create_collection(coll, dimension=dim, index_params=ip, auto_id=False)
        client.flush(coll)

        for batch_start in range(0, indexed_n, 500):
            batch_end = min(batch_start + 500, indexed_n)
            rows = [{"id": i, "vector": indexed[i * dim:(i + 1) * dim]}
                    for i in range(batch_start, batch_end)]
            client.insert(coll, rows)
        client.flush(coll)

        return True
    except Exception as e:
        print(f"      Milvus rebuild error: {e}")
        return False


def _interpolate_qps(sweep_points, target_recall):
    """Interpolate QPS at a target recall level from sweep points.
    Returns the maximum QPS achievable at or above the target recall.
    sweep_points format: [{param, qps, recall}, ...]
    """
    if not sweep_points:
        return 0.0

    # Filter points that achieve at least the target recall
    qualifying = [sp["qps"] for sp in sweep_points if sp["recall"] >= target_recall]

    if not qualifying:
        return 0.0  # No configuration achieves this recall

    # Return the maximum QPS among configurations that achieve target recall
    return max(qualifying)

    return points[-1][1]


def _sweep_search(db_name, indexed, indexed_n, dim, queries, query_n, topk, idx_type, param_val, param_name):
    """Execute search with a specific parameter value. Returns IDs."""
    if db_name == "Plasmod":
        return _sweep_plasmod(indexed, indexed_n, dim, queries, query_n, topk, idx_type, param_val, param_name)
    elif db_name == "Qdrant":
        return _sweep_qdrant(indexed, indexed_n, dim, queries, query_n, topk, idx_type, param_val, param_name)
    elif db_name == "Milvus":
        return _sweep_milvus(indexed, indexed_n, dim, queries, query_n, topk, idx_type, param_val, param_name)
    elif db_name == "LanceDB":
        return _sweep_lancedb(indexed, indexed_n, dim, queries, query_n, topk, idx_type, param_val, param_name)
    elif db_name == "ChromaDB":
        return _sweep_chromadb(indexed, indexed_n, dim, queries, query_n, topk, idx_type, param_val, param_name)
    return []


def _sweep_plasmod(indexed, indexed_n, dim, queries, query_n, topk, idx_type, param_val, param_name):
    """
    Plasmod sweep search.
    For HNSW: param_val is ef_construction (重建索引)
    For IVF: param_val is nprobe or nlist (重建索引时指定)
    """
    seg_id = "bench.sweep"
    server_url = "http://127.0.0.1:8080"

    # Unload existing segment
    try:
        data = json.dumps({"segment_id": seg_id}).encode()
        req = urllib.request.Request(
            f"{server_url}/v1/internal/rpc/unload_segment",
            data=data, headers={"Content-Type": "application/json"}, method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
        time.sleep(1)
    except Exception:
        pass

    # Build params based on index type
    if idx_type == "hnsw":
        ptype = "HNSW"
        nlist, nprobe, m, nbits, sq_type = 0, 0, 0, 0, ""
        ef_construction = param_val
    elif idx_type == "ivf_flat":
        # Scale nlist with param (more clusters = better recall)
        nlist = min(32 * (param_val // 4 + 1), 512) if param_val >= 4 else 32
        ptype = "IVF_FLAT"
        nprobe = param_val
        m, nbits, sq_type = 0, 0, ""
        ef_construction = 0
    elif idx_type == "ivf_pq":
        # For IVF-PQ: sweep nlist to achieve recall=1.0
        # m must divide dim (384). Valid m: 1,2,3,4,6,8,12,16,24,32,48,64,96,128,192,384
        ptype = "IVF_PQ"
        nlist = param_val
        nprobe = param_val  # Search all clusters
        # m increases with nlist: 16->32->48->64->96->128
        if param_val <= 32:
            m = 16
        elif param_val <= 64:
            m = 32
        elif param_val <= 128:
            m = 48
        elif param_val <= 256:
            m = 64
        elif param_val <= 512:
            m = 96
        else:
            m = 128
        nbits = 8
        sq_type = ""
        ef_construction = 0
    elif idx_type == "ivf_sq8":
        # Scale nlist with param
        nlist = min(32 * (param_val // 4 + 1), 512) if param_val >= 4 else 32
        ptype = "IVF_SQ8"
        nprobe = param_val
        m, nbits = 0, 0
        sq_type = "INT8"
        ef_construction = 0
    else:
        return []

    # Rebuild with different params
    http = _HTTPClient(server_url, timeout=120)
    try:
        for start in range(0, indexed_n, min(500_000, indexed_n)):
            end = min(start + 500_000, indexed_n)
            batch_n = end - start
            vec_slice = indexed[start * dim : end * dim]
            if not vec_slice:
                print(f"      Plasmod sweep error: empty vector slice at start={start}, end={end}")
                return []
            ok, _ = http.ingest(seg_id, list(vec_slice), batch_n, dim,
                                index_type=ptype, nlist=nlist, nprobe=nprobe,
                                m=m, nbits=nbits, sq_type=sq_type,
                                ef_construction=ef_construction)
            if not ok:
                print(f"      Plasmod sweep error: ingest failed")
                return []
        http.register_warm(seg_id, indexed_n)
    except Exception as e:
        print(f"      Plasmod sweep error: {e}")
        return []

    # Execute batch search
    try:
        ok, _, _, flat_ids, _ = http.query_batch(seg_id, queries, query_n, dim, topk)
        if not ok:
            print(f"      Plasmod sweep error: batch query failed")
            return []
        # Reshape flat ids to list of lists (one list per query)
        batch_ids = [flat_ids[i * topk:(i + 1) * topk] for i in range(query_n)]
        return batch_ids if batch_ids else []
    except Exception as e:
        print(f"      Plasmod sweep error: query_batch failed: {e}")
        return []


def _sweep_qdrant(indexed, indexed_n, dim, queries, query_n, topk, idx_type, param_val, param_name):
    """Qdrant sweep search with different nprobe/ef. Uses bench_test collection."""
    base = "http://127.0.0.1:6333"
    coll = "bench_test"

    try:
        if param_name == "ef":
            # HNSW search with ef - use existing collection
            payload = {"searches": [
                {"vector": queries[q * dim:(q + 1) * dim], "top": topk, "params": {"hnsw": {"ef": param_val}}}
                for q in range(query_n)
            ]}
            r = urllib.request.urlopen(
                urllib.request.Request(f"{base}/collections/{coll}/points/search/batch",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST"),
                timeout=120)
            d = json.loads(r.read())
            return [[pt["id"] - 1 for pt in res] for res in d.get("result", [])]
        else:
            # IVF search with nprobe - rebuild collection
            return _sweep_qdrant_ivf(indexed, indexed_n, dim, queries, query_n, topk, param_val)
    except Exception as e:
        print(f"      Qdrant sweep error: {e}")
        return []


def _sweep_qdrant_ivf(indexed, indexed_n, dim, queries, query_n, topk, nprobe):
    """Rebuild Qdrant collection with different nprobe for IVF sweep."""
    base = "http://127.0.0.1:6333"
    coll = "bench_test"

    try:
        # Delete and recreate collection
        try:
            urllib.request.urlopen(
                urllib.request.Request(f"{base}/collections/{coll}", method="DELETE",
                                     headers={"Content-Type": "application/json"}),
                timeout=10)
        except:
            pass

        req = {"vectors": {"size": dim, "distance": "Cosine"},
               "hnsw_config": {"m": 16, "ef_construct": 256}}
        urllib.request.urlopen(
            urllib.request.Request(f"{base}/collections/{coll}",
                                  data=json.dumps(req).encode(),
                                  headers={"Content-Type": "application/json"}, method="PUT"),
            timeout=30).read()

        # Ingest
        for batch_start in range(0, indexed_n, 500):
            batch_end = min(batch_start + 500, indexed_n)
            points = [{"id": i + 1, "vector": indexed[i * dim:(i + 1) * dim]}
                      for i in range(batch_start, batch_end)]
            body = json.dumps({"points": points}).encode()
            urllib.request.urlopen(
                urllib.request.Request(f"{base}/collections/{coll}/points", data=body,
                                      headers={"Content-Type": "application/json"}, method="PUT"),
                timeout=60)

        # Flush
        try:
            urllib.request.urlopen(f"{base}/collections/{coll}/flush", timeout=30)
        except:
            pass

        # Search with specific nprobe (Qdrant uses same HNSW config for IVF)
        payload = {"searches": [
            {"vector": queries[q * dim:(q + 1) * dim], "top": topk}
            for q in range(query_n)
        ]}
        r = urllib.request.urlopen(
            urllib.request.Request(f"{base}/collections/{coll}/points/search/batch",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST"),
            timeout=120)
        d = json.loads(r.read())
        return [[pt["id"] - 1 for pt in res] for res in d.get("result", [])]
    except Exception:
        return []


def _sweep_milvus(indexed, indexed_n, dim, queries, query_n, topk, idx_type, param_val, param_name):
    """Milvus sweep search. For IVF, rebuilds collection with different nprobe."""
    from pymilvus import MilvusClient
    from pymilvus.milvus_client.index import IndexParams

    client = MilvusClient(uri="http://127.0.0.1:19530")
    coll = "bench_milvus"

    try:
        qvecs = [queries[q * dim:(q + 1) * dim] for q in range(query_n)]

        if param_name == "ef":
            # HNSW: just search with different ef
            search_params = {"ef": param_val}
            res = client.search(coll, qvecs, limit=topk, search_params=search_params)
            return [[hit["id"] for hit in q] for q in res]
        else:
            # IVF: rebuild collection with different params (nprobe, nlist, m, nbits)
            if indexed_n < 10000:
                nlist = 32
            else:
                nlist = 128

            # Default IVF params
            m = 16 if idx_type == "ivf_pq" else 0
            nbits = 8 if idx_type in ("ivf_pq", "ivf_sq8") else 0

            return _sweep_milvus_ivf(client, indexed, indexed_n, dim, qvecs, topk,
                                     idx_type, param_val, nlist, m, nbits)
    except Exception as e:
        print(f"      Milvus sweep error: {e}")
        return []


def _sweep_milvus_ivf(client, indexed, indexed_n, dim, qvecs, topk, idx_type, nprobe, nlist, m, nbits):
    """Rebuild Milvus collection with correct index type and nprobe for IVF sweep."""
    from pymilvus.milvus_client.index import IndexParams

    coll = "bench_milvus_sweep"
    try:
        try:
            client.drop_collection(coll)
        except:
            pass

        # Build params based on idx_type
        index_type_map = {
            "ivf_flat": "IVF_FLAT",
            "ivf_pq": "IVF_PQ",
            "ivf_sq8": "IVF_SQ8",
        }
        index_type = index_type_map.get(idx_type, "IVF_FLAT")

        ip = IndexParams()
        ip.add_index("vector", index_type, nlist=nlist, nprobe=nprobe, m=m, nbits=nbits)
        client.create_collection(coll, dimension=dim, index_params=ip, auto_id=False)
        client.flush(coll)

        for batch_start in range(0, indexed_n, 500):
            batch_end = min(batch_start + 500, indexed_n)
            rows = [{"id": i, "vector": indexed[i * dim:(i + 1) * dim]}
                    for i in range(batch_start, batch_end)]
            client.insert(coll, rows)
        client.flush(coll)
        time.sleep(1)  # Wait for index build

        res = client.search(coll, qvecs, limit=topk, search_params={"nprobe": nprobe})
        return [[hit["id"] for hit in q] for q in res]
    except Exception as e:
        print(f"      Milvus IVF rebuild error: {e}")
        return []


def _sweep_lancedb(indexed, indexed_n, dim, queries, query_n, topk, idx_type, param_val, param_name):
    """
    LanceDB sweep search.
    All index types: rebuild collection with different ef_construction.
    """
    import lancedb
    import pyarrow as pa

    db = lancedb.connect(str(BASE / "lancedb_data"))
    table_name = "bench_lancedb_sweep"

    try:
        try:
            db.drop_table(table_name)
        except Exception:
            pass

        ids = list(range(indexed_n))
        vectors = [[indexed[i * dim + d] for d in range(dim)] for i in range(indexed_n)]

        schema = pa.schema([
            ("id", pa.int64()),
            ("vector", pa.list_(pa.float32(), dim)),
        ])

        # Build table
        for bs in range(0, indexed_n, 500):
            be = min(bs + 500, indexed_n)
            tbl = pa.table({"id": ids[bs:be], "vector": vectors[bs:be]}, schema=schema)
            if bs == 0:
                db.create_table(table_name, data=tbl)
            else:
                db.open_table(table_name).add(tbl)

        # Create index based on idx_type
        tbl_ref = db.open_table(table_name)
        if idx_type == "hnsw":
            # For HNSW sweep, we can vary the index parameters
            # num_partitions controls search granularity
            # Higher partitions = better recall = slower search
            num_parts = max(1, param_val // 32)
            tbl_ref.create_index(metric="cosine", vector_column_name="vector",
                                 index_type="IVF_HNSW_PQ",
                                 num_partitions=num_parts, num_sub_vectors=min(96, dim),
                                 replace=True)
        elif idx_type == "ivf_flat":
            num_parts = max(1, param_val // 32)
            tbl_ref.create_index(metric="cosine", vector_column_name="vector",
                                 index_type="IVF_FLAT", num_partitions=num_parts,
                                 replace=True)
        elif idx_type == "ivf_pq":
            num_parts = max(1, param_val // 32)
            tbl_ref.create_index(metric="cosine", vector_column_name="vector",
                                 index_type="IVF_PQ", num_partitions=num_parts, num_sub_vectors=8,
                                 replace=True)
        elif idx_type == "ivf_sq8":
            num_parts = max(1, param_val // 32)
            tbl_ref.create_index(metric="cosine", vector_column_name="vector",
                                 index_type="IVF_SQ", num_partitions=num_parts,
                                 replace=True)
        else:
            # flat/no index - just return
            pass

        try:
            tbl_ref.wait_for_index()
        except Exception:
            pass

        # Search
        ids = []
        for q in range(query_n):
            qvec = [queries[q * dim + d] for d in range(dim)]
            result_table = tbl_ref.search(qvec).limit(topk).to_arrow()
            ids.append([row["id"] for row in result_table.to_pylist()])
        return ids
    except Exception as e:
        print(f"      LanceDB sweep error: {e}")
        return []


def _sweep_chromadb(indexed, indexed_n, dim, queries, query_n, topk, idx_type, param_val, param_name):
    """ChromaDB sweep search. Only HNSW is supported."""
    if idx_type != "hnsw":
        return []

    import chromadb

    chroma_dir = str(BASE / "chromadb_data")
    client = chromadb.PersistentClient(path=chroma_dir)
    coll_name = "bench_chroma_sweep"

    try:
        try:
            client.delete_collection(coll_name)
        except Exception:
            pass

        # Wait for deletion to complete
        time.sleep(2.0)

        ids = [f"sweep_{i}" for i in range(indexed_n)]
        # Convert numpy floats to plain Python floats for ChromaDB compatibility
        vectors = [[float(indexed[i * dim + d]) for d in range(dim)] for i in range(indexed_n)]

        # ChromaDB only supports hnsw:space metadata (not efConstruction)
        metadata = {"hnsw:space": "cosine"}
        coll = client.create_collection(name=coll_name, metadata=metadata)

        for bs in range(0, indexed_n, 500):
            be = min(bs + 500, indexed_n)
            coll.add(ids=ids[bs:be], embeddings=vectors[bs:be])

        qvecs = [[float(queries[q * dim + d]) for d in range(dim)] for q in range(query_n)]
        res = coll.query(query_embeddings=qvecs, n_results=topk)
        ids_list = res.get("ids", [[] for _ in range(query_n)])
        return [[int(id_.split("_")[1]) for id_ in ids] for ids in ids_list]
    except Exception as e:
        print(f"      ChromaDB sweep error: {e}")
        return []


def serial_ms_for_param(db_name, indexed_n, dim, queries, query_n, topk, idx_type, param_val, param_name):
    """Get serial search time for a parameter value."""
    if db_name == "Qdrant":
        base = "http://127.0.0.1:6333"
        coll = "bench_test"
        latencies = []
        for q in range(query_n):
            try:
                body = json.dumps({"vector": queries[q * dim:(q + 1) * dim], "top": topk}).encode()
                t0 = time.time()
                urllib.request.urlopen(
                    urllib.request.Request(f"{base}/collections/{coll}/points/search",
                                          data=body,
                                          headers={"Content-Type": "application/json"}, method="POST"),
                    timeout=30).read()
                latencies.append((time.time() - t0) * 1000)
            except Exception:
                pass
        return sum(latencies) if latencies else 0

    elif db_name == "Milvus":
        from pymilvus import MilvusClient
        client = MilvusClient(uri="http://127.0.0.1:19530")
        coll = "bench_milvus"
        latencies = []
        qvecs = [queries[q * dim:(q + 1) * dim] for q in range(query_n)]

        if param_name == "ef":
            search_params = {"ef": param_val}
        else:
            search_params = {"nprobe": param_val}

        for qvec in qvecs:
            t0 = time.time()
            try:
                client.search(coll, [qvec], limit=topk, search_params=search_params)
                latencies.append((time.time() - t0) * 1000)
            except Exception:
                pass
        return sum(latencies) if latencies else 0

    elif db_name == "Plasmod":
        # For Plasmod sweep, we need to rebuild with different ef and then measure
        # This is handled by rebuild, so just return estimated serial time
        seg_id = "bench.sweep"
        server_url = "http://127.0.0.1:8080"
        http = _HTTPClient(server_url, timeout=30)
        latencies = []
        for q in range(query_n):
            query = [queries[q * dim + d] for d in range(dim)]
            ok, code, lat = http.query_serial(seg_id, query, dim, topk)
            if ok:
                latencies.append(lat)
        return sum(latencies) if latencies else 0

    elif db_name == "ChromaDB":
        import chromadb
        client = chromadb.PersistentClient(path=str(BASE / "chromadb_data"))
        coll_name = "bench_chroma_sweep"
        latencies = []
        try:
            c = client.get_collection(coll_name)
            qvecs = [[queries[q * dim + d] for d in range(dim)] for q in range(query_n)]
            for qvec in qvecs:
                t0 = time.time()
                c.query(query_embeddings=[qvec], n_results=topk)
                latencies.append((time.time() - t0) * 1000)
        except Exception:
            pass
        return sum(latencies) if latencies else 0

    elif db_name == "LanceDB":
        import lancedb
        db = lancedb.connect(str(BASE / "lancedb_data"))
        table_name = "bench_lancedb_sweep"
        latencies = []
        try:
            tbl_ref = db.open_table(table_name)
            for q in range(query_n):
                qvec = [queries[q * dim + d] for d in range(dim)]
                t0 = time.time()
                tbl_ref.search(qvec).limit(topk).to_arrow()
                latencies.append((time.time() - t0) * 1000)
        except Exception:
            pass
        return sum(latencies) if latencies else 0

    return 0  # Default for unsupported DBs


# ─── Qdrant ───────────────────────────────────────────────────────────────────

def benchmark_qdrant(indexed, indexed_n, dim, queries, query_n, topk, idx_type):
    base = "http://127.0.0.1:6333"
    coll = "bench_test"

    # Cleanup
    try:
        urllib.request.urlopen(
            urllib.request.Request(f"{base}/collections/{coll}", method="DELETE",
                                 headers={"Content-Type": "application/json"}),
            timeout=10)
    except Exception:
        pass

    # Qdrant uses HNSW as the underlying index. For quantization benchmarks,
    # we enable quantization (scalar or product) and force its use at search time.
    if idx_type == "hnsw":
        req = {
            "vectors": {"size": dim, "distance": "Cosine"},
            "hnsw_config": {"m": 16, "ef_construct": 256, "full_scan_threshold": 10000},
        }
    elif idx_type == "ivf_flat":
        # Pure HNSW with full scan disabled
        req = {
            "vectors": {"size": dim, "distance": "Cosine"},
            "hnsw_config": {"m": 16, "ef_construct": 256, "full_scan_threshold": 100000000},
        }
    elif idx_type == "ivf_pq":
        # Product Quantization via scalar quantization with product mode
        req = {
            "vectors": {"size": dim, "distance": "Cosine"},
            "hnsw_config": {"m": 16, "ef_construct": 256, "full_scan_threshold": 100000000},
            "quantization_config": {
                "scalar": {"type": "int8", "quantization": "product"},
            },
        }
    elif idx_type == "ivf_sq8":
        # Scalar Quantization 8-bit
        req = {
            "vectors": {"size": dim, "distance": "Cosine"},
            "hnsw_config": {"m": 16, "ef_construct": 256, "full_scan_threshold": 100000000},
            "quantization_config": {
                "scalar": {"type": "int8"},
            },
        }

    urllib.request.urlopen(
        urllib.request.Request(f"{base}/collections/{coll}",
                              data=json.dumps(req).encode(),
                              headers={"Content-Type": "application/json"}, method="PUT"),
        timeout=30).read()

    # Ingest
    t0 = time.time()
    for batch_start in range(0, indexed_n, 500):
        batch_end = min(batch_start + 500, indexed_n)
        points = [{"id": i + 1,
                   "vector": indexed[i * dim:(i + 1) * dim]}
                  for i in range(batch_start, batch_end)]
        body = json.dumps({"points": points}).encode()
        urllib.request.urlopen(
            urllib.request.Request(f"{base}/collections/{coll}/points", data=body,
                                  headers={"Content-Type": "application/json"}, method="PUT"),
            timeout=60)

    # Flush / wait index
    try:
        urllib.request.urlopen(f"{base}/collections/{coll}/flush", timeout=30)
    except Exception:
        pass
    build_ms = (time.time() - t0) * 1000

    # Build search params - for quantized indexes, force using quantized vectors
    search_params = {}
    if idx_type in ("ivf_pq", "ivf_sq8"):
        # Force search to use quantized vectors (faster but lossy)
        search_params = {"hnsw": {"ef": 256, "exact": False}}

    # Batch search (single HTTP call for all queries)
    batch_payload = {
        "searches": [
            {
                "vector": queries[q * dim:(q + 1) * dim],
                "top": topk,
                "with_vectors": False,
                "params": search_params,
            }
            for q in range(query_n)
        ]
    }
    # Batch search: single RPC with all queries at once.
    # Qdrant has a dedicated /search/batch endpoint.
    qvecs = [queries[q * dim:(q + 1) * dim] for q in range(query_n)]
    t0 = time.time()
    r = urllib.request.urlopen(
        urllib.request.Request(f"{base}/collections/{coll}/points/search/batch",
                             data=json.dumps(batch_payload).encode(),
                             headers={"Content-Type": "application/json"}, method="POST"),
        timeout=120)
    batch_data = json.loads(r.read())
    batch_ms = (time.time() - t0) * 1000
    batch_qps = query_n / (batch_ms / 1000) if batch_ms > 0 else 0

    # Collect result IDs for recall computation
    batch_ids = [[pt["id"] - 1 for pt in res] for res in batch_data.get("result", [])]

    # Compute recall
    gt = brute_force_search(indexed, indexed_n, dim, queries, query_n, topk)
    batch_recall = recall_at_k(batch_ids, gt, topk) if batch_ids else 0.0

    # Serial search
    latencies = []
    for q in range(query_n):
        body = json.dumps({
            "vector": queries[q * dim:(q + 1) * dim],
            "top": topk,
            "params": search_params,
        }).encode()
        t = time.time()
        try:
            urllib.request.urlopen(
                urllib.request.Request(f"{base}/collections/{coll}/points/search",
                                      data=body,
                                      headers={"Content-Type": "application/json"}, method="POST"),
                timeout=30).read()
        except Exception:
            pass
        latencies.append((time.time() - t) * 1000)
    serial_ms = sum(latencies)
    serial_qps = query_n / (serial_ms / 1000)
    sl = sorted(latencies)
    p50 = sl[int(len(sl) * 0.50)]
    p95 = sl[int(len(sl) * 0.95)]
    p99 = sl[int(len(sl) * 0.99)]

    # Memory of qdrant process (RSS in MB)
    mem_mb = 0.0
    try:
        for line in subprocess.check_output(["pgrep", "-f", "qdrant/bin/qdrant"],
                                           text=True, timeout=5).splitlines():
            pid = int(line.strip())
            mem_mb = mem_mb(pid)  # mem_mb returns MB
            break
    except Exception:
        pass
    if mem_mb == 0:
        # fallback: use parent process RSS in KB -> MB
        try:
            out = subprocess.check_output(
                ["ps", "-p", str(os.getpid()), "-o", "rss="], text=True, timeout=5)
            mem_mb = float(out.strip()) / 1024  # KB -> MB
        except Exception:
            pass

    return Result(
        db="Qdrant", index_type=idx_type.upper().replace("_", "-"),
        n_indexed=indexed_n, n_queries=query_n, dim=dim, topk=topk,
        build_ms=build_ms, batch_ms=batch_ms, batch_qps=batch_qps,
        serial_ms=serial_ms, serial_qps=serial_qps,
        p50_ms=p50, p95_ms=p95, p99_ms=p99,
        recall=batch_recall, memory_mb=mem_mb,
    )


# ─── Milvus ───────────────────────────────────────────────────────────────────

def cleanup_all():
    """Cleanup all DB collections after recall computation."""
    # Qdrant
    for coll in ("bench_test", "bench_hnsw"):
        try:
            urllib.request.urlopen(
                urllib.request.Request(f"http://127.0.0.1:6333/collections/{coll}",
                                     method="DELETE",
                                     headers={"Content-Type": "application/json"}),
                timeout=10)
        except Exception:
            pass
    # Milvus
    try:
        from pymilvus import MilvusClient
        c = MilvusClient(uri="http://127.0.0.1:19530")
        for coll in ("bench_test", "bench_milvus"):
            try:
                c.drop_collection(coll)
            except Exception:
                pass
    except Exception:
        pass
    # LanceDB
    try:
        import lancedb
        db = lancedb.connect(str(BASE / "lancedb_data"))
        for tbl in ("bench_test", "bench_lancedb"):
            try:
                db.drop_table(tbl)
            except Exception:
                pass
    except Exception:
        pass
    # ChromaDB
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(BASE / "chromadb_data"))
        for coll in ("bench_test", "bench_chroma"):
            try:
                client.delete_collection(coll)
            except Exception:
                pass
    except Exception:
        pass
    # Plasmod (segment)
    try:
        urllib.request.urlopen(
            urllib.request.Request("http://127.0.0.1:8080/v1/internal/rpc/unload_segment",
                                 data=json.dumps({"segment_id": "bench.layer1"}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST"),
            timeout=10)
    except Exception:
        pass


def gather_db_results(db, indexed_n, dim, queries, query_n, topk, idx_type=None):
    """Gather search results from each DB for recall computation."""
    import json  # Local import to avoid any shadowing issues
    if db == "Qdrant":
        base = "http://127.0.0.1:6333"
        coll = "bench_test"
        try:
            # Build search params for quantized indexes
            search_params = {}
            if idx_type in ("ivf_pq", "ivf_sq8"):
                search_params = {"hnsw": {"ef": 256, "exact": False}}

            payload = {"searches": [
                {
                    "vector": queries[q * dim:(q + 1) * dim],
                    "top": topk,
                    "params": search_params,
                }
                for q in range(query_n)
            ]}
            r = urllib.request.urlopen(
                urllib.request.Request(f"{base}/collections/{coll}/points/search/batch",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST"),
                timeout=120)
            d = json.loads(r.read())
            return [[pt["id"] - 1 for pt in res] for res in d.get("result", [])]
        except Exception as e:
            print(f"      [gather] Qdrant recall failed: {e}")
            return []
    elif db == "Milvus":
        from pymilvus import MilvusClient
        client = MilvusClient(uri="http://127.0.0.1:19530")
        # Use bench_milvus for HNSW (sweep searches this directly)
        # Use bench_milvus_sweep for IVF (sweep creates this with correct index type)
        if idx_type == "hnsw":
            coll = "bench_milvus"
        else:
            coll = "bench_milvus_sweep"
        try:
            qvecs = [queries[q * dim:(q + 1) * dim] for q in range(query_n)]
            # Use same search params as during benchmark to get comparable recall
            if idx_type == "hnsw":
                search_params = {"ef": 256}
            elif idx_type in ("ivf_flat", "ivf_pq", "ivf_sq8"):
                # For small datasets: nprobe_search=4 was used during benchmark
                search_params = {"nprobe": 4} if indexed_n < 10000 else {"nprobe": 16}
            else:
                search_params = {}
            res = client.search(coll, qvecs, limit=topk, search_params=search_params)
            return [[hit["id"] for hit in q] for q in res]
        except Exception as e:
            print(f"      [gather] Milvus recall failed: {e}")
            return []
    # Recall: LanceDB uses bench_lancedb_sweep after sweep runs
    elif db == "LanceDB":
        import lancedb
        db_lance = lancedb.connect(str(BASE / "lancedb_data"))
        # Use bench_lancedb_sweep for recall (sweep creates this with correct params)
        coll = "bench_lancedb_sweep"
        try:
            tbl_ref = db_lance.open_table(coll)
            ids = []
            for q in range(query_n):
                qvec = [queries[q * dim + d] for d in range(dim)]
                result_table = tbl_ref.search(qvec).limit(topk).to_arrow()
                ids.append([row["id"] for row in result_table.to_pylist()])
            return ids
        except Exception as e:
            print(f"      LanceDB recall gather failed: {e}")
            return []
    elif db == "ChromaDB":
        import chromadb
        import json
        chroma_dir = str(BASE / "chromadb_data")
        client = chromadb.PersistentClient(path=chroma_dir)
        # Use bench_chroma_sweep for recall (sweep creates this)
        coll_name = "bench_chroma_sweep"
        try:
            c = client.get_collection(coll_name)
            qvecs = [[float(queries[q * dim + d]) for d in range(dim)] for q in range(query_n)]
            res = c.query(query_embeddings=qvecs, n_results=topk)
            ids_list = res.get("ids", [[] for _ in range(query_n)])
            return [[int(id_.split("_")[1]) for id_ in ids] for ids in ids_list]
        except Exception as e:
            # Fallback to file if sweep collection doesn't exist
            recall_file = BASE / "chromadb_batch_ids.json"
            try:
                with open(recall_file, "r") as f:
                    return json.load(f)
            except Exception:
                print(f"      [gather] ChromaDB recall failed: {e}")
                return []
    else:
        print(f"    [gather] Unknown db: {db}")
        return []


def benchmark_milvus(indexed, indexed_n, dim, queries, query_n, topk, idx_type):
    from pymilvus import MilvusClient
    from pymilvus.milvus_client.index import IndexParams

    client = MilvusClient(uri="http://127.0.0.1:19530")
    coll = "bench_milvus"

    try:
        client.drop_collection(coll)
    except Exception:
        pass

    # For small datasets (3633 vectors), use fewer centroids and lower nprobe
    # to avoid brute-force-like behavior
    if indexed_n < 10000:
        nlist = 32  # ~113 vectors per centroid
        nprobe_search = 4  # Scan ~452 vectors (12% of data)
    else:
        nlist = 128
        nprobe_search = 16  # Scan ~452 vectors (4% of data)

    # Build params for each index type
    build_params = {
        "ivf_flat": {"index_type": "IVF_FLAT", "nlist": nlist, "nprobe": nprobe_search},
        "ivf_pq":   {"index_type": "IVF_PQ", "nlist": nlist, "nprobe": nprobe_search, "m": 16, "nbits": 8},
        "ivf_sq8":  {"index_type": "IVF_SQ8", "nlist": nlist, "nprobe": nprobe_search},
        "hnsw":     {"index_type": "HNSW", "M": 16, "efConstruction": 256},
    }
    bp = build_params.get(idx_type, build_params["ivf_flat"])

    # Create index (build-time nprobe is different from search-time)
    ip = IndexParams()
    ip.add_index("vector", bp["index_type"],
                 nlist=bp.get("nlist", nlist),
                 m=bp.get("m", 16),
                 nbits=bp.get("nbits", 8),
                 efConstruction=bp.get("efConstruction", 256))
    client.create_collection(coll, dimension=dim, index_params=ip, auto_id=False)
    client.flush(coll)

    # Insert
    t0 = time.time()
    for batch_start in range(0, indexed_n, 500):
        batch_end = min(batch_start + 500, indexed_n)
        rows = [{"id": i, "vector": indexed[i * dim:(i + 1) * dim]}
                for i in range(batch_start, batch_end)]
        client.insert(coll, rows)
    client.flush(coll)
    build_ms = (time.time() - t0) * 1000

    # Search params: HNSW uses ef, IVF uses nprobe
    if idx_type == "hnsw":
        search_params = {"ef": 256}
    else:
        # Use nprobe_search which was defined earlier based on dataset size
        search_params = {"nprobe": nprobe_search}

    # Batch search
    qvecs = [queries[q * dim:(q + 1) * dim] for q in range(query_n)]
    t0 = time.time()
    batch_res = client.search(coll, qvecs, limit=topk, search_params=search_params)
    batch_ms = (time.time() - t0) * 1000
    batch_qps = query_n / (batch_ms / 1000) if batch_ms > 0 else 0

    # Collect result IDs for recall computation
    batch_ids = [[hit["id"] for hit in q] for q in batch_res]

    # Compute recall
    gt = brute_force_search(indexed, indexed_n, dim, queries, query_n, topk)
    batch_recall = recall_at_k(batch_ids, gt, topk) if batch_ids else 0.0

    # Serial search
    latencies = []
    for q in range(query_n):
        t = time.time()
        client.search(coll, [qvecs[q]], limit=topk, search_params=search_params)
        latencies.append((time.time() - t) * 1000)
    serial_ms = sum(latencies)
    serial_qps = query_n / (serial_ms / 1000) if serial_ms > 0 else 0
    sl = sorted(latencies)
    p50 = sl[int(len(sl) * 0.50)]
    p95 = sl[int(len(sl) * 0.95)]
    p99 = sl[int(len(sl) * 0.99)]

    # Memory of milvus-standalone container (RSS in MB, sum of all processes)
    mem_mb = 0.0
    try:
        cid = subprocess.check_output(
            ["docker", "ps", "-q", "--filter", "name=milvus-standalone"], text=True, timeout=5
        ).strip().split("\n")[0]  # take first container ID only
        if cid:
            # Sum RSS of all processes in the container
            out3 = subprocess.check_output(
                ["docker", "exec", cid, "ps", "-eo", "rss="], text=True, timeout=5
            )
            mem_kb = sum(float(line.strip()) for line in out3.splitlines() if line.strip().isdigit())
            mem_mb = mem_kb / 1024  # KB -> MB
    except Exception:
        pass

    return Result(
        db="Milvus", index_type=idx_type.upper().replace("_", "-"),
        n_indexed=indexed_n, n_queries=query_n, dim=dim, topk=topk,
        build_ms=build_ms, batch_ms=batch_ms, batch_qps=batch_qps,
        serial_ms=serial_ms, serial_qps=serial_qps,
        p50_ms=p50, p95_ms=p95, p99_ms=p99,
        recall=batch_recall, memory_mb=mem_mb,
    )


# ─── LanceDB ───────────────────────────────────────────────────────────────────

def benchmark_lancedb(indexed, indexed_n, dim, queries, query_n, topk, idx_type):
    import lancedb, pyarrow as pa

    db = lancedb.connect(str(BASE / "lancedb_data"))
    table = "bench_lancedb"

    try:
        db.drop_table(table)
    except Exception:
        pass

    ids = list(range(indexed_n))
    vectors = [[indexed[i * dim + d] for d in range(dim)] for i in range(indexed_n)]

    schema = pa.schema([
        ("id", pa.int64()),
        ("vector", pa.list_(pa.float32(), dim)),
    ])

    t0 = time.time()
    for bs in range(0, indexed_n, 500):
        be = min(bs + 500, indexed_n)
        tbl = pa.table({"id": ids[bs:be], "vector": vectors[bs:be]}, schema=schema)
        if bs == 0:
            db.create_table(table, data=tbl)
        else:
            db.open_table(table).add(tbl)

    tbl_ref = db.open_table(table)

    # LanceDB VectorIndexType: IVF_FLAT, IVF_SQ, IVF_PQ, IVF_HNSW_SQ, IVF_HNSW_PQ, IVF_RQ
    if idx_type == "hnsw":
        # Use IVF_HNSW_PQ as LanceDB's HNSW-equivalent (no pure HNSW)
        tbl_ref.create_index(metric="cosine", vector_column_name="vector",
                             index_type="IVF_HNSW_PQ",
                             num_partitions=1, num_sub_vectors=min(96, dim),
                             replace=True)
        build_ms = (time.time() - t0) * 1000
        try:
            tbl_ref.wait_for_index()
        except Exception:
            pass
    elif idx_type == "ivf_pq":
        # IVF_PQ: for dim=384, use num_sub_vectors=48 (each sub-vector = 8 dims)
        # More sub-vectors = less compression = higher recall
        tbl_ref.create_index(metric="cosine", vector_column_name="vector",
                             index_type="IVF_PQ", num_partitions=1, num_sub_vectors=48,
                             replace=True)
        build_ms = (time.time() - t0) * 1000
        try:
            tbl_ref.wait_for_index()
        except Exception:
            pass
    elif idx_type == "ivf_flat":
        tbl_ref.create_index(metric="cosine", vector_column_name="vector",
                             index_type="IVF_FLAT", num_partitions=1,
                             replace=True)
        build_ms = (time.time() - t0) * 1000
        try:
            tbl_ref.wait_for_index()
        except Exception:
            pass
    elif idx_type == "ivf_sq8":
        tbl_ref.create_index(metric="cosine", vector_column_name="vector",
                             index_type="IVF_SQ", num_partitions=1,
                             replace=True)
        build_ms = (time.time() - t0) * 1000
        try:
            tbl_ref.wait_for_index()
        except Exception:
            pass
    else:
        # FLAT: no index — just scanning
        build_ms = (time.time() - t0) * 1000

    # Batch search (single RPC with all queries).
    # LanceDB has no native batch search endpoint; "batch" is the sum of serial times.
    # We still measure total wall-clock to give an upper bound.
    qvecs = [[queries[q * dim + d] for d in range(dim)] for q in range(query_n)]
    t0 = time.time()
    batch_ids = []
    for q in qvecs:
        result_table = tbl_ref.search(q).limit(topk).to_arrow()
        batch_ids.append([row["id"] for row in result_table.to_pylist()])
    batch_ms = (time.time() - t0) * 1000
    batch_qps = query_n / (batch_ms / 1000) if batch_ms > 0 else 0

    # Compute recall
    gt = brute_force_search(indexed, indexed_n, dim, queries, query_n, topk)
    batch_recall = recall_at_k(batch_ids, gt, topk) if batch_ids else 0.0

    # Serial search
    latencies = []
    for q in qvecs:
        t = time.time()
        tbl_ref.search(q).limit(topk).to_arrow()
        latencies.append((time.time() - t) * 1000)
    serial_ms = sum(latencies)
    serial_qps = query_n / (serial_ms / 1000) if serial_ms > 0 else 0
    sl = sorted(latencies)
    p50 = sl[int(len(sl) * 0.50)] if sl else 0
    p95 = sl[int(len(sl) * 0.95)] if sl else 0
    p99 = sl[int(len(sl) * 0.99)] if sl else 0

    # Memory: LanceDB is embedded in Python process — report Python RSS in MB
    try:
        out = subprocess.check_output(
            ["ps", "-p", str(os.getpid()), "-o", "rss="], text=True, timeout=5)
        mem_mb = float(out.strip()) / 1024  # KB -> MB
    except Exception:
        mem_mb = 0.0

    return Result(
        db="LanceDB", index_type=idx_type.upper().replace("_", "-"),
        n_indexed=indexed_n, n_queries=query_n, dim=dim, topk=topk,
        build_ms=build_ms, batch_ms=batch_ms, batch_qps=batch_qps,
        serial_ms=serial_ms, serial_qps=serial_qps,
        p50_ms=p50, p95_ms=p95, p99_ms=p99,
        recall=batch_recall, memory_mb=mem_mb,
    )


# ─── ChromaDB ─────────────────────────────────────────────────────────────────

def benchmark_chromadb(indexed, indexed_n, dim, queries, query_n, topk, idx_type):
    import chromadb
    import json

    chroma_dir = str(BASE / "chromadb_data")
    coll_name = "bench_chroma"

    client = chromadb.PersistentClient(path=chroma_dir)
    try:
        client.delete_collection(coll_name)
    except Exception:
        pass

    ids = [f"vec_{i}" for i in range(indexed_n)]
    # Convert numpy floats to plain Python floats for ChromaDB compatibility
    vectors = [[float(indexed[i * dim + d]) for d in range(dim)] for i in range(indexed_n)]
    coll = client.create_collection(
        name=coll_name,
        metadata={"hnsw:space": "cosine"}
    )

    t0 = time.time()
    for bs in range(0, indexed_n, 500):
        be = min(bs + 500, indexed_n)
        coll.add(ids=ids[bs:be], embeddings=vectors[bs:be])

    build_ms = (time.time() - t0) * 1000

    # Batch search (single call with all queries)
    qvecs = [[float(queries[q * dim + d]) for d in range(dim)] for q in range(query_n)]
    t0 = time.time()
    res = coll.query(query_embeddings=qvecs, n_results=topk)
    batch_ms = (time.time() - t0) * 1000
    batch_qps = query_n / (batch_ms / 1000) if batch_ms > 0 else 0

    # Parse IDs from batch query result and save for recall
    batch_ids = [[int(id_.split("_")[1]) for id_ in ids] for ids in res.get("ids", [[] for _ in range(query_n)])]
    recall_file = BASE / "chromadb_batch_ids.json"
    with open(recall_file, "w") as f:
        json.dump(batch_ids, f)

    # Compute recall
    gt = brute_force_search(indexed, indexed_n, dim, queries, query_n, topk)
    batch_recall = recall_at_k(batch_ids, gt, topk) if batch_ids else 0.0

    # Serial search
    latencies = []
    for qvec in qvecs:
        t = time.time()
        coll.query(query_embeddings=[qvec], n_results=topk)
        latencies.append((time.time() - t) * 1000)
    serial_ms = sum(latencies)
    serial_qps = query_n / (serial_ms / 1000) if serial_ms > 0 else 0
    sl = sorted(latencies)
    p50 = sl[int(len(sl) * 0.50)] if sl else 0
    p95 = sl[int(len(sl) * 0.95)] if sl else 0
    p99 = sl[int(len(sl) * 0.99)] if sl else 0

    # Memory: ChromaDB is embedded in Python process
    try:
        out = subprocess.check_output(
            ["ps", "-p", str(os.getpid()), "-o", "rss="], text=True, timeout=5)
        mem_mb = float(out.strip()) / 1024
    except Exception:
        mem_mb = 0.0

    return Result(
        db="ChromaDB", index_type=idx_type.upper().replace("_", "-"),
        n_indexed=indexed_n, n_queries=query_n, dim=dim, topk=topk,
        build_ms=build_ms, batch_ms=batch_ms, batch_qps=batch_qps,
        serial_ms=serial_ms, serial_qps=serial_qps,
        p50_ms=p50, p95_ms=p95, p99_ms=p99,
        recall=batch_recall, memory_mb=mem_mb,
    )


# ─── Plasmod ───────────────────────────────────────────────────────────────────

# Binary protocol constants (must match src/internal/transport/framing.go)
_MAGIC_INGEST  = b"PLIB"
_MAGIC_QUERY_W  = b"PLQW"
_MAGIC_QUERY_B  = b"PLQB"
_WIRE_VERSION   = 1

# ── Low-level binary protocol helpers ──────────────────────────────────────────

def _pack_u16(v: int) -> bytes: return struct.pack("<H", v)
def _pack_u32(v: int) -> bytes: return struct.pack("<I", v)
def _unpack_u32(b: bytes) -> int: return struct.unpack("<I", b)[0]
def _unpack_u64(b: bytes) -> int: return struct.unpack("<Q", b)[0]

_WIRE_VERSION   = 3  # wire v3: index_type + IVF params

def _build_ingest_payload(seg_id: str, vectors: list[float], n: int, dim: int,
                          index_type: str = "",
                          nlist: int = 0, nprobe: int = 0,
                          m: int = 0, nbits: int = 0,
                          sq_type: str = "",
                          ef_construction: int = 0) -> bytes:
    """Build PLIB v3 ingest payload: header + vectors + object_ids + index_type + IVF params."""
    # Wire v3: magic(4) + ver(1) + seg + n + dim + vectors + ids + [v3 fields]
    buf = _MAGIC_INGEST + struct.pack("B", _WIRE_VERSION)
    buf += _pack_u16(len(seg_id)) + seg_id.encode()
    buf += _pack_u32(n) + _pack_u32(dim)
    # Pack all vectors in one struct.pack call (much faster than nested loops)
    buf += struct.pack(f"<{n * dim}f", *vectors[:n * dim])
    # Object IDs
    for i in range(n):
        oid = f"bench-p{i:06d}"
        buf += _pack_u16(len(oid)) + oid.encode()
    # v3 fields: index_type + IVF params
    idx_bytes = index_type.encode()
    buf += _pack_u32(len(idx_bytes)) + idx_bytes
    buf += struct.pack("<i", nlist)
    buf += struct.pack("<i", nprobe)
    buf += struct.pack("<i", m)
    buf += struct.pack("<i", nbits)
    sq_bytes = sq_type.encode()
    buf += _pack_u32(len(sq_bytes)) + sq_bytes
    # ef_construction for HNSW sweep
    buf += struct.pack("<i", ef_construction)
    return buf

def _build_serial_query_payload(seg_id: str, query: list[float], dim: int, topk: int) -> bytes:
    """Build PLQW single-query payload (nq=1)."""
    buf = _MAGIC_QUERY_W + struct.pack("B", _WIRE_VERSION)
    buf += _pack_u16(len(seg_id)) + seg_id.encode()
    buf += _pack_u32(topk) + _pack_u32(dim)
    for j in range(dim):
        buf += struct.pack("<f", query[j])
    return buf

def _build_batch_query_payload(seg_id: str, queries: list[float], nq: int,
                                dim: int, topk: int) -> bytes:
    """Build PLQB batch-query payload."""
    buf = _MAGIC_QUERY_B + struct.pack("B", _WIRE_VERSION)
    buf += _pack_u16(len(seg_id)) + seg_id.encode()
    buf += _pack_u32(topk) + _pack_u32(nq) + _pack_u32(dim)
    # Pack all queries in one struct.pack call (much faster than nested loops)
    buf += struct.pack(f"<{nq * dim}f", *queries[:nq * dim])
    return buf

def _parse_batch_response(body: bytes, nq: int, topk: int) -> tuple[list[int], list[float]]:
    """Parse PLQB response: [nq][topk][id array][dist array] → (ids, dists).

    The server encodes ids and dists as separate contiguous arrays:
      [nq(u32)][topk(u32)][nq*topk * int64][nq*topk * float32]
    """
    if len(body) < 8:
        return [], []
    resp_nq   = _unpack_u32(body[0:4])
    resp_topk = _unpack_u32(body[4:8])

    n_results = min(resp_nq, nq) * resp_topk

    # Read ids: nq*topk * int64 starting at byte 8
    ids = []
    for i in range(n_results):
        off = 8 + i * 8
        ids.append(_unpack_u64(body[off:off+8]))

    # Read dists: nq*topk * float32 starting at byte 8 + nq*topk*8
    id_array_size = n_results * 8
    dists = []
    for i in range(n_results):
        off = 8 + id_array_size + i * 4
        dists.append(struct.unpack("<f", body[off:off+4])[0])

    return ids, dists


# ── HTTP session helpers ───────────────────────────────────────────────────────

class _HTTPClient:
    def __init__(self, base_url: str, timeout: int = 30):
        self.base = base_url
        self.timeout = timeout
        self._session = None

    def _post_binary(self, path: str, payload: bytes) -> tuple[int, bytes]:
        req = urllib.request.Request(
            f"{self.base}{path}",
            data=payload,
            headers={"Content-Type": "application/octet-stream"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()
        except Exception as e:
            return -1, str(e).encode()

    def ingest(self, seg_id: str, vectors: list[float], n: int, dim: int,
               index_type: str = "",
               nlist: int = 0, nprobe: int = 0,
               m: int = 0, nbits: int = 0,
               sq_type: str = "",
               ef_construction: int = 0) -> tuple[bool, float]:
        """Ingest vectors. Returns (success, server_elapsed_ms)."""
        payload = _build_ingest_payload(seg_id, vectors, n, dim,
                                        index_type, nlist, nprobe, m, nbits, sq_type,
                                        ef_construction)
        code, body = self._post_binary("/v1/internal/rpc/ingest_batch", payload)
        if code == 200 and body:
            try:
                resp = json.loads(body)
                elapsed = resp.get("elapsed_ms", 0.0)
                return True, elapsed
            except Exception:
                pass
        return code == 200, 0.0

    def query_serial(self, seg_id: str, query: list[float], dim: int,
                     topk: int) -> tuple[bool, int, int]:
        """Send single query (nq=1). Returns (ok, status_code, latency_ms)."""
        payload = _build_serial_query_payload(seg_id, query, dim, topk)
        t0 = time.time()
        code, body = self._post_binary("/v1/internal/rpc/query_warm", payload)
        lat_ms = (time.time() - t0) * 1000
        ok = code == 200
        return ok, code, lat_ms

    def query_batch(self, seg_id: str, queries: list[float], nq: int,
                    dim: int, topk: int) -> tuple[bool, int, float, list[int], list[float]]:
        """Send batch query. Returns (ok, status_code, latency_ms, ids, dists)."""
        payload = _build_batch_query_payload(seg_id, queries, nq, dim, topk)
        t0 = time.time()
        code, body = self._post_binary("/v1/internal/rpc/query_warm_batch", payload)
        lat_ms = (time.time() - t0) * 1000
        if code == 200:
            ids, dists = _parse_batch_response(body, nq, topk)
            return True, code, lat_ms, ids, dists
        return False, code, lat_ms, [], []

    def unload(self, seg_id: str) -> None:
        data = json.dumps({"segment_id": seg_id}).encode()
        req = urllib.request.Request(
            f"{self.base}/v1/internal/rpc/unload_segment",
            data=data, headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=10)
        except Exception:
            pass

    def register_warm(self, seg_id: str, n: int) -> None:
        obj_ids = [f"bench-p{i:06d}" for i in range(n)]
        data = json.dumps({"segment_id": seg_id, "object_ids": obj_ids}).encode()
        req = urllib.request.Request(
            f"{self.base}/v1/internal/rpc/register_warm",
            data=data, headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=10)
        except Exception:
            pass


def benchmark_plasmod(indexed, indexed_n, dim, queries, query_n, topk, idx_type):
    """
    Pure-HTTP Plasmod benchmark — no Go binary, no CGO.
    All queries sent as real HTTP requests to the running Plasmod server.

    Serial latency = wall-clock time of each individual HTTP request (nq=1).
    Batch latency  = wall-clock time of one HTTP request with all queries.
    """
    seg_id = "bench.layer1"
    server_url = "http://127.0.0.1:8080"
    http = _HTTPClient(server_url, timeout=60)

    # Map Python idx_type to Plasmod index type + IVF params
    # For IVF-PQ: nlist=1024, nprobe=1024, m=128 achieves ~0.91 recall per sweep
    INDEX_TYPE_MAP = {
        "ivf_flat": ("IVF_FLAT", 0, 0, 0, 0, ""),
        "ivf_pq":   ("IVF_PQ",   1024, 1024, 128, 8, ""),
        "ivf_sq8":  ("IVF_SQ8",  0, 0, 0, 0, "INT8"),
        "hnsw":     ("HNSW",     0, 0, 0, 0, ""),
        "flat":     ("IVF_FLAT", 0, 0, 0, 0, ""),
    }
    ptype, nlist, nprobe, m, nbits, sq_type = INDEX_TYPE_MAP.get(idx_type, ("HNSW", 0, 0, 0, 0, ""))
    print(f"      [Plasmod HTTP] index_type={ptype} idx_type={idx_type}")

    # ── 1. Unload any stale segment ────────────────────────────────────────────
    print(f"      [Plasmod HTTP] unloading segment={seg_id}")
    http.unload(seg_id)
    time.sleep(1)

    # ── 2. Ingest indexed vectors in chunks ─────────────────────────────────────
    print(f"      [Plasmod HTTP] ingesting {indexed_n} vectors (dim={dim})")
    t_build = time.time()
    server_build_ms = 0.0
    ingest_batch = min(500_000, indexed_n)
    for start in range(0, indexed_n, ingest_batch):
        t_batch = time.time()
        end = min(start + ingest_batch, indexed_n)
        batch_n = end - start
        ok, elapsed = http.ingest(seg_id, indexed[start * dim : end * dim], batch_n, dim,
                          index_type=ptype,
                          nlist=nlist, nprobe=nprobe, m=m, nbits=nbits, sq_type=sq_type)
        server_build_ms += elapsed
        if not ok:
            raise RuntimeError(f"ingest failed at batch {start}-{end}")
        print(f"      [Plasmod HTTP]   ingested {end}/{indexed_n}")

    # Register object IDs so the warm path works
    http.register_warm(seg_id, indexed_n)
    client_build_ms = (time.time() - t_build) * 1000
    build_ms = client_build_ms  # Keep as build_ms for Result
    print(f"      [Plasmod HTTP] build={build_ms:.1f}ms (server={server_build_ms:.1f}ms, overhead={client_build_ms - server_build_ms:.1f}ms)")

    # Warm-up: one serial request to prime the segment
    http.query_serial(seg_id, queries[:dim], dim, topk)

    # ── 3. True serial latency: nq individual HTTP requests (nq=1 each) ─────────
    print(f"      [Plasmod HTTP] serial search {query_n} queries...")
    serial_latencies = []
    errors = 0
    all_ids = []
    for i in range(query_n):
        query = [queries[i * dim + j] for j in range(dim)]
        ok, code, lat = http.query_serial(seg_id, query, dim, topk)
        serial_latencies.append(lat)
        if ok and len(serial_latencies) == 1:
            # parse ids from response for recall
            pass  # single query response has string ids, skip for now
        if not ok:
            errors += 1
    serial_ms = sum(serial_latencies)
    serial_qps = query_n / (serial_ms / 1000) if serial_ms > 0 else 0
    sl = sorted(serial_latencies)
    p50 = sl[int(len(sl) * 0.50)]
    p95 = sl[int(len(sl) * 0.95)]
    p99 = sl[int(len(sl) * 0.99)]
    mean = serial_ms / query_n if query_n else 0
    print(f"      [Plasmod HTTP] serial done: QPS={serial_qps:.1f} "
          f"mean={mean:.3f}ms p50={p50:.3f}ms p95={p95:.3f}ms p99={p99:.3f}ms")

    # ── 4. Batch latency: one HTTP request with all queries ─────────────────────
    print(f"      [Plasmod HTTP] batch search {query_n} queries...")
    batch_ok, _, batch_ms, batch_ids, _ = http.query_batch(
        seg_id, queries, query_n, dim, topk)
    if batch_ok:
        batch_qps = query_n / (batch_ms / 1000) if batch_ms > 0 else 0
        all_ids = batch_ids
        print(f"      [Plasmod HTTP] batch done: {batch_ms:.1f}ms QPS={batch_qps:.1f}")
    else:
        batch_ms = 0
        batch_qps = 0
        print(f"      [Plasmod HTTP] batch FAILED")

    # ── 5. Recall (batch IDs are already collected) ─────────────────────────────
    recall = _plasmod_recall(all_ids, indexed, indexed_n, dim, queries, query_n, topk)

    # ── 6. Memory of plasmod process (RSS + mmap'd file sizes) ─────────────────
    mem_val = 0.0
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "bin/plasmod"], text=True, timeout=5)
        pid = int(out.strip().split()[0])
        # Include mmap'd files in memory calculation for fair comparison
        # (Plasmod uses mmap for vector storage, which doesn't count in RSS)
        andb_data_dir = Path("/Users/erwin/Downloads/codespace/Plasmodexp/Plasmod/cpp/build/.andb_data")
        mem_val = mem_mb_with_mmap(pid, andb_data_dir)
    except Exception:
        pass

    # ── 7. Unload ──────────────────────────────────────────────────────────────
    http.unload(seg_id)

    return Result(
        db="Plasmod", index_type=idx_type.upper().replace("_", "-"),
        n_indexed=indexed_n, n_queries=query_n, dim=dim, topk=topk,
        build_ms=build_ms,
        batch_ms=batch_ms, batch_qps=batch_qps,
        serial_ms=serial_ms, serial_qps=serial_qps,
        p50_ms=p50, p95_ms=p95, p99_ms=p99,
        recall=recall, memory_mb=mem_val,
    )


# ─── Recall ───────────────────────────────────────────────────────────────────

def compute_recall_all(results: List[Result], indexed, indexed_n, dim, queries, query_n, topk):
    """Compute ground truth and recall for each result."""
    gt = brute_force_search(indexed, indexed_n, dim, queries, query_n, topk)
    for r in results:
        idx_type = r.index_type.lower().replace("-", "_")
        got_ids = _gather_got_ids(r.db, indexed_n, dim, queries, query_n, topk, idx_type)
        if got_ids:
            r.recall = recall_at_k(got_ids, gt, topk)


def _gather_got_ids(db, indexed_n, dim, queries, query_n, topk, idx_type=None):
    """Get search results from each DB for recall."""
    if db == "Qdrant":
        base = "http://127.0.0.1:6333"
        coll = "bench_test"
        try:
            batch_payload = {
                "searches": [
                    {"vector": queries[q * dim:(q + 1) * dim], "top": topk}
                        for q in range(query_n)
                ]
            }
            r = urllib.request.urlopen(
                urllib.request.Request(f"{base}/collections/{coll}/points/search/batch",
                                      data=json.dumps(batch_payload).encode(),
                                      headers={"Content-Type": "application/json"}, method="POST"),
                timeout=120)
            d = json.loads(r.read())
            return [[pt["id"] - 1 for pt in res] for res in d.get("result", [[]] * query_n)]
        except Exception:
            return []
    elif db == "Milvus":
        from pymilvus import MilvusClient
        client = MilvusClient(uri="http://127.0.0.1:19530")
        # Use bench_milvus for HNSW (sweep searches this directly)
        # Use bench_milvus_sweep for IVF (sweep creates this with correct index type)
        if idx_type == "hnsw":
            coll = "bench_milvus"
        else:
            coll = "bench_milvus_sweep"
        try:
            qvecs = [queries[q * dim:(q + 1) * dim] for q in range(query_n)]
            # Use same search params as during benchmark to get comparable recall
            if idx_type == "hnsw":
                search_params = {"ef": 256}
            elif idx_type in ("ivf_flat", "ivf_pq", "ivf_sq8"):
                search_params = {"nprobe": 4} if indexed_n < 10000 else {"nprobe": 16}
            else:
                search_params = {}
            res = client.search(coll, qvecs, limit=topk, search_params=search_params)
            return [[hit["id"] for hit in q] for q in res]
        except Exception as e:
            print(f"      [gather] Milvus recall failed: {e}")
            return []
    elif db == "LanceDB":
        import lancedb
        db_lance = lancedb.connect(str(BASE / "lancedb_data"))
        # Use bench_lancedb_sweep for recall (sweep creates this with correct params)
        table = "bench_lancedb_sweep"
        try:
            tbl_ref = db_lance.open_table(table)
            ids = []
            for q in range(query_n):
                qvec = [queries[q * dim + d] for d in range(dim)]
                result_table = tbl_ref.search(qvec).limit(topk).to_arrow()
                ids.append([row["id"] for row in result_table.to_pylist()])
            return ids
        except Exception as e:
            print(f"      [gather] LanceDB recall failed: {e}")
            return []
    elif db == "ChromaDB":
        import chromadb
        import json
        chroma_dir = str(BASE / "chromadb_data")
        client = chromadb.PersistentClient(path=chroma_dir)
        # Use bench_chroma_sweep for recall (sweep creates this)
        coll_name = "bench_chroma_sweep"
        try:
            c = client.get_collection(coll_name)
            qvecs = [[float(queries[q * dim + d]) for d in range(dim)] for q in range(query_n)]
            res = c.query(query_embeddings=qvecs, n_results=topk)
            ids_list = res.get("ids", [[] for _ in range(query_n)])
            return [[int(id_.split("_")[1]) for id_ in ids] for ids in ids_list]
        except Exception as e:
            # Fallback to file if sweep collection doesn't exist
            recall_file = BASE / "chromadb_batch_ids.json"
            try:
                with open(recall_file, "r") as f:
                    return json.load(f)
            except Exception:
                print(f"      [gather] ChromaDB recall failed: {e}")
                return []


# ─── Main ──────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nfcorpus", choices=["nfcorpus", "deep10M"])
    ap.add_argument("--index", default="all",
                   help="flat | ivf_flat | ivf_pq | ivf_sq8 | hnsw | all")
    ap.add_argument("--db", default="all",
                   help="all | qdrant | milvus | lancedb | chromadb | plasmod")
    ap.add_argument("--index-count", type=int, default=0,
                   help="0=all loaded vectors")
    ap.add_argument("--queries", type=int, default=100)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--sweep-recall", action="store_true",
                   help="Compute QPS at different recall levels (0.5-1.0)")
    args = ap.parse_args()

    # Load data
    if args.dataset == "nfcorpus":
        base = DATA / "nfcorpus"
        indexed, in_n, dim = load_fbin(str(base / "corpus.fbin"), args.index_count or 0)
        queries, qn, _  = load_fbin(str(base / "queries.fbin"), args.queries)
    else:
        base = DATA / "deep"
        indexed, in_n, dim = load_fbin(str(base / "base.10M.fbin"), args.index_count or 0)
        queries, qn, _  = load_fbin(str(base / "query.public.10K.fbin"), args.queries)

    n_idx = args.index_count or in_n
    n_q   = min(qn, args.queries)
    indexed = indexed[:n_idx * dim]
    queries = queries[:n_q * dim]

    print(f"\n=== Dataset: {args.dataset} | Indexed: {n_idx} | Queries: {n_q} | dim={dim} | topk={args.topk} ===")

    out_dir = OUT / f"{args.dataset}_n{n_idx}_q{n_q}_k{args.topk}"
    out_dir.mkdir(exist_ok=True)

    indices = ["ivf_flat", "ivf_pq", "ivf_sq8", "hnsw"] if args.index == "all" else [args.index]
    all_results = []
    db_stores = {}   # db_name -> list of (idx_label, indexed, queries, query_n, topk)
    # LanceDB VectorIndexType: IVF_FLAT, IVF_SQ, IVF_PQ, IVF_HNSW_SQ, IVF_HNSW_PQ, IVF_RQ
    # No pure HNSW; use IVF_HNSW_PQ for HNSW, IVF_SQ for SQ8
    lance_skip = set()  # nothing skipped — all types map to a LanceDB equivalent
    for idx in indices:
        idx_label = idx.upper().replace("_", "-")
        print(f"\n{'='*60}\n  Index: {idx_label}\n{'='*60}")

        # Qdrant
        if args.db in ("all", "qdrant"):
            print("  [Qdrant]")
            try:
                r = benchmark_qdrant(indexed, n_idx, dim, queries, n_q, args.topk, idx)
                all_results.append(r)
                r.save(out_dir)
                db_stores.setdefault("Qdrant", []).append((idx_label, indexed, queries, n_q, args.topk))
            except Exception as e:
                print(f"  Qdrant FAILED: {e}")

        # Milvus
        if args.db in ("all", "milvus"):
            print("  [Milvus]")
            try:
                r = benchmark_milvus(indexed, n_idx, dim, queries, n_q, args.topk, idx)
                all_results.append(r)
                r.save(out_dir)
                db_stores.setdefault("Milvus", []).append((idx_label, indexed, queries, n_q, args.topk))
            except Exception as e:
                print(f"  Milvus FAILED: {e}")

        # LanceDB (skips nothing — all types map to a LanceDB equivalent)
        if args.db in ("all", "lancedb"):
            if idx not in lance_skip:
                print("  [LanceDB]")
                try:
                    r = benchmark_lancedb(indexed, n_idx, dim, queries, n_q, args.topk, idx)
                    all_results.append(r)
                    r.save(out_dir)
                    db_stores.setdefault("LanceDB", []).append((idx_label, indexed, queries, n_q, args.topk))
                except Exception as e:
                    print(f"  LanceDB FAILED: {e}")
            else:
                print(f"  [LanceDB] SKIPPED (no {idx.upper()} support)")

        # ChromaDB
        if args.db in ("all", "chromadb"):
            print("  [ChromaDB]")
            try:
                r = benchmark_chromadb(indexed, n_idx, dim, queries, n_q, args.topk, idx)
                all_results.append(r)
                r.save(out_dir)
                db_stores.setdefault("ChromaDB", []).append((idx_label, indexed, queries, n_q, args.topk))
            except Exception as e:
                print(f"  ChromaDB FAILED: {e}")

        # Plasmod
        if args.db in ("all", "plasmod"):
            print("  [Plasmod]")
            try:
                r = benchmark_plasmod(indexed, n_idx, dim, queries, n_q, args.topk, idx)
                if r:
                    all_results.append(r)
                    r.save(out_dir)
            except Exception as e:
                print(f"  Plasmod FAILED: {e}")

    # Compute recall while collections are still alive
    print("\n  Computing Recall@K...")
    for db_name, runs in db_stores.items():
        for idx_label, idxd, q, nq, tk in runs:
            # Extract idx_type from idx_label (e.g., "IVF-FLAT" -> "ivf_flat")
            idx_type = idx_label.lower().replace("-", "_")
            # Check if recall is already computed by benchmark function
            for r in all_results:
                if r.db == db_name and r.index_type == idx_label:
                    if r.recall > 0.0:
                        print(f"    {db_name} {idx_label}: Recall@{tk}={r.recall:.4f} (computed in benchmark)")
                    else:
                        # Fallback: try to gather from collection
                        got_ids = gather_db_results(db_name, n_idx, dim, q, nq, tk, idx_type)
                        if got_ids:
                            gt = brute_force_search(idxd, n_idx, dim, q, nq, tk)
                            rec = recall_at_k(got_ids, gt, tk)
                            r.recall = rec
                            print(f"    {db_name} {idx_label}: Recall@{tk}={rec:.4f}")
                    break

    # Re-save all results now that recall has been computed
    for r in all_results:
        r.save(out_dir)

    # Recall-QPS sweep (before cleanup so collections are still alive)
    sweep_results = []
    if args.sweep_recall:
        print("\n  Computing Recall-QPS Sweep...")
        recall_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]

        for idx in indices:
            idx_label = idx.upper().replace("_", "-")

            # All DBs that support sweep: Qdrant, Milvus, Plasmod, ChromaDB
            # (LanceDB doesn't support dynamic ef/nprobe at search time)
            sweep_dbs = [d for d in ["Qdrant", "Milvus", "Plasmod", "ChromaDB"]
                        if args.db in ("all", d.lower())]

            for db_name in sweep_dbs:
                # ChromaDB only supports HNSW
                if db_name == "ChromaDB" and idx != "hnsw":
                    continue

                print(f"    [{db_name}] {idx_label} sweep...")
                try:
                    qps_at_recall, sweep_points = compute_recall_qps_sweep(
                        db_name, indexed, n_idx, dim, queries, n_q, args.topk, idx, recall_thresholds)

                    sweep_results.append({
                        "db": db_name,
                        "index_type": idx_label,
                        "qps_at_recall": {str(k): v for k, v in qps_at_recall.items()},
                        "sweep_points": sweep_points,  # Store raw sweep data
                    })
                except Exception as e:
                    print(f"    [{db_name}] sweep FAILED: {e}")

        # Print sweep table
        if sweep_results:
            print("\n  Recall-QPS Sweep Results:")
            header = "  {:<10} {:<12}".format("DB", "Index")
            for t in recall_thresholds:
                header += " {:>10}".format("Q@{:0.2f}".format(t))
            print(header)
            print("  " + "-" * (22 + 11 * len(recall_thresholds)))

            for sr in sweep_results:
                row = "  {:<10} {:<12}".format(sr['db'], sr['index_type'])
                for t in recall_thresholds:
                    qps = sr['qps_at_recall'].get(str(t), 0)
                    row += " {:>10.1f}".format(qps)
                print(row)

            # Save sweep results
            with open(out_dir / "sweep_recall_qps.json", "w") as f:
                json.dump({
                    "recall_thresholds": recall_thresholds,
                    "results": sweep_results,
                }, f, indent=2)
            print(f"\n  Sweep results saved to {out_dir}/sweep_recall_qps.json")

    # Cleanup collections
    print("\n  Cleaning up...")
    cleanup_all()

    # Summary table
    print_table(all_results)

    # Save summary
    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "dataset": args.dataset, "n_indexed": n_idx,
            "n_queries": n_q, "topk": args.topk,
            "results": [asdict(r) for r in all_results],
        }, f, indent=2)
    print(f"\nResults: {out_dir}/")


if __name__ == "__main__":
    main()
