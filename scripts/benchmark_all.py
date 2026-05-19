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

    metric = "Cosine"
    # HNSW is always on; IVF types configure optimizer quantization
    cfg = {}
    if idx_type == "ivf_flat":
        cfg = {"indexing": {"metric_type": "Cosine"}}
    elif idx_type == "ivf_pq":
        cfg = {"indexing": {"metric_type": "Cosine", "compression": "PQ", "compression_ratio": 16}}
    elif idx_type == "ivf_sq8":
        cfg = {"indexing": {"metric_type": "Cosine", "compression": "SQ", "compression": 8}}

    req = {"vectors": {"size": dim, "distance": metric}}
    if idx_type == "hnsw":
        req["hnsw_config"] = {"m": 16, "ef_construct": 256}
        req["optimizer_config"] = {}
    else:
        req["hnsw_config"] = {"m": 16, "ef_construct": 256}
        req["optimizer_config"] = cfg

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

    # Batch search (single HTTP call for all queries)
    batch_payload = {
        "searches": [
            {"vector": queries[q * dim:(q + 1) * dim], "top": topk, "with_vectors": False}
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
    r.read()
    batch_ms = (time.time() - t0) * 1000
    batch_qps = query_n / (batch_ms / 1000) if batch_ms > 0 else 0

    # Serial search
    latencies = []
    for q in range(query_n):
        body = json.dumps({"vector": queries[q * dim:(q + 1) * dim], "top": topk}).encode()
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
        recall=0.0, memory_mb=mem_mb,
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
    # Plasmod (segment)
    try:
        urllib.request.urlopen(
            urllib.request.Request("http://127.0.0.1:8080/v1/internal/rpc/unload_segment",
                                 data=json.dumps({"segment_id": "bench.layer1"}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST"),
            timeout=10)
    except Exception:
        pass


def gather_db_results(db, indexed_n, dim, queries, query_n, topk):
    """Gather search results from each DB for recall computation."""
    if db == "Qdrant":
        base = "http://127.0.0.1:6333"
        coll = "bench_test"
        try:
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
    elif db == "Milvus":
        from pymilvus import MilvusClient
        client = MilvusClient(uri="http://127.0.0.1:19530")
        coll = "bench_milvus"
        try:
            qvecs = [queries[q * dim:(q + 1) * dim] for q in range(query_n)]
            res = client.search(coll, qvecs, limit=topk)
            return [[hit["id"] for hit in q] for q in res]
        except Exception:
            return []
    # Recall: LanceDB bench collection is still alive — use bench_lancedb
    elif db == "LanceDB":
        import lancedb
        db_lance = lancedb.connect(str(BASE / "lancedb_data"))
        coll = "bench_lancedb"
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

    index_map = {
        "ivf_flat": ("IVF_FLAT", {"nlist": 128, "nprobe": 10}),
        "ivf_pq":   ("IVF_PQ",   {"nlist": 128, "nprobe": 10, "m": 16, "nbits": 8}),
        "ivf_sq8":  ("IVF_SQ8",  {"nlist": 128, "nprobe": 10}),
        "hnsw":     ("HNSW",    {"M": 16, "efConstruction": 256}),
    }
    idx_name, idx_params = index_map.get(idx_type, ("IVF_FLAT", {"nlist": 128, "nprobe": 10}))

    ip = IndexParams()
    ip.add_index("vector", idx_name, **idx_params)
    client.create_collection(coll, dimension=dim, index_params=ip, auto_id=False)
    client.flush(coll)

    # Insert: pymilvus 3.x expects List[Dict] with column keys
    t0 = time.time()
    for batch_start in range(0, indexed_n, 500):
        batch_end = min(batch_start + 500, indexed_n)
        rows = [{"id": i, "vector": indexed[i * dim:(i + 1) * dim]}
                for i in range(batch_start, batch_end)]
        client.insert(coll, rows)
    client.flush(coll)
    build_ms = (time.time() - t0) * 1000

    # Batch search
    qvecs = [queries[q * dim:(q + 1) * dim] for q in range(query_n)]
    t0 = time.time()
    client.search(coll, qvecs, limit=topk, search_params={"ef": 256})
    batch_ms = (time.time() - t0) * 1000
    batch_qps = query_n / (batch_ms / 1000) if batch_ms > 0 else 0

    # Serial search
    latencies = []
    for q in range(query_n):
        t = time.time()
        client.search(coll, [qvecs[q]], limit=topk, search_params={"ef": 256})
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
        recall=0.0, memory_mb=mem_mb,
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
        # IVF_PQ: minimal sub-vectors for small datasets
        tbl_ref.create_index(metric="cosine", vector_column_name="vector",
                             index_type="IVF_PQ", num_partitions=1, num_sub_vectors=8,
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
    for q in qvecs:
        tbl_ref.search(q).limit(topk).to_arrow()
    batch_ms = (time.time() - t0) * 1000
    batch_qps = query_n / (batch_ms / 1000) if batch_ms > 0 else 0

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
        recall=0.0, memory_mb=mem_mb,
    )


# ─── Plasmod ───────────────────────────────────────────────────────────────────

def benchmark_plasmod(indexed, indexed_n, dim, queries, query_n, topk, idx_type):
    """Uses the benchmark binary in http-query mode."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".fbin", delete=False) as f1, \
         tempfile.NamedTemporaryFile(suffix=".fbin", delete=False) as f2:
        tmp_idx, tmp_q = f1.name, f2.name

    def write_fbin(path, vecs, n, d):
        with open(path, "wb") as f:
            f.write(struct.pack("<II", n, d))
            f.write(struct.pack(f"<{n*d}f", *vecs))

    write_fbin(tmp_idx, indexed, indexed_n, dim)
    write_fbin(tmp_q, queries, query_n, dim)

    # For FLAT vs IVF: Plasmod uses HNSW internally, toggle via plugin flag
    plugin = "0" if idx_type == "flat" else "1"
    mode = "http-query-raw" if idx_type == "flat" else "http-query"

    try:
        r = subprocess.run(
            [str(BASE / "bin" / "benchmark"),
             "--mode", mode,
             "--server-url", "http://127.0.0.1:8080",
             "--segment", "bench.layer1",
             "--indexed-dataset", tmp_idx,
             "--query-dataset", tmp_q,
             "--queries", str(query_n),
             "--topk", str(topk),
             "--plugin", plugin,
             "--batch-size", str(query_n)],
            capture_output=True, text=True, timeout=600,
            cwd=str(BASE),
        )
        stderr_clean = [l for l in r.stderr.splitlines()
                       if "FD from fork" not in l and l.strip()]
        if r.returncode != 0 and stderr_clean:
            print(f"    Plasmod stderr: {'; '.join(stderr_clean[:3])}")
        output = r.stdout.strip().split("\n")[-1]
        data = json.loads(output)
        # Memory (plasmod process)
        pid = None
        try:
            out = subprocess.check_output(
                ["pgrep", "-f", "bin/plasmod"], text=True, timeout=5)
            pid = int(out.strip().split()[0])
        except Exception:
            pass
        mem_val = mem_mb(pid) if pid else 0.0

        return Result(
            db="Plasmod", index_type=idx_type.upper().replace("_", "-"),
            n_indexed=indexed_n, n_queries=query_n, dim=dim, topk=topk,
            build_ms=data["build_ms"],
            batch_ms=data["batch_ms"], batch_qps=data["batch_qps"],
            serial_ms=data["serial_ms"], serial_qps=data["serial_qps"],
            p50_ms=data["p50_ms"], p95_ms=data["p95_ms"], p99_ms=data["p99_ms"],
            recall=_plasmod_recall(data.get("int_ids", []), indexed, indexed_n, dim, queries, query_n, topk),
            memory_mb=mem_val,
        )
    finally:
        os.unlink(tmp_idx)
        os.unlink(tmp_q)


# ─── Recall ───────────────────────────────────────────────────────────────────

def compute_recall_all(results: List[Result], indexed, indexed_n, dim, queries, query_n, topk):
    """Compute ground truth and recall for each result."""
    gt = brute_force_search(indexed, indexed_n, dim, queries, query_n, topk)
    for r in results:
        got_ids = _gather_got_ids(r.db, indexed_n, dim, queries, query_n, topk)
        if got_ids:
            r.recall = recall_at_k(got_ids, gt, topk)


def _gather_got_ids(db, indexed_n, dim, queries, query_n, topk):
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
        coll = "bench_milvus"
        try:
            qvecs = [queries[q * dim:(q + 1) * dim] for q in range(query_n)]
            res = client.search(coll, qvecs, limit=topk)
            return [[hit["id"] for hit in q] for q in res]
        except Exception:
            return []
    elif db == "LanceDB":
        import lancedb
        db_lance = lancedb.connect(str(BASE / "lancedb_data"))
        table = "bench_test"
        try:
            tbl_ref = db_lance.open_table(table)
            return [[] for _ in range(query_n)]
        except Exception:
            return []
        finally:
            try:
                db_lance.drop_table(table)
            except Exception:
                pass
    return []


# ─── Main ──────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nfcorpus", choices=["nfcorpus", "deep10M"])
    ap.add_argument("--index", default="all",
                   help="flat | ivf_flat | ivf_pq | all")
    ap.add_argument("--index-count", type=int, default=0,
                   help="0=all loaded vectors")
    ap.add_argument("--queries", type=int, default=100)
    ap.add_argument("--topk", type=int, default=10)
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
        print("  [Qdrant]")
        try:
            r = benchmark_qdrant(indexed, n_idx, dim, queries, n_q, args.topk, idx)
            all_results.append(r)
            r.save(out_dir)
            db_stores.setdefault("Qdrant", []).append((idx_label, indexed, queries, n_q, args.topk))
        except Exception as e:
            print(f"  Qdrant FAILED: {e}")

        # Milvus
        print("  [Milvus]")
        try:
            r = benchmark_milvus(indexed, n_idx, dim, queries, n_q, args.topk, idx)
            all_results.append(r)
            r.save(out_dir)
            db_stores.setdefault("Milvus", []).append((idx_label, indexed, queries, n_q, args.topk))
        except Exception as e:
            print(f"  Milvus FAILED: {e}")

        # LanceDB (skips nothing — all types map to a LanceDB equivalent)
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

        # Plasmod (HNSW-only DB; flat returns None for non-HNSW types)
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
            got_ids = gather_db_results(db_name, n_idx, dim, q, nq, tk)
            if got_ids:
                gt = brute_force_search(idxd, n_idx, dim, q, nq, tk)
                rec = recall_at_k(got_ids, gt, tk)
                for r in all_results:
                    if r.db == db_name and r.index_type == idx_label:
                        r.recall = rec
                        break
                print(f"    {db_name} {idx_label}: Recall@{tk}={rec:.4f}")

    # Re-save all results now that recall has been computed
    for r in all_results:
        r.save(out_dir)

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
