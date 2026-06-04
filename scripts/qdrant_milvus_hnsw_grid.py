#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
BASE = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import benchmark_all as bench  # noqa: E402


DEFAULT_TARGETS = "0.8,0.85,0.9,0.95,1.0"
DEFAULT_M_VALUES = "2,4,8,16"
DEFAULT_BUILD_EF_VALUES = "8,16,32,64"
DEFAULT_SEARCH_EF_VALUES = "12,16,24,32,48,64,96,128,192,256"


@dataclass
class GridPoint:
    method: str
    db: str
    build_m: int
    build_ef_construct: int
    search_ef: int
    recall: float
    batch_ms: float
    batch_qps: float
    serial_ms: float | None
    serial_qps: float | None
    n_indexed: int
    n_queries: int
    dim: int
    topk: int
    build_ms: float
    notes: str = ""


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_ints(raw: str) -> list[int]:
    values = [int(p.strip()) for p in raw.split(",") if p.strip()]
    if not values:
        raise ValueError("empty int list")
    return values


def parse_floats(raw: str) -> list[float]:
    values = [float(p.strip()) for p in raw.split(",") if p.strip()]
    if not values:
        raise ValueError("empty float list")
    return values


def load_deep(index_count: int, queries_count: int):
    base = bench.DATA / "deep"
    indexed, in_n, dim = bench.load_fbin(str(base / "base.10M.fbin"), index_count)
    queries, qn, _ = bench.load_fbin(str(base / "query.public.10K.fbin"), queries_count)
    n_idx = min(index_count, in_n)
    n_q = min(queries_count, qn)
    return indexed[: n_idx * dim], n_idx, dim, queries[: n_q * dim], n_q


def load_ground_truth(path: Path, indexed, n_idx: int, dim: int, queries, n_q: int, topk: int):
    if path.exists():
        import numpy as np

        _log(f"Loading cached ground truth: {path}")
        return np.load(path)[:, :topk].tolist()
    gt = bench.brute_force_search(indexed, n_idx, dim, queries, n_q, topk)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import numpy as np

        np.save(path, np.asarray(gt, dtype=np.int32))
    except Exception:
        pass
    return gt


def iter_query_chunks(queries, n_q: int, dim: int, chunk_size: int):
    for start in range(0, n_q, chunk_size):
        end = min(start + chunk_size, n_q)
        yield start, end, queries[start * dim : end * dim]


def build_qdrant(indexed, n_idx: int, dim: int, m: int, ef_construct: int, coll: str) -> float:
    base = "http://127.0.0.1:6333"
    try:
        urllib.request.urlopen(
            urllib.request.Request(f"{base}/collections/{coll}", method="DELETE"),
            timeout=30,
        ).read()
    except Exception:
        pass
    body = {
        "vectors": {"size": dim, "distance": "Cosine"},
        "hnsw_config": {
            "m": m,
            "ef_construct": ef_construct,
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
    _log(f"[Qdrant] ingesting {n_idx:,} vectors m={m} efc={ef_construct}")
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


def search_qdrant(queries, n_q: int, dim: int, topk: int, search_ef: int, coll: str, chunk_size: int):
    base = "http://127.0.0.1:6333"
    ids: list[list[int]] = []
    t0 = time.time()
    for start, end, qchunk in iter_query_chunks(queries, n_q, dim, chunk_size):
        chunk_n = end - start
        payload = {
            "searches": [
                {
                    "vector": qchunk[i * dim : (i + 1) * dim],
                    "top": topk,
                    "with_vectors": False,
                    "params": {"hnsw": {"ef": search_ef, "exact": False}},
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
        ids.extend([[int(pt["id"]) - 1 for pt in row] for row in data.get("result", [])])
    return ids, (time.time() - t0) * 1000


def serial_qps_qdrant(queries, n_q: int, dim: int, topk: int, search_ef: int, coll: str, sample_n: int):
    if sample_n <= 0:
        return None, None
    sample_n = min(sample_n, n_q)
    base = "http://127.0.0.1:6333"
    lat = []
    for q in range(sample_n):
        body = bench._json_dumps({
            "vector": queries[q * dim : (q + 1) * dim],
            "top": topk,
            "with_vectors": False,
            "params": {"hnsw": {"ef": search_ef, "exact": False}},
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
        lat.append((time.time() - t0) * 1000)
    serial_ms = (sum(lat) / sample_n) * n_q
    return serial_ms, n_q / (serial_ms / 1000)


def run_qdrant(indexed, n_idx: int, dim: int, queries, n_q: int, topk: int,
               gt, m_values: list[int], build_efs: list[int], search_efs: list[int],
               chunk_size: int, serial_samples: int) -> list[GridPoint]:
    coll = "bench_hnsw_grid"
    out: list[GridPoint] = []
    for m in m_values:
        for efc in build_efs:
            try:
                build_ms = build_qdrant(indexed, n_idx, dim, m, efc, coll)
            except Exception as e:
                _log(f"[Qdrant] build failed m={m} efc={efc}: {e}")
                continue
            for ef in search_efs:
                _log(f"[Qdrant] search m={m} efc={efc} ef={ef}")
                ids, batch_ms = search_qdrant(queries, n_q, dim, topk, ef, coll, chunk_size)
                recall = bench.recall_at_k(ids, gt, topk)
                serial_ms, serial_qps = serial_qps_qdrant(queries, n_q, dim, topk, ef, coll, serial_samples)
                out.append(GridPoint(
                    method="Qdrant HNSW",
                    db="Qdrant",
                    build_m=m,
                    build_ef_construct=efc,
                    search_ef=ef,
                    recall=recall,
                    batch_ms=batch_ms,
                    batch_qps=n_q / (batch_ms / 1000) if batch_ms else 0.0,
                    serial_ms=serial_ms,
                    serial_qps=serial_qps,
                    n_indexed=n_idx,
                    n_queries=n_q,
                    dim=dim,
                    topk=topk,
                    build_ms=build_ms,
                ))
    return out


def build_milvus(indexed, n_idx: int, dim: int, m: int, ef_construct: int, coll: str):
    from pymilvus import MilvusClient
    from pymilvus.milvus_client.index import IndexParams

    client = MilvusClient(uri="http://127.0.0.1:19530")
    try:
        client.drop_collection(coll)
    except Exception:
        pass
    ip = IndexParams()
    ip.add_index("vector", "HNSW", m=m, efConstruction=ef_construct)
    client.create_collection(coll, dimension=dim, index_params=ip, auto_id=False)
    client.flush(coll)
    _log(f"[Milvus] ingesting {n_idx:,} vectors m={m} efc={ef_construct}")
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


def search_milvus(client, coll: str, qvecs, topk: int, search_ef: int, chunk_size: int):
    ids: list[list[int]] = []
    t0 = time.time()
    for start in range(0, len(qvecs), chunk_size):
        end = min(start + chunk_size, len(qvecs))
        res = client.search(coll, qvecs[start:end], limit=topk, search_params={"ef": search_ef})
        ids.extend([[int(hit["id"]) for hit in row] for row in res])
    return ids, (time.time() - t0) * 1000


def serial_qps_milvus(client, coll: str, qvecs, topk: int, search_ef: int, n_q: int, sample_n: int):
    if sample_n <= 0:
        return None, None
    sample_n = min(sample_n, n_q)
    lat = []
    for qvec in qvecs[:sample_n]:
        t0 = time.time()
        client.search(coll, [qvec], limit=topk, search_params={"ef": search_ef})
        lat.append((time.time() - t0) * 1000)
    serial_ms = (sum(lat) / sample_n) * n_q
    return serial_ms, n_q / (serial_ms / 1000)


def run_milvus(indexed, n_idx: int, dim: int, queries, n_q: int, topk: int,
               gt, m_values: list[int], build_efs: list[int], search_efs: list[int],
               chunk_size: int, serial_samples: int) -> list[GridPoint]:
    coll = "bench_hnsw_grid"
    out: list[GridPoint] = []
    qvecs = [queries[q * dim : (q + 1) * dim] for q in range(n_q)]
    for m in m_values:
        for efc in build_efs:
            try:
                client, build_ms = build_milvus(indexed, n_idx, dim, m, efc, coll)
            except Exception as e:
                _log(f"[Milvus] build failed m={m} efc={efc}: {e}")
                continue
            for ef in search_efs:
                _log(f"[Milvus] search m={m} efc={efc} ef={ef}")
                try:
                    ids, batch_ms = search_milvus(client, coll, qvecs, topk, ef, chunk_size)
                    recall = bench.recall_at_k(ids, gt, topk)
                    serial_ms, serial_qps = serial_qps_milvus(client, coll, qvecs, topk, ef, n_q, serial_samples)
                except Exception as e:
                    _log(f"[Milvus] search failed m={m} efc={efc} ef={ef}: {e}")
                    continue
                out.append(GridPoint(
                    method="Milvus HNSW",
                    db="Milvus",
                    build_m=m,
                    build_ef_construct=efc,
                    search_ef=ef,
                    recall=recall,
                    batch_ms=batch_ms,
                    batch_qps=n_q / (batch_ms / 1000) if batch_ms else 0.0,
                    serial_ms=serial_ms,
                    serial_qps=serial_qps,
                    n_indexed=n_idx,
                    n_queries=n_q,
                    dim=dim,
                    topk=topk,
                    build_ms=build_ms,
                ))
    return out


def closest_above_summary(points: list[GridPoint], targets: list[float]) -> list[dict[str, Any]]:
    by_method: dict[str, list[GridPoint]] = {}
    for p in points:
        by_method.setdefault(p.method, []).append(p)
    rows = []
    for method, pts in by_method.items():
        for target in targets:
            qualifying = [p for p in pts if p.recall >= target]
            achieved = bool(qualifying)
            if achieved:
                best = min(qualifying, key=lambda p: (p.recall, -p.batch_qps))
            else:
                best = max(pts, key=lambda p: (p.recall, p.batch_qps))
            rows.append({
                "method": method,
                "target_recall": target,
                "achieved": achieved,
                "selected_recall": best.recall,
                "build_m": best.build_m,
                "build_ef_construct": best.build_ef_construct,
                "search_ef": best.search_ef,
                "batch_qps": best.batch_qps,
                "serial_qps": best.serial_qps,
                "fallback_reason": "" if achieved else "target not reached; selected highest recall point",
            })
    return rows


def write_outputs(out_dir: Path, points: list[GridPoint], summary: list[dict[str, Any]], metadata: dict[str, Any]):
    out_dir.mkdir(parents=True, exist_ok=True)
    point_rows = [asdict(p) for p in points]
    with open(out_dir / "grid_points.json", "w") as f:
        json.dump({"metadata": metadata, "points": point_rows}, f, indent=2)
    with open(out_dir / "target_summary.json", "w") as f:
        json.dump({"metadata": metadata, "targets": summary}, f, indent=2)
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"metadata": metadata, "points": point_rows, "target_summary": summary}, f, indent=2)
    if point_rows:
        with open(out_dir / "grid_points.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(point_rows[0].keys()))
            writer.writeheader()
            writer.writerows(point_rows)
    if summary:
        with open(out_dir / "target_summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)


def print_summary(rows: list[dict[str, Any]]) -> None:
    print("\nTarget recall summary")
    print(f"{'Method':<14} {'Target':>7} {'OK':>3} {'Recall':>8} {'M':>3} {'efC':>5} {'ef':>4} {'Batch QPS':>11} {'Serial QPS':>11}")
    print("-" * 86)
    for r in rows:
        serial = "" if r["serial_qps"] is None else f"{r['serial_qps']:.1f}"
        print(
            f"{r['method']:<14} {r['target_recall']:>7.2f} {'yes' if r['achieved'] else 'no':>3} "
            f"{r['selected_recall']:>8.4f} {r['build_m']:>3} {r['build_ef_construct']:>5} "
            f"{r['search_ef']:>4} {r['batch_qps']:>11.1f} {serial:>11}"
        )


def main():
    ap = argparse.ArgumentParser(description="Qdrant/Milvus HNSW build+search grid for recall-controlled 1M/10k experiments.")
    ap.add_argument("--db", default="qdrant,milvus", help="qdrant,milvus or one of them")
    ap.add_argument("--index-count", type=int, default=1_000_000)
    ap.add_argument("--queries", type=int, default=10_000)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--m-values", default=DEFAULT_M_VALUES)
    ap.add_argument("--build-ef-values", default=DEFAULT_BUILD_EF_VALUES)
    ap.add_argument("--search-ef-values", default=DEFAULT_SEARCH_EF_VALUES)
    ap.add_argument("--targets", default=DEFAULT_TARGETS)
    ap.add_argument("--batch-chunk-size", type=int, default=1000)
    ap.add_argument("--serial-samples", type=int, default=500)
    ap.add_argument("--groundtruth-cache", default="results/groundtruth_deep10M_n1000000_q10000_k10.npy")
    ap.add_argument("--output-dir", default="")
    args = ap.parse_args()

    dbs = [d.strip().lower() for d in args.db.split(",") if d.strip()]
    m_values = parse_ints(args.m_values)
    build_efs = parse_ints(args.build_ef_values)
    search_efs = [ef for ef in parse_ints(args.search_ef_values) if ef > args.topk]
    targets = parse_floats(args.targets)
    out_dir = Path(args.output_dir) if args.output_dir else (
        bench.OUT / f"qdrant_milvus_hnsw_grid_deep10M_n{args.index_count}_q{args.queries}_k{args.topk}_{time.strftime('%Y%m%d_%H%M%S')}"
    )

    _log(f"Starting grid dbs={dbs} m={m_values} build_ef={build_efs} search_ef={search_efs}")
    indexed, n_idx, dim, queries, n_q = load_deep(args.index_count, args.queries)
    gt = load_ground_truth(Path(args.groundtruth_cache), indexed, n_idx, dim, queries, n_q, args.topk)

    points: list[GridPoint] = []
    if "qdrant" in dbs:
        points.extend(run_qdrant(indexed, n_idx, dim, queries, n_q, args.topk, gt,
                                 m_values, build_efs, search_efs, args.batch_chunk_size, args.serial_samples))
    if "milvus" in dbs:
        points.extend(run_milvus(indexed, n_idx, dim, queries, n_q, args.topk, gt,
                                 m_values, build_efs, search_efs, args.batch_chunk_size, args.serial_samples))

    summary = closest_above_summary(points, targets)
    metadata = {
        "dataset": "deep10M",
        "n_indexed": n_idx,
        "n_queries": n_q,
        "dim": dim,
        "topk": args.topk,
        "dbs": dbs,
        "m_values": m_values,
        "build_ef_values": build_efs,
        "search_ef_values": search_efs,
        "targets": targets,
        "selection_policy": "closest_above",
        "notes": [
            "Target rows select the lowest measured recall >= target.",
            "If target is unreachable, the highest measured recall point is selected.",
        ],
    }
    write_outputs(out_dir, points, summary, metadata)
    print_summary(summary)
    _log(f"Saved results to {out_dir}")


if __name__ == "__main__":
    main()
