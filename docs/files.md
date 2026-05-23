# 文件管理

## 目录结构

### 数据存储目录

| 目录 | 说明 |
|------|------|
| `.andb_data/` | Plasmod 数据 |
| `lancedb_data/` | LanceDB 数据 |
| `chromadb_data*/` | ChromaDB 数据 (多个) |
| `data/nfcorpus/` | nfcorpus 数据集 |
| `data/deep/` | deep10M 数据集 |
| `models/` | ONNX 模型 |

### 结果存储目录

```
plasmod_test_env/results/
├── nfcorpus_n3633_q3237_k10/     # 每次实验一个目录
│   ├── summary.json              # 汇总结果
│   ├── sweep_recall_qps.json     # Recall-QPS 数据
│   ├── Plasmod_HNSW.json         # 各 DB/索引结果
│   ├── Plasmod_IVF-FLAT.json
│   ├── Plasmod_IVF-PQ.json
│   ├── Plasmod_IVF-SQ8.json
│   ├── Qdrant_HNSW.json
│   ├── Milvus_HNSW.json
│   └── ...
│
├── deep10M_n10000000_q10000_k10/
│   └── ...
│
└── nfcorpus_n3633_q100_k10/
    └── ...
```

## 二进制数据格式

### .fbin (向量数据)

```
Header: [n: uint32][dim: uint32]
Body: [n * dim * float32]
```

### .ibin (ground truth)

```
Header: [nq: uint32][topk: uint32]
Body: [nq * topk * int32]
```

## 实验结果格式

### 单个结果 JSON

```json
{
  "db": "Plasmod",
  "index_type": "HNSW",
  "n_indexed": 3633,
  "n_queries": 3237,
  "dim": 384,
  "topk": 10,
  "build_ms": 1234.56,
  "batch_ms": 56.78,
  "batch_qps": 57040.12,
  "serial_ms": 890.12,
  "serial_qps": 3637.89,
  "p50_ms": 0.25,
  "p95_ms": 0.35,
  "p99_ms": 0.45,
  "recall": 0.999,
  "memory_mb": 256.0,
  "qps_at_recall": {
    "0.5": 57040.12,
    "0.6": 57040.12
  }
}
```

### summary.json 格式

```json
{
  "dataset": "nfcorpus",
  "n_indexed": 3633,
  "n_queries": 3237,
  "topk": 10,
  "results": [/* array of Result objects */]
}
```

## 数据管理命令

```bash
# 导出结果为 CSV
python3 scripts/summary_to_csv.py results/nfcorpus_n3633_q3237_k10/summary.json output.csv

# 清理结果目录 (保留最近一次)
rm -rf results/old_*

# 清理数据存储 (重新开始)
rm -rf .andb_data/
rm -rf lancedb_data/
rm -rf chromadb_data*/

# 查看数据集大小
du -sh data/
du -sh results/
```