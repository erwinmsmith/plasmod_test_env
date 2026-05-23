# 项目概述

## 项目结构

```
Plasmodexp/
├── Plasmod/                    # 核心数据库项目 (Go + C++)
│   ├── src/
│   │   ├── cmd/server/         # 服务入口
│   │   └── internal/          # 核心模块
│   │       ├── access/        # HTTP 网关
│   │       ├── dataplane/     # 数据层
│   │       ├── retrieval/     # 检索引擎
│   │       └── ...
│   ├── cpp/                   # C++ 检索层 (Knowhere/HNSW)
│   │   └── retrieval/
│   ├── configs/               # 配置文件
│   └── Makefile               # 构建脚本
│
└── plasmod_test_env/           # 实验环境
    ├── scripts/
    │   ├── benchmark_all.py   # 主 benchmark 脚本
    │   └── summary_to_csv.py  # 结果转换
    ├── data/                  # 数据集
    │   ├── nfcorpus/          # 小数据集 (3.6K vectors)
    │   └── deep/              # 大数据集 (10M vectors)
    ├── models/                # ONNX 模型
    │   └── all-MiniLM-L6-v2.onnx
    ├── results/               # 实验结果
    ├── start_all.sh           # 启动所有服务
    ├── verify_env.sh          # 验证环境
    └── .env                   # 环境配置
```

---

## 核心组件

### Plasmod (核心数据库)

| 组件 | 说明 |
|------|------|
| **Go Server** | HTTP API 服务 (端口 8080) |
| **C++ 检索层** | Knowhere/HNSW 实现，通过 CGO 调用 |
| **支持索引** | IVF-FLAT, IVF-PQ, IVF-SQ8, HNSW |

### 实验环境数据库

| 数据库 | 端口 | 类型 | 说明 |
|--------|------|------|------|
| **Plasmod** | 8080 | Go + C++ | 主测试对象 |
| **Qdrant** | 6333 (HTTP), 6334 (gRPC) | Rust | 向量数据库 |
| **Milvus** | 19530 (gRPC), 9091 (Console) | Go + C++ | 通过 Docker 运行 |
| **Milvus-MinIO** | 9002 | Docker 内部 | Milvus 依赖的 MinIO |
| **MinIO** | 9000 (API), 9001 (Console) | S3 存储 | Plasmod 冷存储 |
| **LanceDB** | - | Python | 嵌入式，无端口 |
| **ChromaDB** | - | Python | 嵌入式，无端口 |

### 数据集

| 数据集 | 向量数 | 维度 | 用途 |
|--------|--------|------|------|
| **nfcorpus** | 3,633 | 384 | 快速测试 |
| **deep10M** | 10,000,000 | 96 | 完整压测 (~3.8GB) |

---

## Benchmark 脚本

### benchmark_all.py

主 benchmark 脚本，对比 Qdrant、Milvus、LanceDB、ChromaDB、Plasmod 的性能。

**关键函数：**

| 函数 | 说明 |
|------|------|
| `load_fbin()` | 读取 .fbin 二进制格式数据 |
| `brute_force_search()` | 计算 ground truth |
| `benchmark_qdrant()` | Qdrant HTTP REST API 测试 |
| `benchmark_milvus()` | Milvus Python SDK 测试 |
| `benchmark_lancedb()` | LanceDB Python SDK 测试 |
| `benchmark_chromadb()` | ChromaDB Python SDK 测试 |
| `benchmark_plasmod()` | Plasmod HTTP + 二进制协议测试 |
| `compute_recall_qps_sweep()` | 不同 recall 级别的 QPS 测量 |

**指标：**

- Build 时间
- Batch QPS / Serial QPS
- Recall@K
- P50/P95/P99 延迟
- 内存使用 (RSS + mmap)

### summary_to_csv.py

将 `summary.json` 转换为 CSV 格式。

```bash
python3 scripts/summary_to_csv.py results/nfcorpus_n3633_q3237_k10/summary.json output.csv
```