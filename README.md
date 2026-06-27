# Plasmod Multi-DB Benchmark

横向对比 **Plasmod / Qdrant / Milvus / LanceDB / ChromaDB** 在
ANN 索引（FLAT / IVF-FLAT / IVF-PQ / IVF-SQ8 / HNSW）下的性能与 Recall。

主脚本：`scripts/benchmark_all.py`

HNSW recall/QPS 专项 sweep：`scripts/hnsw_recall_sweep.py`

---

## 0. 先决条件

### Python 环境（必看）

包都装在 **conda base** 里，**不要**用 brew python：

```bash
# 验证 python 路径
which python
# 期望：/Users/codesoul/miniforge3/bin/python
```

若不是，临时切换：

```bash
source /Users/codesoul/miniforge3/etc/profile.d/conda.sh && conda activate base && hash -r
```

或在所有命令里直接用绝对路径 `/Users/codesoul/miniforge3/bin/python`（推荐，下文都用这种）。

### 启动所有服务

```bash
cd /Users/codesoul/Downloads/codespace/Plasmodexp/plasmod_test_env
bash start_all.sh
```

预期看到所有端口 UP：MinIO(9000) / Qdrant(6333) / Milvus(19530) / Plasmod(8080)。

---

## 1. 三档实验命令

每条都是**单行版本**（粘贴不会断），都通过 `nohup + caffeinate` 后台跑、防 mac 休眠。

### 档位 A —— 冒烟测试（~3-10 分钟）

验证代码改动没坏、五个 DB 都能通。

```bash
mkdir -p logs && nohup caffeinate -i /Users/codesoul/miniforge3/bin/python scripts/benchmark_all.py --dataset deep10M --index-count 10000 --index all --queries 500 --topk 10 > logs/deep_smoke_10K_500.log 2>&1 &
```

```bash
echo "PID=$!" && tail -f logs/deep_smoke_10K_500.log
```

### 档位 B —— 中等规模（~30-90 分钟）

Recall 数据足够稳定，速度差异已显著，可出初步对比。

```bash
mkdir -p logs && nohup caffeinate -i /Users/codesoul/miniforge3/bin/python scripts/benchmark_all.py --dataset deep10M --index-count 1000000 --index all --queries 10000 --topk 10 > logs/deep_1M_10K.log 2>&1 &
```

```bash
echo "PID=$!" && tail -f logs/deep_1M_10K.log
```

### HNSW Recall/QPS sweep —— 1M index / 10k queries / topk=10

用于比较 Qdrant / Milvus / Plasmod 的 HNSW 在不同 recall target 下的最高 QPS。
默认 recall targets 是 `0.8,0.85,0.9,0.95,1.0`，其中 `0.85` 对应实验记录里的 `9.85/0.85` 项。

```bash
mkdir -p logs && nohup caffeinate -i /Users/codesoul/miniforge3/bin/python scripts/hnsw_recall_sweep.py > logs/hnsw_recall_sweep_1M_10K.log 2>&1 &
```

```bash
echo "PID=$!" && tail -f logs/hnsw_recall_sweep_1M_10K.log
```

如果要让 Plasmod 也真正扫 `PLASMOD_HNSW_EF_SEARCH`，需要允许脚本按每个 ef 重启 Plasmod 并重建索引：

```bash
mkdir -p logs && nohup caffeinate -i /Users/codesoul/miniforge3/bin/python scripts/hnsw_recall_sweep.py --restart-plasmod-per-ef > logs/hnsw_recall_sweep_1M_10K_plasmod_restart.log 2>&1 &
```

输出位置：`results/hnsw_recall_sweep_deep10M_n1000000_q10000_k10_<timestamp>/`，包含 `sweep_points.json/csv` 和 `target_summary.json/csv`。

### 档位 C —— 正式全量 10M（数小时到十几小时）

论文级数据。

```bash
mkdir -p logs && nohup caffeinate -i /Users/codesoul/miniforge3/bin/python scripts/benchmark_all.py --dataset deep10M --index all --queries 10000 --topk 10 > logs/deep10M_full.log 2>&1 &
```

```bash
echo "PID=$!" && tail -f logs/deep10M_full.log
```

### NFCorpus 小数据集（dim=384, 3.6K vec）

快速 sanity check：

```bash
nohup caffeinate -i /Users/codesoul/miniforge3/bin/python scripts/benchmark_all.py --dataset nfcorpus --index all --queries 3237 --topk 10 > logs/nfcorpus_full.log 2>&1 &
```

---

## 2. 推荐流程

1. **先档位 A 冒烟** → 看日志末尾的对比表，确认所有 DB 都正常出数（没有 `FAILED` 行）
2. **再档位 C 全量** → 后台挂着，定期 `tail -f` 看进度
3. **Ctrl+C 退出 `tail -f` 不会停 benchmark**（nohup + `&` 已脱离终端）

---

## 3. 进程管理

```bash
# 看进程
ps aux | grep benchmark_all | grep -v grep

# 看 CPU / 内存
top -pid $(pgrep -f benchmark_all)

# 中止
pkill -f benchmark_all.py
```

---

## 4. 常见问题排障

### Milvus FAILED: `Fail connecting to server on 127.0.0.1:19530`

Milvus 容器挂了（常见 OOM exit 137）。检查并重启：

```bash
docker ps -a --filter name=milvus-standalone --format "{{.Status}}"
docker start milvus-standalone
sleep 15
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:9091/healthz
# 期望 200
```

如果反复 OOM，给 Docker Desktop 调大内存（Settings → Resources → Memory ≥ 8 GB）。

### Plasmod ingest `rc=-3` (kErrBuildFailed)

IVF_PQ 训练样本不够。最小训练集 ~39×nlist，默认 nlist=32 需要至少 ~1250 个向量。
**用 `--index-count >= 10000`** 即可避免。

### `dquote>` 提示卡死

粘贴多行命令时引号/反斜杠在 markdown 中变了字符。
按 **Ctrl+C** 退出，改用上面的**单行版本**。

### `ModuleNotFoundError: No module named 'pymilvus'`

`python` 解析到了 brew python 而非 conda base。

```bash
which python   # 必须是 /Users/codesoul/miniforge3/bin/python
```

若错，要么 `conda activate base && hash -r`，要么直接用绝对路径
`/Users/codesoul/miniforge3/bin/python`（推荐）。

### 服务一键重启

```bash
bash stop_all.sh && bash start_all.sh
```

---

## 5. 结果文件

每次跑完会写到：

- `logs/<run-name>.log`：完整带时间戳的执行日志
- `results/<dataset>_n<N>_q<Q>_k<K>/*.json`：各 DB × 各索引的指标 JSON

汇总表会在日志末尾打印（DB / Index / Build / Batch QPS / Serial QPS / Recall@K / P50 / P95 / P99 / Memory）。

第二层 Dynamic Event Stream and State Visibility 数据放在：

- `data/layer2_dynamic_events/synthetic/`：Synthetic Agent Event Stream
- `data/layer2_dynamic_events/replay_traces/`：Replayable Agent Execution Trace
- `data/layer2_dynamic_events/queries/`：查询 workload
- `data/layer2_dynamic_events/ground_truth/`：可见性、状态、关系和 timing ground truth
- `data/layer2_dynamic_events/manifests/`：数据版本、生成参数和校验信息

第二层实验数据只放在 `plasmod_test_env`，不要放进 `Plasmod` 核心库。

---

## 6. 关键参数

| 参数 | 含义 |
|---|---|
| `--dataset {nfcorpus,deep10M}` | 数据集 |
| `--index-count N` | 索引前 N 条向量（0 = 全部） |
| `--queries N` | 查询数（deep 最多 10000） |
| `--index {flat,ivf_flat,ivf_pq,ivf_sq8,hnsw,all}` | 索引类型 |
| `--db {all,plasmod,qdrant,milvus,lancedb,chromadb}` | 选 DB |
| `--topk N` | 返回 top-K |
| `--sweep-recall` | 扫 recall 0.5-1.0 的 QPS 曲线 |
