# 运行实验

## 启动服务

```bash
cd plasmod_test_env

# 启动所有服务
bash start_all.sh

# 或只启动 Plasmod
bash start_server.sh
```

## 运行 Benchmark

### 快速测试 (nfcorpus)

```bash
# 完整测试 (所有索引，所有数据库)
python3 scripts/benchmark_all.py --dataset nfcorpus --index all --queries 3237 --topk 10

# 只测 Plasmod
python3 scripts/benchmark_all.py --dataset nfcorpus --index all --db plasmod --queries 3237 --topk 10

# 只测单个索引类型
python3 scripts/benchmark_all.py --dataset nfcorpus --index hnsw --db plasmod --queries 3237 --topk 10
```

### 完整压测 (deep10M)

```bash
# 大数据集测试
python3 scripts/benchmark_all.py --dataset deep10M --index all --db plasmod --queries 10000 --topk 10

# 限制索引数量（内存不足时）
python3 scripts/benchmark_all.py --dataset deep10M --index-count 1000000 --index hnsw --db plasmod --queries 10000 --topk 10
```

### 高级功能

```bash
# Recall-QPS 分析
python3 scripts/benchmark_all.py --dataset nfcorpus --index all --sweep-recall

# 特定索引类型的 recall sweep
python3 scripts/benchmark_all.py --dataset nfcorpus --index ivf_pq --db plasmod --sweep-recall
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | nfcorpus | nfcorpus 或 deep10M |
| `--index` | all | flat, ivf_flat, ivf_pq, ivf_sq8, hnsw, all |
| `--db` | all | qdrant, milvus, lancedb, chromadb, plasmod, all |
| `--index-count` | 0 | 索引向量数量 (0=全部) |
| `--queries` | 100 | 查询数量 |
| `--topk` | 10 | 返回前 K 个 |
| `--sweep-recall` | - | 计算不同 recall 级别的 QPS |

## 后台运行

### 方法 1: nohup + caffeinate

```bash
cd plasmod_test_env

# 防止睡眠 + 后台运行 + 日志记录
nohup caffeinate -i python3 scripts/benchmark_all.py \
  --dataset deep10M --index all --db plasmod \
  --queries 10000 --topk 10 > results.log 2>&1 &

# 查看日志
tail -f results.log

# 查看进程
ps aux | grep benchmark
```

### 方法 2: screen 会话

```bash
# 创建会话
screen -S bench
python3 scripts/benchmark_all.py --dataset deep10M --index all --db plasmod

# 断开会话 (保持运行)
# Ctrl+A D

# 恢复会话
screen -r bench

# 查看所有会话
screen -ls
```

## 验证结果

```bash
# 查看结果目录
ls -la results/nfcorpus_n3633_q3237_k10/

# 查看汇总
cat results/nfcorpus_n3633_q3237_k10/summary.json | python3 -m json.tool

# 导出 CSV
python3 scripts/summary_to_csv.py results/nfcorpus_n3633_q3237_k10/summary.json output.csv
```