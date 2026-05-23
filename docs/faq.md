# 常见问题

## 服务启动问题

### Q: 服务启动失败

```bash
# 检查端口占用
lsof -i :8080 -i :6333 -i :19530

# 重启服务
bash stop_all.sh
bash start_all.sh
```

### Q: Docker Milvus 无法启动

```bash
# 检查 Docker 状态
docker ps -a | grep milvus

# 查看 Docker 日志
docker logs milvus-etcd
docker logs milvus-minio
docker logs milvus

# 重启 Milvus
cd ../milvus
docker compose down
docker compose up -d
```

### Q: Plasmod 启动报错

```bash
# 检查日志
cat /tmp/plasmod.log

# 检查 C++ 库是否存在
ls -la ../Plasmod/cpp/build/libplasmod_retrieval.dylib

# 重新构建
cd ../Plasmod
make build
```

## 内存问题

### Q: deep10M 测试内存不足

deep10M 需要约 10GB+ 内存。解决方案：

```bash
# 限制索引数量
python3 scripts/benchmark_all.py --dataset deep10M --index-count 1000000 --index hnsw --db plasmod

# 只跑小型索引类型
python3 scripts/benchmark_all.py --dataset deep10M --index ivf_pq --db plasmod
```

### Q: 内存监控

```bash
# 查看进程内存
ps aux | grep plasmod
ps aux | grep qdrant

# 使用 Activity Monitor 查看
open -a "Activity Monitor"
```

## Benchmark 问题

### Q: recall 为 0 或很低

可能原因：
1. 索引未正确构建
2. 查询参数不正确
3. ground truth 计算问题

```bash
# 检查结果
cat results/nfcorpus_n3633_q3237_k10/Plasmod_HNSW.json | python3 -m json.tool

# 查看详细日志
python3 scripts/benchmark_all.py --dataset nfcorpus --index hnsw --db plasmod --verbose
```

### Q: benchmark 运行时间过长

```bash
# 减少查询数量
python3 scripts/benchmark_all.py --dataset deep10M --index all --db plasmod --queries 1000

# 只跑单个索引
python3 scripts/benchmark_all.py --dataset deep10M --index hnsw --db plasmod
```

## 数据问题

### Q: 数据集文件缺失

```bash
# 检查数据集
ls -la data/nfcorpus/
ls -la data/deep/

# 验证文件完整性
file data/nfcorpus/corpus.fbin
file data/deep/base.10M.fbin
```

### Q: 重新生成 nfcorpus embedding

```bash
# 使用 embed_nfcorpus.py 生成
python3 embed_nfcorpus.py
```

## 合盖运行问题

### Q: 如何合盖后继续运行

```bash
# 方法 1: caffeinate 防止睡眠
nohup caffeinate -i python3 scripts/benchmark_all.py ... > results.log 2>&1 &

# 方法 2: 确保系统设置
# 系统设置 → 电池 → 选项 → "防止自动进入睡眠"
```

## 其他问题

### Q: 如何清理所有数据重新开始

```bash
# 停止所有服务
bash stop_all.sh

# 清理数据目录
rm -rf .andb_data/
rm -rf lancedb_data/
rm -rf chromadb_data*/
rm -rf results/*

# 重启服务
bash start_all.sh
```

### Q: 查看服务健康状态

```bash
curl http://127.0.0.1:8080/healthz     # Plasmod
curl http://127.0.0.1:6333/             # Qdrant
docker ps                                # Milvus
curl http://127.0.0.1:9000/minio/health/live  # MinIO
```