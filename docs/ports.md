# 服务端口速查

## 服务端口表

| 服务 | 地址 | 端口 | 协议/用途 | 检查命令 |
|------|------|------|-----------|----------|
| **Plasmod** | 127.0.0.1 | 8080 | HTTP API | `curl http://127.0.0.1:8080/healthz` |
| **Qdrant** | 127.0.0.1 | 6333 | HTTP REST | `curl http://127.0.0.1:6333/` |
| **Qdrant** | 127.0.0.1 | 6334 | gRPC | - |
| **Milvus** | 127.0.0.1 | 19530 | gRPC | `docker ps \| grep milvus` |
| **Milvus** | 127.0.0.1 | 9091 | Console | `docker ps \| grep milvus` |
| **Milvus-etcd** | 127.0.0.1 | 23780 | Docker 内部 | Docker 内部 |
| **Milvus-MinIO** | 127.0.0.1 | 9002 | Docker 映射端口 | Docker 内部 |
| **MinIO** | 127.0.0.1 | 9000 | S3 API | `curl http://127.0.0.1:9000/minio/health/live` |
| **MinIO** | 127.0.0.1 | 9001 | Console | `open http://127.0.0.1:9001` |
| **LanceDB** | - | - | 嵌入式 | - |
| **ChromaDB** | - | - | 嵌入式 | - |

## Docker 服务

Milvus 使用 docker-compose 启动，包含以下服务：

```
milvus-etcd:     23780 (映射到 2379)
milvus-minio:    9002 (映射到 9000)
milvus:          19530, 9091
```

## 检查所有服务

```bash
# 启动脚本中的检查
bash start_all.sh

# 或手动检查
for port in 8080 6333 6334 19530 9091 9000 9001; do
  curl -s --connect-timeout 2 http://127.0.0.1:$port/ > /dev/null 2>&1 \
    && echo "Port $port: UP" || echo "Port $port: DOWN"
done
```

## 端口占用检查

```bash
# 查看端口占用
lsof -i :8080
lsof -i :6333
lsof -i :19530

# 查看所有 Plasmod 相关端口
lsof -i -P | grep -E "8080|6333|6334|19530|9091|9000|9001"
```

## 停止服务

```bash
# 停止所有服务
bash stop_all.sh

# 停止单个服务
kill $(lsof -t -i :8080)   # Plasmod
kill $(lsof -t -i :6333)   # Qdrant
docker stop milvus          # Milvus
```