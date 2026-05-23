# 环境配置

## 重新配置环境的步骤

### 1. 必需依赖

```bash
# 1. Go 1.25+
brew install go

# 2. Python 3.9+
brew install python3

# 3. Docker (用于 Milvus)
brew install --cask docker

# 4. 系统库
brew install libomp abseil googletest onnxruntime
```

### 2. Python 包

```bash
pip3 install qdrant-client pymilvus lancedb chromadb onnxruntime numpy
```

### 3. 二进制文件

实验环境依赖以下外部二进制文件（需手动下载）：

| 二进制 | 用途 | 位置 |
|--------|------|------|
| MinIO | S3 存储 | `../minio/minio` |
| Qdrant | 向量数据库 | `../qdrant/bin/qdrant` |
| Milvus | 向量数据库 | Docker (自动下载) |
| Plasmod | 主测试对象 | `../Plasmod/bin/plasmod` |

#### 下载 MinIO

```bash
mkdir -p minio
cd minio
curl -O https://dl.min.io/server/minio/release/darwin-arm64/minio
chmod +x minio
```

#### 下载 Qdrant

```bash
mkdir -p qdrant
cd qdrant
curl -L https://github.com/qdrant/qdrant/releases/latest/download/qdrant-macos-aarch64.zip -o qdrant.zip
unzip qdrant.zip
mv qdrant-macos-aarch64/* .
rm -rf qdrant-macos-aarch64 qdrant.zip
```

### 4. ONNX 模型

```bash
# 位置: plasmod_test_env/models/all-MiniLM-L6-v2.onnx
# 如果缺失，从 HuggingFace 下载
```

### 5. 数据集

```bash
cd plasmod_test_env/data/

# nfcorpus (已有，验证)
ls -la nfcorpus/

# deep10M - 需要手动下载 (3.8GB)
# 下载链接: 从项目存储或公共 S3 获取
# 文件:
#   - deep/base.10M.fbin
#   - deep/query.public.10K.fbin
#   - deep/groundtruth.public.10K.ibin
```

### 6. 构建 Plasmod

```bash
cd ../Plasmod

# 构建 C++ 库
cmake -S cpp -B cpp/build && cmake --build cpp/build --parallel

# 构建 Go 服务
make build
```

### 7. 启动服务

```bash
cd ../plasmod_test_env

# 启动所有服务 (MinIO, Milvus, Qdrant, Plasmod)
bash start_all.sh

# 验证环境
bash verify_env.sh

# 查看服务状态
curl http://127.0.0.1:8080/healthz   # Plasmod
curl http://127.0.0.1:6333/            # Qdrant
```