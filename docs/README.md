# Plasmod 实验环境文档

## 目录结构

| 文档 | 说明 |
|------|------|
| [项目概述](project.md) | 项目结构、核心组件 |
| [环境配置](setup.md) | 重新配置的步骤和依赖 |
| [运行实验](experiments.md) | 如何运行 benchmark |
| [文件管理](files.md) | 数据存储和结果管理 |
| [服务端口](ports.md) | 服务端口速查 |
| [常见问题](faq.md) | FAQ 和故障排除 |

## 快速开始

```bash
cd plasmod_test_env

# 1. 启动服务
bash start_all.sh

# 2. 验证环境
bash verify_env.sh

# 3. 运行实验
python3 scripts/benchmark_all.py --dataset nfcorpus --index all --db plasmod
```

## 项目结构

```
Plasmodexp/
├── Plasmod/           # 核心数据库 (Go + C++)
└── plasmod_test_env/  # 实验环境
    ├── scripts/       # benchmark 脚本
    ├── data/          # 数据集
    ├── models/       # ONNX 模型
    ├── results/      # 实验结果
    └── docs/         # 文档 (本目录)
```