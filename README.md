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

### 公开实验数据

动态 agent event workload 和 replay 输入发布在 [CodeSoulco/plasmod-agent-event-data-release](https://huggingface.co/datasets/CodeSoulco/plasmod-agent-event-data-release)。该发布包含重新收集并规范化的两组 agent trajectory：一组覆盖 MetaGPT x SWE-bench、LangGraph x BIRD-SQL/tau-bench、LlamaIndex x GAIA、PydanticAI x MINT 和 AutoGen x AgentBench；另一组覆盖 GPT Researcher x MS MARCO/GAIA、OpenManus x ToolBench 和 AutoGen x MS MARCO/GAIA。数据集卡提供各框架与 benchmark 的官方引用链接、采集规模和字段定义。

将其下载到本仓库预期的位置并解压 shard archive：

```bash
hf download CodeSoulco/plasmod-agent-event-data-release \
  --repo-type dataset \
  --local-dir data/layer2_dynamic_events
zstd -d --stdout data/layer2_dynamic_events/traces_collected.tar.zst \
  | tar -xf - -C data/layer2_dynamic_events
```

该数据集提供经过脱敏和标准化的 JSONL 数据资产；本仓库提供用于加载、回放、压力测试和结果整理的实验代码。上游 benchmark 的原始数据、许可证和访问条件不随本仓库分发。下载后目录包含：

- `data/layer2_dynamic_events/traces_collected/`：按 source trajectory 分组的规范化 agent event shards
- `data/layer2_dynamic_events/events.jsonl`：有序 replay trace，包含 research、tool-use 与 multi-agent 协作场景的事件记录

第二层实验数据只放在 `plasmod_test_env`，不要放进 `Plasmod` 核心库。

第二层实验脚本：

```bash
python3 scripts/layer2_dynamic_event_benchmark.py analyze
```

### 当前第二层实验方案：Dynamic Event Stream and State Visibility

第二层实验只在 `plasmod_test_env` 中实现和运行；`Plasmod` 核心库不放实验逻辑。实验输入来自真实 agent runtime 记录后整理的数据：

- `data/layer2_dynamic_events/traces_collected/`：Synthetic Agent Event Stream，用于 Table 4/5/6 的动态写入、可见性和 freshness 压测
- `data/layer2_dynamic_events/events.jsonl`：Replayable Agent Execution Trace，用于 Table 7/8 的 replay、recovery 和 state correctness 验证

当前 Table 5 的正式口径是：同一批 agent event object，在不同 write rate 下并发触发写入、query、visibility probe，输出 `query_p50_ms`、`query_p95_ms`、`write_to_visible_p95_ms`、`materialization_lag_p95_ms`、`stale_result_rate` 等指标。默认 `--query-qps 0` 表示查询不额外限速，CSV 中的 `query_qps` 是实际完成查询吞吐；如果论文表格需要固定查询 workload，应显式设置 `--query-qps <N>` 并在表注中说明。

当前 Table 6 的正式口径是：固定 write rate 和 query workload，对比 Plasmod 的 `Strict / Bounded Staleness / Eventual` 三种 visibility mode，以及 Milvus 的 best-effort baseline。Table 6 结果应写入单独 sheet，不要混入 Table 5 的 Sheet1。

推荐运行参数：

- embedding: `--embedding-provider hash`，复用确定性预计算/哈希 embedding，避免把 embedding 计算开销混入数据库性能
- Milvus index: `--milvus-index-type FLAT`
- Milvus visibility: `--milvus-visibility-policy deferred`
- large payload: `--milvus-payload-json-bytes 0`，大 payload 外部化，避免 Milvus varchar 长度限制影响实验
- visibility probes: `--visibility-probe-limit 5000`
- full data: `--events-per-rate 0 --query-limit 0`

### Milvus-only Table 5 启动方案

只跑 Milvus baseline 时，不需要启动 Plasmod、Qdrant 或本地 MinIO；只需要 Docker Desktop 和 `../milvus/docker-compose.yml` 里的 Milvus 三容器：`milvus-etcd`、`milvus-minio`、`milvus-standalone`。

启动 Milvus：

```bash
cd /Users/erwin/Downloads/codespace/Plasmodexp/milvus
docker compose up -d
curl http://127.0.0.1:9091/healthz
```

期望输出 `OK`。若 Python SDK 也要确认：

```bash
cd /Users/erwin/Downloads/codespace/Plasmodexp/plasmod_test_env
python3 - <<'PY'
from pymilvus import MilvusClient
c = MilvusClient(uri="http://127.0.0.1:19530")
print(c.list_collections())
PY
```

后台运行 Milvus Table 5 的 10 / 100 / 500 events/s：

```bash
cd /Users/erwin/Downloads/codespace/Plasmodexp/plasmod_test_env
RUN_ID=table5_milvus_rates_$(date +%Y%m%d_%H%M%S)
RUN_DIR="results/layer2_dynamic_events/${RUN_ID}"
mkdir -p "${RUN_DIR}"
screen -dmS layer2_milvus_table5 bash -lc "cd /Users/erwin/Downloads/codespace/Plasmodexp/plasmod_test_env && caffeinate -dimsu python3 scripts/layer2_dynamic_event_benchmark.py run --run-id ${RUN_ID} --tables 5 --systems milvus --embedding-provider hash --write-rates 500 100 10 --events-per-rate 0 --query-limit 0 --query-qps 0 --workers 32 --visible-timeout-ms 30000 --visible-poll-ms 25 --visibility-probe-limit 5000 --http-timeout 120 --milvus-index-type FLAT --milvus-visibility-policy deferred --milvus-payload-json-bytes 0 --reset-between-runs >> ${RUN_DIR}/run.log 2>&1"
echo "${RUN_DIR}"
```

查看进度：

```bash
tail -f /Users/erwin/Downloads/codespace/Plasmodexp/plasmod_test_env/results/layer2_dynamic_events/<run-id>/run.log
```

查看已完成行：

```bash
cat /Users/erwin/Downloads/codespace/Plasmodexp/plasmod_test_env/results/layer2_dynamic_events/<run-id>/table5_freshness_under_write_load.csv
```

停止当前 Milvus Table 5 实验：

```bash
screen -S layer2_milvus_table5 -X quit 2>/dev/null || true
pkill -f "layer2_dynamic_event_benchmark.py run --run-id table5_milvus" 2>/dev/null || true
pkill -f "caffeinate -dimsu python3 scripts/layer2_dynamic_event_benchmark.py" 2>/dev/null || true
```

停止 Milvus 服务：

```bash
cd /Users/erwin/Downloads/codespace/Plasmodexp/milvus
docker compose down
```

如果 `milvus-standalone` 秒退且日志反复出现 `find no available mixcoord`，优先认为是上次强停后 Milvus metadata / rocksmq 状态不一致。不要直接删除旧数据，先备份 volume 再重启：

```bash
cd /Users/erwin/Downloads/codespace/Plasmodexp/milvus
docker compose down
mv volumes "volumes_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p volumes
docker compose up -d
curl http://127.0.0.1:9091/healthz
```

小规模 Milvus baseline 冒烟：

```bash
python3 scripts/layer2_dynamic_event_benchmark.py run --tables 4 8 --systems milvus --events-per-type 10 --table8-updates 50 --replay-events 1000 --run-id layer2_smoke_milvus
```

跑 Table 4-8（需要先 `bash start_all.sh` 启动 Milvus 和 Plasmod）：

```bash
python3 scripts/layer2_dynamic_event_benchmark.py run --tables all --systems milvus plasmod --reset-between-runs --run-id layer2_full
```

输出位置：`results/layer2_dynamic_events/<run-id>/`，包含各表 CSV、`summary.json` 和 `run_metadata.json`。

---

## 6. Agent-native Database 消融与完整数据库实验

这组实验只由 `plasmod_test_env` 编排。核心库只提供可部署、可验证的 runtime capability，不包含 variant 名称、实验循环或指标聚合。每个 variant 都使用同一个当前 Plasmod 二进制，并通过环境配置关闭真实数据路径；runner 启动后会读取 `/v1/admin/capabilities`，配置未生效会立即停止。

### 6.1 输入与完整服务栈

输入同时覆盖：

- `data/layer2_dynamic_events/events.jsonl`：可重放 agent execution trace，验证恢复、状态、artifact、relation 和 provenance。
- `data/layer2_dynamic_events/traces_collected/*.jsonl`：记录并规范化的动态 agent event stream，验证写入、物化、治理、evidence 和 tiering。
- `results/layer2_dynamic_events/embedding_cache.sqlite3`：确定性预计算 embedding cache；数据库实验复用向量，不重复计入 embedding 计算成本。

runner 会启动完整的真实服务路径：当前 C++ retrieval build、Go Plasmod server、文件 WAL、Badger canonical store、hot/warm retrieval plane，以及本地 MinIO 提供的真实 S3 cold store。每个 variant 使用独立 data directory 和 S3 prefix，结束后停止 Plasmod 与 runner 自己启动的 MinIO。

### 6.2 Variant 与指标

| 分组 | Variant |
|---|---|
| WAL / Event Log | Full Plasmod, No-WAL, In-memory WAL, File WAL, WAL without replay, Replay without index rebuild |
| Materialization | Full Plasmod, No-materialization, Memory-only, No-agent-state, No-artifact, No-edge, No-object-version |
| Evidence / Provenance | Full Plasmod, No-evidence, No-provenance, No-edge-expansion, One-hop only, No-proof-trace, Vector-only |
| Access / Scope / Governance | Full Plasmod, No-access-policy, Metadata-filter-only, No-share-contract, No-quarantine, No-delete-propagation |
| Hot / Warm / Cold | Full Tiering, No-hot-cache, Warm-only, No-cold, No-promotion, Hot-size-64/512/2000 |

五个分组中的 Full 是同一条逻辑基线。runner 只启动一次
`shared/Full Plasmod`、写入一次公共 workload，并在该生命周期中依次采集五组 Full
指标；分组 CSV 和主表仍各保留自己的 Full 对照行。其余 29 个消融/控制变量独立运行，
因此输出保持 34 个逻辑行，实际只有 30 次物理 variant 运行。

| 结果文件 | 完整指标 |
|---|---|
| `wal_event_log_ablation.csv` | event log size, recovered objects/relations/latest state, recovery time, replay throughput, query availability, lost events, duplicates |
| `materialization_ablation.csv` | write QPS/p95, write-to-visible p95, materialization lag p95, object coverage, latest-state hit, artifact accuracy, relation recovery, stale rate |
| `evidence_provenance_ablation.csv` | query p95, evidence assembly p95, provenance completeness, edge recall, proof completeness, citation/evidence correctness, stale evidence |
| `governance_ablation.csv` | private leakage, authorized/unauthorized hit, delete visibility delay, quarantine exclusion, policy overhead |
| `tiered_storage_ablation.csv` | query p50/p95/p99, hot/warm/cold hit rate, promotion p95, RSS memory, stale rate |
| `full_database_baseline.csv` | 五组 Full variant 的全部指标，合并为一行完整数据库基线 |
| `ablation_master_table.csv` | 34 个逻辑 Full / `w/o` 对照行的统一总表：五个 Full 行共享同一次实测基线；非适用项显式为 `N/A (not applicable)` |
| `ablation_master_style.json` | 总表列组的颜色与语义清单，供 Excel/论文表格导出器稳定复用 |

指标来自真实 HTTP ACK、canonical state、replay API、query diagnostics 和 S3 tier counters。`Object Visibility Coverage` 按 baseline 中 event/memory/state/artifact/edge/version 的对象数量计算；`Stale Rate` 使用 tier 查询后 canonical target 是否可见计算，不把 ANN recall 当作 freshness；governance 对象使用独立 session，避免 conflict lifecycle 干扰 ACL 测量。

#### 6.2.1 最大交集公共口径

`ablation_master_table.csv` 使用 `agent-native-common-v1` 参数集。每个变体在模块专属 probe 之前都执行完全相同的 recorded agent-native workload，包括 governance 组；因此公共性能列不是由不同实验表拼接或估算得到的。

| 公共参数 | 固定规则 |
|---|---|
| Event input | 同一有序数据源前缀；`--event-limit 0` 时为全部输入 |
| Query input | 同一 query sample 选择规则和 `--query-limit` |
| TopK | 20 |
| Embedding | 384 维确定性缓存向量，不计入数据库请求延迟 |
| Write consistency | `strict` |
| Query consistency | `eventual` |
| Canonical storage | Badger disk |
| Cold storage | 真实本地 MinIO S3；即使某个 tier variant 关闭 cold capability，服务拓扑仍保持一致 |
| Server path | 同一个 Go server、C++ retrieval build 和 HTTP API |

每一行统一计算 19 个公共数值字段：event/query/stale-check 数量，TopK，embedding 维度，write QPS 与 p50/p95/p99，write-to-visible p50/p95，materialization lag p95，query QPS 与 p50/p95/p99，RSS memory，object visibility coverage 和 target stale rate。WAL、materialization、evidence、governance、tier 的特殊指标保留在各自带前缀的列组中；其他模块行写 `N/A (not applicable)`，不使用空白或伪造的 0。

`Comparison Label` 明确给出论文中的 `Full`、`w/o WAL / Event Log`、`w/o Replay`、`w/o Canonical Materialization`、`w/o Agent State`、`w/o Evidence Assembly`、`w/o Access Policy`、`w/o Hot Cache` 等名称；控制变量行使用 `File WAL control`、`Hot Cache = 64/512/2000` 等标签，不错误标成 `w/o`。

### 6.3 Smoke、定量和全量运行

Smoke 使用 8 条记录事件和最多 3 个 query context，验证全部 34 个逻辑对照行（30 次物理运行），并生成 1 行完整数据库聚合基线：

```bash
cd /Users/erwin/Downloads/codespace/Plasmodexp/plasmod_test_env
bash scripts/run_agent_native_ablation.sh smoke
```

默认定量运行使用每个通用 variant 1000 条事件和 100 个最新 agent/session query context：

```bash
bash scripts/run_agent_native_ablation.sh run
```

全量运行读取 `events.jsonl` 和全部 `traces_collected/*.jsonl`，每个通用 variant 都使用完整输入；query context 默认保持 100，防止查询样本量随事件总量失控：

```bash
bash scripts/run_agent_native_ablation.sh full
```

直接调用 runner 可固定 run id、端口和 query 数：

```bash
python3 scripts/agent_native_ablation_benchmark.py run \
  --run-id agent_native_ablation_full_$(date +%Y%m%d_%H%M%S) \
  --event-limit 0 \
  --query-limit 100 \
  --port 18080
```

### 6.4 后台运行、进度与恢复

```bash
cd /Users/erwin/Downloads/codespace/Plasmodexp/plasmod_test_env
RUN_ID=agent_native_ablation_full_$(date +%Y%m%d_%H%M%S)
mkdir -p "logs/${RUN_ID}"
nohup caffeinate -dimsu python3 scripts/agent_native_ablation_benchmark.py run \
  --run-id "${RUN_ID}" --event-limit 0 --query-limit 100 --port 18080 \
  > "logs/${RUN_ID}/run.log" 2>&1 &
echo "$!" > "logs/${RUN_ID}/runner.pid"
tail -f "logs/${RUN_ID}/run.log"
```

关闭 `tail` 不会停止后台实验。已完整写出的分组 CSV 可复用；同一 run id 修复服务后续跑：

```bash
python3 scripts/agent_native_ablation_benchmark.py run \
  --run-id "${RUN_ID}" --event-limit 0 --query-limit 100 --port 18080 --resume
```

结果位于 `results/agent_native_ablation/<run-id>/`。只有分组 CSV 和 34 行总表全部非空、19 个公共指标全部为有限数值、capability 回读一致、服务日志没有 panic/fatal/S3 错误时才生成 `COMPLETE` 和 `summary.json`；任何 HTTP、服务、指标或日志错误都会停止并生成 `FAILED`。共享 Full 的 capability、公共指标、五组 Full probe 与恢复清单位于 `variants/shared-full-plasmod/`，其余 29 个 variant 保持各自目录。`summary.json` 使用 `shared_full_runs=1` 和 `physical_variant_runs=30` 记录实际执行次数。

最近通过完整 smoke 的结果目录：

```text
results/agent_native_ablation/agent_native_ablation_smoke_20260722_v8/
```

### 6.5 迁移到 Linux 服务器运行

#### 6.5.1 目录、资源与依赖

两个仓库必须保持同级目录，因为 runner 通过 `plasmod_test_env/../Plasmod` 定位核心库：

```text
/srv/Plasmodexp/
├── Plasmod/              # dev branch
└── plasmod_test_env/     # main branch
```

CPU 版本不需要 GPU、Milvus、Docker 或额外 Python package。正式运行条件如下；版本条件来自当前 `go.mod`、CMake 和 runner 代码，而不是通用建议。

| 类别 | 必需条件 | 不满足时的处理 |
|---|---|---|
| OS/architecture | Linux x86_64 或 aarch64；glibc 环境 | 不要复用 macOS binary；在服务器重编译 |
| Go | 1.25.x 或更高，且满足 `Plasmod/go.mod` | 升级 Go 后重新 `make build` |
| Python | 3.9+；runner 只用标准库，无需 `pip install` | 升级系统 Python |
| C++ build | CMake 3.20+、C++17 compiler、GNU Make | 安装 build toolchain 后重建 `.so` |
| Runtime tools | Git、curl、`minio` server、`mc` client、`ps` | 安装对应 executable 并加入 `PATH` |
| 数据 | `events.jsonl`、至少一个 `traces_collected/*.jsonl` | 从数据准备机单独同步，Git 不包含它们 |
| Embedding | 推荐同步 `embedding_cache.sqlite3` | 缺失时可以运行，但会重新生成 cache |
| CPU | 推荐至少 8 个逻辑核 | 可以 smoke；正式数值需注明硬件限制 |
| Memory | 推荐至少 32 GB | 先 smoke；监控 OOM/swap 后再决定是否全量 |
| Storage | 本地 SSD；full 默认至少 250 GB 可用，建议准备 500 GB | 清理/扩容；不要把 Badger data dir 放在 NFS 上 |
| File limit | full 推荐 `ulimit -n 65536` | 提高当前 shell 或 systemd 的 `LimitNOFILE` |
| Ports | `127.0.0.1:18080`、`:9000`、`:9001` 可用 | 停止冲突进程或仅为 Plasmod 改 `--port` |
| Execution user | 对仓库、结果目录、cache 和 MinIO data dir 可读写 | 修复 owner/permission，不使用混合用户运行 |

完整性能对比还需要满足实验条件：同一台机器、同一 commit、同一数据副本和参数；使用交流电或稳定的服务器供电；关闭 sleep；运行期间不要并行编译、跑其他数据库或其他实验；避免 CPU governor、虚拟机资源和磁盘配额在 variant 之间变化。runner 串行运行所有 variant，因此不要另外启动第二个 runner 来“加速”，否则 CPU、I/O 和 MinIO 会相互干扰，数值不再公平。

Ubuntu/Debian 基础构建依赖可先安装：

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git curl rsync python3 ca-certificates
python3 --version # 必须 >= 3.9
go version        # 必须满足 Plasmod/go.mod 当前声明的 Go 版本
cmake --version   # 必须 >= 3.20
minio --version
mc --version
```

Go、MinIO 和 `mc` 如果不在发行版仓库中，应使用对应项目的官方 binary/package 安装。不要把 macOS 下的 `bin/plasmod` 或 `.dylib` 复制到 Linux；必须在服务器重新生成 `.so` 和 Go binary。

全量输入当前约 3.6 GB、36k 文件。共享 Full 使用一个 Badger data directory 和 S3 prefix，其余 29 个物理 variant 各自隔离；完整运行的实际占用仍会显著大于原始数据。执行前用 `df -h` 检查专用磁盘。预检默认把 full 的 250 GB 作为最低门槛，并建议根据 smoke/中等运行后的单 variant 占用扩容到 500 GB 以上。

#### 6.5.2 启动责任与服务拓扑

不要把“安装组件”和“手工启动服务”混在一起。消融 runner 是唯一的生命周期管理者：

| 组件 | 是否必须安装/构建 | 是否手工提前启动 | 实际启动者与生命周期 |
|---|---|---|---|
| C++ retrieval `.so` | 必须在 Linux 构建一次，每次核心代码变化后重建 | 否；它不是独立服务 | 由 Go binary 通过 CGO 动态加载 |
| Go `bin/plasmod` | 必须存在；默认 runner 每次先执行 `make build` | **禁止手工启动** | runner 为每个 variant 启动一个，健康检查后测试，再停止 |
| Badger canonical store | 包含在 Plasmod 中 | 否 | 每个 variant 在独立 `variants/<variant>/data` 内打开 |
| WAL、materializer、evidence、governance、tier workers | 包含在 Plasmod 中 | 否 | 随当前 variant 的 Plasmod 进程启动，profile 由环境变量选择 |
| MinIO server | `minio` executable 必须在 `PATH` | 通常不需要 | 9000 无健康实例时由 runner 启动；结束时只停止自己启动的实例 |
| MinIO bucket | `mc` executable 必须在 `PATH` | 否 | runner 创建/复用 `plasmod-experiments` |
| Embedding service | 不需要 | 否 | runner 从 SQLite cache 读取确定性 384 维向量；缺失项在本地计算并缓存 |
| gRPC service | 不需要 | 否 | runner 显式设置 `PLASMOD_GRPC_ENABLED=0`，只走真实 HTTP API |
| Milvus/Docker/GPU | 本消融实验不需要 | **不要启动** | 不参与该表，不应占用实验资源 |

runner 使用的固定拓扑是：

```text
agent_native_ablation_benchmark.py
  -> MinIO 127.0.0.1:9000 (S3 cold tier)
  -> Plasmod 127.0.0.1:18080 (one variant at a time)
       -> file/memory/disabled WAL profile
       -> Badger canonical object graph
       -> C++ retrieval shared library
       -> materialization/evidence/governance/tiering profile
```

因此不要运行核心库的 `scripts/run_full.sh`、`make dev`、`docker compose up`、旧 `start_all.sh` 或单独的 MinIO+Plasmod 启动脚本。唯一例外是复用一套已经健康的 MinIO；它必须监听 `127.0.0.1:9000`，使用 `minioadmin/minioadmin`，并允许 `mc` 创建 `plasmod-experiments` bucket。

#### 6.5.3 Clone 并锁定分支

```bash
sudo mkdir -p /srv/Plasmodexp
sudo chown "$(id -u):$(id -g)" /srv/Plasmodexp
cd /srv/Plasmodexp

git clone --branch dev https://github.com/CodeSoul-co/Plasmod.git
git clone --branch main https://github.com/erwinmsmith/plasmod_test_env.git

git -C Plasmod pull --ff-only origin dev
git -C plasmod_test_env pull --ff-only origin main
git -C Plasmod rev-parse HEAD
git -C plasmod_test_env rev-parse HEAD
```

私有仓库可将 HTTPS URL 换成已配置 deploy key 的 SSH URL。正式运行期间不要再次 `git pull`；`run_metadata.json` 会记录两个仓库的 commit。

#### 6.5.4 单独传输数据和 embedding cache

实验数据和 embedding cache 被 `.gitignore` 排除，`git clone` 后不会出现，必须从当前机器单独复制：

```bash
# 先创建服务器目标目录。
ssh USER@SERVER 'mkdir -p \
  /srv/Plasmodexp/plasmod_test_env/data/layer2_dynamic_events \
  /srv/Plasmodexp/plasmod_test_env/results/layer2_dynamic_events'

# 在当前机器执行；替换 SERVER 和用户名。
cd /Users/erwin/Downloads/codespace/Plasmodexp/plasmod_test_env

rsync -a --info=progress2 --checksum \
  data/layer2_dynamic_events/ \
  USER@SERVER:/srv/Plasmodexp/plasmod_test_env/data/layer2_dynamic_events/

rsync -a --info=progress2 --checksum \
  results/layer2_dynamic_events/embedding_cache.sqlite3 \
  USER@SERVER:/srv/Plasmodexp/plasmod_test_env/results/layer2_dynamic_events/embedding_cache.sqlite3
```

两端至少核对文件数、容量和关键文件 checksum：

```bash
cd /srv/Plasmodexp/plasmod_test_env
find data/layer2_dynamic_events -type f | wc -l
du -sh data/layer2_dynamic_events results/layer2_dynamic_events/embedding_cache.sqlite3
sha256sum data/layer2_dynamic_events/events.jsonl \
  results/layer2_dynamic_events/embedding_cache.sqlite3
```

macOS 本地对应命令是 `shasum -a 256 <file>`。如果不复制 cache，runner 会重新计算并写入 cache，但这会增加首次运行准备时间，因此正式服务器应复用现有文件。

#### 6.5.5 在服务器重新构建 Plasmod

```bash
cd /srv/Plasmodexp/Plasmod

# CPU/HNSW C++ retrieval shared library。
bash scripts/build_cpp.sh

# Makefile 检测 cpp/build/libplasmod_retrieval.so 后，自动使用 retrieval build tag。
make build

test -f cpp/build/libplasmod_retrieval.so
test -x bin/plasmod
ldd bin/plasmod | grep -E 'plasmod_retrieval|not found' || true
go test ./src/...
```

runner 在 Linux 上设置 `LD_LIBRARY_PATH=cpp/build:cpp/build/vendor`，在 macOS 上设置对应的 `DYLD_LIBRARY_PATH`。如果 `ldd` 出现 `not found`，不要开始实验，应先修复 shared-library 路径或重新执行 `make build`。

#### 6.5.6 自动预检

代码、数据和 Linux 构建完成后，先运行仓库内的 preflight。`smoke` 检查 10 GB 可用空间，`full` 默认要求 250 GB；可通过 `PLASMOD_ABLATION_MIN_FREE_GB` 提高门槛，但正式实验不建议降低。

```bash
cd /srv/Plasmodexp/plasmod_test_env

# smoke 前检查。
bash scripts/preflight_agent_native_ablation.sh smoke --port 18080

# full 前重新检查；正式运行以这次结果为准。
ulimit -n 65536
bash scripts/preflight_agent_native_ablation.sh full --port 18080
```

preflight 会核对：软件命令和最低版本、核心 `dev`/实验 `main` 分支、tracked working tree、Linux shared library 和 Go binary、动态链接、两类输入、SQLite cache、目录权限、磁盘、CPU、内存、file limit、Plasmod/MinIO 端口、MinIO 凭据以及其他 ablation runner。任何 `[FAIL]` 都必须先解决；`[WARN]` 不会阻止启动，但正式论文运行前需要确认其影响。

#### 6.5.7 端口和 MinIO 行为

runner 会执行以下工作：

1. 检查 `127.0.0.1:9000/minio/health/live`；
2. 如果没有现有 MinIO，则调用 `minio server`，数据放到当前 run 目录的 `minio-data/`；
3. 使用 `mc` 创建 `plasmod-experiments` bucket；
4. 每次只启动一个 Plasmod variant，统一监听指定的 HTTP port；
5. 正常或失败退出时停止自己启动的 Plasmod 和 MinIO。

因此消融实验**不需要**提前运行 `start_all.sh`，也不要手工启动另一个 Plasmod。运行前检查端口：

```bash
ss -ltnp | grep -E ':18080|:9000|:9001' || true
curl -fsS http://127.0.0.1:9000/minio/health/live || true
```

如果 `18080` 被占用，传入例如 `--port 18081`。如果复用已有 MinIO，它必须允许 `minioadmin/minioadmin` 创建实验 bucket；否则停止它，让 runner 启动隔离实例。不要通过公网暴露 9000、9001 或实验 HTTP port。

#### 6.5.8 唯一正确的启动顺序

从空服务器开始，完整顺序固定为：

1. 安装 build/runtime tools 和 `minio`/`mc`，但不手工启动 Plasmod；
2. clone `Plasmod/dev` 和 `plasmod_test_env/main`，记录 commit；
3. 传输两类 JSONL 数据和 embedding cache；
4. 在 Linux 先构建 C++ `.so`，再构建 Go `bin/plasmod`；
5. 执行 `preflight ... smoke`，修复所有 `[FAIL]`；
6. 运行 smoke，确认 1 次共享 Full、29 次独立消融运行、34 行结果表和 `COMPLETE`；
7. 执行 `preflight ... full`，确认磁盘、端口和后台进程仍满足条件；
8. 使用固定 `RUN_ID` 启动一个全量 runner；不要再启动任何数据库进程；
9. 用 `tail -F` 观察，不修改代码、数据、commit、端口或参数；
10. 仅在失败原因修复后用同一 `RUN_ID` 和完全相同参数 `--resume`。

#### 6.5.9 先执行服务器 smoke

```bash
cd /srv/Plasmodexp/plasmod_test_env
bash scripts/run_agent_native_ablation.sh smoke --port 18080
```

结束后检查最新目录：

```bash
RUN_DIR="$(find results/agent_native_ablation -maxdepth 1 -type d \
  -name 'agent_native_ablation_smoke_*' | sort | tail -n 1)"
test -f "${RUN_DIR}/COMPLETE"
test ! -f "${RUN_DIR}/FAILED"
cat "${RUN_DIR}/summary.json"
wc -l "${RUN_DIR}/ablation_master_table.csv"
```

预期主表为表头加 34 个逻辑对照行，即 `wc -l` 为 35；`summary.json` 应同时报告 `shared_full_runs=1` 和 `physical_variant_runs=30`。smoke 未通过时不要启动全量任务，应先检查 `FAILED`、`variants/*/server.log` 和 `minio.log`。

#### 6.5.10 Linux 后台全量运行

Linux 服务器不需要 macOS 的 `caffeinate`。使用固定 `RUN_ID`，以便失败后复用已完成分组：

```bash
cd /srv/Plasmodexp/plasmod_test_env
RUN_ID="agent_native_ablation_full_$(date -u +%Y%m%d_%H%M%S)"
RUN_DIR="results/agent_native_ablation/${RUN_ID}"
LOG_DIR="logs/${RUN_ID}"
mkdir -p "${LOG_DIR}"

nohup python3 scripts/agent_native_ablation_benchmark.py run \
  --run-id "${RUN_ID}" \
  --event-limit 0 \
  --query-limit 100 \
  --port 18080 \
  > "${LOG_DIR}/run.log" 2>&1 &

echo "$!" > "${LOG_DIR}/runner.pid"
printf 'RUN_ID=%s\nPID=%s\n' "${RUN_ID}" "$(cat "${LOG_DIR}/runner.pid")"
```

SSH 断开后 `nohup` 进程会继续运行。完成依赖、代码和数据下载后，实验不需要外网，但本机 loopback、Plasmod 与 MinIO 之间的连接必须保持正常。

#### 6.5.11 查看进度、恢复和取回结果

```bash
cd /srv/Plasmodexp/plasmod_test_env
tail -F "logs/${RUN_ID}/run.log"
ps -fp "$(cat "logs/${RUN_ID}/runner.pid")" || true
find "results/agent_native_ablation/${RUN_ID}" -maxdepth 1 -name '*.csv' -print
```

runner 在每个分组完成后立即写对应 CSV；只有五个分组、`full_database_baseline.csv` 和 `ablation_master_table.csv` 全部通过校验后才写 `COMPLETE`。如果服务器重启或任务失败，修复服务/磁盘问题后使用完全相同的 run id 和参数：

```bash
nohup python3 scripts/agent_native_ablation_benchmark.py run \
  --run-id "${RUN_ID}" \
  --event-limit 0 \
  --query-limit 100 \
  --port 18080 \
  --resume \
  >> "logs/${RUN_ID}/run.log" 2>&1 &
echo "$!" > "logs/${RUN_ID}/runner.pid"
```

`--resume` 只复用已经完整写出的分组 CSV；失败中的分组会从该组重新运行，不会把半行结果拼入正式表。任务完成后先取回轻量结果：

```bash
RUN_PATH="results/agent_native_ablation/${RUN_ID}"
(
  cd "${RUN_PATH}"
  tar -czf "${HOME}/${RUN_ID}_tables.tar.gz" \
    -- *.csv summary.json run_metadata.json ablation_master_style.json COMPLETE
)
```

如果需要完整审计，再单独同步 `variants/*/capabilities.json`、`measurements.json`、`common_metrics.json` 和 `server.log`。不要默认打包整个 `variants/*/data` 与 `minio-data`，它们可能达到数百 GB。

#### 6.5.12 启动命令与参数语义

| 启动方式/参数 | 实际含义 | 正式运行约束 |
|---|---|---|
| `run_agent_native_ablation.sh smoke` | 共享 Full 和 29 个独立 variant 各使用 8 条 event、最多 3 个 query context | 生成 34 行逻辑对照，只验证完整性 |
| `run_agent_native_ablation.sh run` | 共享 Full 和 29 个独立 variant 默认各使用 1000 条 event、100 个 query context | 用于中等规模检查 |
| `run_agent_native_ablation.sh full` | 30 次物理运行均使用全部输入，query 默认 100；输出仍为 34 行 | 正式全量输入 |
| `--run-id ID` | 固定结果目录名和 S3 prefix | full 必须显式保存；恢复时不可改变 |
| `--event-limit 0` | 顺序读取 `events.jsonl` 和全部 trace 文件中的全部 event | full 固定为 0；正整数仅用于 smoke/中等测试 |
| `--query-limit N` | 从已写入的 agent/session context 中选择最多 N 个 query | 必须大于 0；同一组实验保持一致 |
| `--port N` | 当前 variant 的 Plasmod HTTP port | 只改 Plasmod port；MinIO 仍固定 9000/9001 |
| `--skip-build` | 跳过 runner 开始时的 `make build` | 仅限已经按当前 core commit 构建并通过 preflight；不跳过 C++ `.so` 检查 |
| `--resume` | 复用同一 run id 中已经完整写出的分组 CSV | 必须保持 commit、数据、event/query limit 和 port 不变 |

默认不要设置 runner 内部的 `PLASMOD_*` 或 `S3_*` 环境变量：脚本会为每个 variant 写入固定配置，variant 自身的 ablation profile 最后覆盖对应项。外部 shell 只需要设置 `RUN_ID`；`PLASMOD_ABLATION_MIN_FREE_GB` 仅控制 preflight 的磁盘门槛，不改变实验 workload。

---

## 7. 关键参数

| 参数 | 含义 |
|---|---|
| `--dataset {nfcorpus,deep10M}` | 数据集 |
| `--index-count N` | 索引前 N 条向量（0 = 全部） |
| `--queries N` | 查询数（deep 最多 10000） |
| `--index {flat,ivf_flat,ivf_pq,ivf_sq8,hnsw,all}` | 索引类型 |
| `--db {all,plasmod,qdrant,milvus,lancedb,chromadb}` | 选 DB |
| `--topk N` | 返回 top-K |
| `--sweep-recall` | 扫 recall 0.5-1.0 的 QPS 曲线 |
