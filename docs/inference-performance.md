# 1000 QPS 推理部署与调优

## 结论

当前模型是 MacBERT 意图分类器，不是生成式 LLM。对单系统约 1000 QPS 的场景，建议保留 FastAPI 作为系统边界，但不要让每个 HTTP 请求单独触发一次 GPU 前向计算。本项目已经在服务内部加入动态微批处理：并发请求先进入有界队列，在最多 3ms 内合并成最多 64 条，再执行一次分词和模型前向。

本机 GPU 对 v22、同一条 20 字左右中文文本的进程内纯模型测试结果如下：

| Batch size | 吞吐（条/秒） | 每批耗时 |
| ---: | ---: | ---: |
| 1 | 102.2 | 9.78ms |
| 8 | 794.2 | 10.07ms |
| 16 | 1658.7 | 9.65ms |
| 32 | 2740.6 | 11.68ms |
| 64 | 3491.1 | 18.33ms |

最终 FastAPI 进程内压测中，3000 条单请求、并发 256、batch 64 得到 1113.2 QPS，p99 为 394.35ms。调用方每次聚合 16 条后，只需 188 次 HTTP 请求，同一批 3000 条提升到 2280.3 QPS，p99 为 309.61ms。纯模型虽有更高吞吐，逐条 HTTP 的校验、JSON 和异步调度仍会消耗 CPU。因此 1000 QPS 场景默认采用 batch 64，并优先让调用方小批量提交；若更重视低流量下的单请求延迟，可改为 batch 32。结果会随 GPU 型号、CPU、文本长度和意图数量变化，部署机器仍需复测。

## 启动配置

```bash
export LAZY_AGENT_ROUTER_DEVICE=cuda
export LAZY_ROUTER_PRELOAD_MODEL=lazy-agent-router-macbert-v22
export LAZY_ROUTER_MAX_BATCH_SIZE=64
export LAZY_ROUTER_MAX_WAIT_MS=3
export LAZY_ROUTER_QUEUE_SIZE=4096
export LAZY_ROUTER_REQUEST_TIMEOUT_MS=500

uv run uvicorn lazy_agent_router.serving.app:app \
  --host 0.0.0.0 --port 8000 --workers 1
```

只有一张 GPU 时使用一个 worker。多个 Uvicorn worker 会各自加载一份约 400MB 模型、各自维护队列，既增加显存占用，也会把本来可以合并的请求拆散。若有多张 GPU，建议每张卡运行一个独立实例，再由网关负载均衡。

`LAZY_ROUTER_PRELOAD_MODEL` 会在端口开始接收流量前加载模型，并执行一个完整 batch 的 GPU 预热；模型不存在或加载失败时服务会启动失败，不会产生“健康但不能推理”的实例。启动后可再执行一次业务探针：

```bash
curl -sS http://127.0.0.1:8000/v1/models
curl -sS -X POST http://127.0.0.1:8000/v1/route \
  -H 'content-type: application/json' \
  -d '{"model":"lazy-agent-router-macbert-v22","query":"服务预热"}'
```

## 两种调用方式

原有单条接口保持不变。即使客户端并发调用，服务端也会自动合批：

```http
POST /v1/route
{"model":"lazy-agent-router-macbert-v22","query":"查询采购审批进度"}
```

如果调用系统能在本地先聚合请求，优先使用显式批量接口，一次 HTTP 最多提交 256 条，能继续减少连接、JSON 解析和网关开销：

```http
POST /v1/route/batch
{
  "model": "lazy-agent-router-macbert-v22",
  "queries": ["查询采购审批进度", "SAP 接口报错", "账号无法登录"]
}
```

客户端应启用 HTTP/1.1 keep-alive 连接池，或在网关支持时使用 HTTP/2，不要为每个请求重新创建连接。对于同一 Python 系统，也可直接在进程内持有 `LazyAgentRouter` 并调用 `predict_batch()`，这样完全没有 HTTP 开销；代价是业务进程会与 PyTorch、模型显存和升级生命周期耦合。

## 监控与背压

`GET /v1/inference/stats` 返回累计请求数、批次数、当前队列深度、平均 batch、平均推理耗时和错误数。重点关注：

- `average_batch_size`：持续高并发时应明显大于 1；接近 64 说明合批充分；
- `queue_depth`：偶发升高正常，持续增长表示到达速率超过处理能力；
- HTTP 429：队列已满，客户端应执行带抖动的短退避，不能无限重试；
- HTTP 504：请求等待超过配置值，需要降低上游并发、扩容 GPU 实例或检查超长文本；
- p95/p99 延迟：吞吐达标不等于尾延迟达标，必须用真实文本长度分布测试。

## 压测

进程内 API 压测会加载真实模型，覆盖 FastAPI、队列、微批、分词和 GPU 推理，但不包含网卡、反向代理和 TLS：

```bash
uv run python scripts/benchmark_inference.py \
  --model lazy-agent-router-macbert-v22 \
  --requests 5000 --concurrency 256 --batch-size 64
```

如果调用方能一次提交多条，再增加 `--client-batch-size 16` 测试显式批量接口。这里的吞吐按业务文本条数计算，而不是 HTTP 请求数。

上线前还应从另一台机器使用 k6、wrk 或 JMeter 对实际入口压测。建议至少验证稳定 1000 QPS 持续 10 分钟、突发 1500 QPS、p95/p99、429/504 比例、GPU 利用率与显存，并使用生产文本长度分布，而不是只发同一句短文本。

## 什么时候不使用 FastAPI

FastAPI 本身通常不是这个模型的主要瓶颈；batch=1 时 GPU 前向只有约 102 QPS，远早于 HTTP 框架达到上限。合批后，如果真实压测显示 HTTP/JSON 已成为瓶颈，可按顺序处理：

1. 调用方改用 `/v1/route/batch`；
2. 同一系统、同一进程部署时直接调用 `predict_batch()`；
3. 跨语言且极低延迟时增加 gRPC 接口；
4. 多系统共用时拆出独立推理服务，并按 GPU 水平扩容。

vLLM 和 Ollama 面向生成式模型的自回归解码、KV Cache 和通用 LLM 服务，本模型是 `AutoModelForSequenceClassification`，不能直接按普通聊天模型接入它们。当前的动态批处理是更匹配分类模型的优化方向。
