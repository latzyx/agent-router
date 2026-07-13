# lazy-agent-router

面向企业 AI Agent 的中文意图路由服务。它将自然语言请求分类为业务意图，提取常见实体，评估路由置信度与风险，并输出目标 Agent 和可执行的工具计划。

> 当前默认使用无需下载模型的关键词分类器，便于本地体验；生产环境可接入已微调的 MacBERT 分类器。

## 特性

- 配置驱动的意图、Agent、风险和工具映射
- 关键词基线与可替换的 MacBERT 分类器
- 规则实体提取（例如员工编号）
- 低置信度回退与高风险操作确认
- FastAPI HTTP API、浏览器控制台与训练任务接口
- LangGraph、MCP 和 OpenAI 集成适配层

## 快速开始

要求：Python 3.11+，推荐使用 [uv](https://docs.astral.sh/uv/)。

```bash
uv sync --extra dev
uv run pytest -q
uv run uvicorn lazy_agent_router.serving.app:app --host 0.0.0.0 --port 8000
```

服务启动后可访问：

- 控制台：<http://127.0.0.1:8000/>
- OpenAPI 文档：<http://127.0.0.1:8000/docs>
- 健康检查：<http://127.0.0.1:8000/health>

发起一次路由请求：

```bash
curl -X POST http://127.0.0.1:8000/v1/route \
  -H 'content-type: application/json' \
  -d '{"query":"帮我查询采购审批流程"}'
```

示例响应：

```json
{
  "query": "帮我查询采购审批流程",
  "intent": "workflow.query",
  "confidence": 1.0,
  "agent": "workflow-agent",
  "decision": "direct_route",
  "risk_level": "low",
  "candidate_intents": [{"intent": "workflow.query", "confidence": 1.0}],
  "entities": {},
  "tool_plan": [{"name": "workflow.get_status", "description": "查询审批流程状态"}]
}
```

## 路由策略

请求依次经过分类、Agent 查询、实体提取和工具匹配。置信度低于 `configs/router.yaml` 中的阈值时，结果会路由到 `fallback-agent`，并标记为 `fallback`。风险级别为 `high` 的操作会标记为 `confirmation_required`；调用方必须在实际执行前征得用户确认。

主要配置文件：

| 文件 | 用途 |
| --- | --- |
| `configs/intents.yaml` | 意图、关键词、目标 Agent 和风险级别 |
| `configs/router.yaml` | 直接路由置信度阈值 |
| `configs/risk_policy.yaml` | 需要确认的风险级别 |
| `configs/tools.yaml` | 每个意图可用的工具与必填实体 |

## Python 使用方式

```python
from lazy_agent_router import LazyAgentRouter
from lazy_agent_router.classifiers.keyword import KeywordIntentClassifier
from lazy_agent_router.core.registry import AgentRegistry

router = LazyAgentRouter(
    classifier=KeywordIntentClassifier({"workflow.query": ["查询", "流程"]}),
    registry=AgentRegistry({
        "fallback_agent": "fallback-agent",
        "intents": {
            "workflow.query": {"agent": "workflow-agent", "risk_level": "low"}
        },
    }),
)

result = router.predict("帮我查询采购流程")
print(result.to_dict())
```

## 使用本地 MacBERT 模型

训练或下载后的 Transformers 模型目录应包含模型配置、分词器和权重。请将模型保存在仓库外部或 `models/` 目录（该目录不会被 Git 提交），然后在创建路由器时注入 `MacBERTClassifier`：

```python
from lazy_agent_router import LazyAgentRouter
from lazy_agent_router.classifiers.macbert import MacBERTClassifier
from lazy_agent_router.core.registry import AgentRegistry

router = LazyAgentRouter(
    classifier=MacBERTClassifier("/path/to/lazy-agent-router-macbert"),
    registry=AgentRegistry("configs/intents.yaml"),
)
```

训练数据采用 JSONL 格式，示例数据位于 `datasets/raw/intents.jsonl`。服务还提供：

- `POST /v1/training/start`：以表单方式提交 `model_source`，可选上传 `.jsonl` 数据集；
- `GET /v1/training/status`：查询训练任务状态。

## 容器运行

```bash
docker compose up --build
```

默认端口为 `8000`。如需 GPU 或远程模型，请按部署环境调整镜像、运行时和模型路径。

## 项目结构

```text
src/lazy_agent_router/
├── classifiers/     # 关键词、MacBERT、嵌入与集成分类器
├── core/            # 路由、置信度、风险策略与 Agent 注册表
├── entities/        # 实体提取
├── integrations/    # LangGraph、MCP、OpenAI 适配器
├── serving/         # FastAPI 应用、路由与控制台
├── tools/           # 工具注册与匹配
└── training/        # 数据集、训练、评估与语料生成
```

## 开发

```bash
uv sync --extra dev
uv run pytest -q
```

请勿提交本地模型、训练上传文件或密钥。复制 `.env.example` 为 `.env` 后，在本机填写环境变量。

## License

本项目采用 [Apache License 2.0](LICENSE)。
