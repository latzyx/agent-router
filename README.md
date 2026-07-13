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

训练器会在入口处去重并检查冲突标签，随后按意图分层切分 80% 训练集和 20% 验证集。训练阶段使用置信度感知的奖励/惩罚损失，默认奖励系数为 `0.20`、惩罚系数为 `0.75`；验证阶段使用不带奖励权重的标准交叉熵，并同时计算准确率和宏平均 F1。训练支持动态 padding、CUDA FP16、学习率预热、权重衰减和早停，最终模型目录会生成 `training_report.json` 供审计与模型比较。

`datasets/evaluation/realistic_queries.jsonl` 是与训练语料分离的人工标注评测集，包含真实企业业务风格的表达，以及路由、风险、工具和实体期望。它不是生产用户日志；接入脱敏的真实用户样本后，应由业务人员复核标注并扩展该评测集。

上传到训练接口的 JSONL 会先去重再训练。当前 `datasets/uploads/` 中的三份文件内容相同，去重后得到 390 条、13 个 Ecology/OA 业务意图样本；可用以下命令生成可复现的处理数据集：

```bash
uv run python -m lazy_agent_router.training.prepare_dataset
```

13 类业务模型对应的 Agent 映射见 `configs/intents_uploaded.yaml`。控制台会自动发现 `models/` 下的模型，可直接选择当前候选模型 v15 进行测试；不选择模型时仍使用关键词基线。

当前 v15 使用 489 条去重样本训练。除 uploads 回归集外，项目保留了三套与训练文本零重叠、每类各 3 条的人工标注挑战集，避免仅用训练数据评价模型：

| 评测集 | v14 Accuracy | v15 Accuracy | v15 Macro-F1 | v15 Agent Accuracy |
| --- | ---: | ---: | ---: | ---: |
| 冻结挑战集 v1（39 条） | 92.31% | 94.87% | 94.73% | 100% |
| 冻结挑战集 v2（39 条） | 87.18% | 89.74% | 89.73% | 97.44% |
| 首次揭示盲测集 v3（39 条） | 79.49% | 92.31% | 91.59% | 100% |
| 三套挑战集平均 | 86.32% | 92.31% | 92.02% | 99.15% |
| uploads 回归集（29,250 行、390 条唯一文本） | 100% | 100% | 100% | 100% |

挑战集规模仍然较小，因此这些结果适合用于版本回归和方向判断，不能替代上线后的脱敏真实流量测试。低置信度结果建议转人工或回退到安全路由。

### 下载 v15 训练模型

v15 权重通过 GitHub Release 单独发布，不进入 Git 仓库。克隆项目后可直接下载、校验并解压：

```bash
curl -fL -o /tmp/lazy-agent-router-macbert-v15.tar.gz \
  https://github.com/latzyx/agent-router/releases/download/model-v15/lazy-agent-router-macbert-v15.tar.gz
echo "03baa4503c3fdc25a1c100d90b143b02bcd7f3ef0e1ff43b75047eeb14e73be4  /tmp/lazy-agent-router-macbert-v15.tar.gz" \
  | sha256sum -c -
mkdir -p models
tar -xzf /tmp/lazy-agent-router-macbert-v15.tar.gz -C models
```

启动服务后，前端模型列表会自动出现 `lazy-agent-router-macbert-v15`。

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
