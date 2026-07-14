# 模型训练与动态调参教程

本文既是训练教程，也是 `src/lazy_agent_router/training/trainer.py` 的实现说明。推荐先用控制台完成第一次训练，再根据验证集和独立挑战集的结果调整参数。

## 是否需要调整模型

当前推荐 v22/MacBERT。它在五套与训练语料零文本重叠的挑战集上，平均 Macro-F1 为 `0.9404`、平均 Agent Accuracy 为 `0.9846`；相比 v15，平均 Macro-F1 提升约 3.6 个百分点，总意图错误从 18 条降到 11 条。v22 的提升来自边界数据和跨 Agent 错误惩罚，不需要换用更大的基础模型。

建议采用以下顺序：

1. 继续使用 v22/MacBERT 作为生产候选，优先补齐低准确率意图的真实表达。
2. 固定训练集、验证集和一份从未参与调参的盲测集，再比较模型，避免因数据变化误判模型收益。
3. 需要模型对照实验时，使用相同参数分别训练 `hfl/chinese-macbert-base`、`hfl/chinese-roberta-wwm-ext` 和 `bert-base-chinese`，以独立测试集 Macro-F1、P95 推理延迟和显存占用共同决策。
4. 只有当新增数据后的 Macro-F1 连续多个版本不再提升，或线上查询明显超过 128 token，才优先考虑更大模型；若更关注吞吐，应测试蒸馏模型，而不是盲目增大模型。

## 1. 准备环境并启动控制台

```bash
uv sync --extra dev
uv run lazy-agent-router
```

浏览器访问 `http://localhost:8000`。控制台包含四个 Tab：

- **模型训练**：选择基础模型、数据集并启动后台训练；
- **数据集**：上传并校验 JSONL；
- **测试**：选择已训练模型执行意图路由；
- **配置**：动态修改超参数，配置保存在浏览器中，开始训练时通过 API 提交。

## 2. 准备并上传数据集

文件使用 UTF-8 编码的 JSONL，每行一个对象：

```json
{"text":"帮我查询采购审批流程","intent":"workflow.query"}
{"text":"采购单现在审批到哪一步","intent":"workflow.query"}
{"text":"创建一个采购审批","intent":"workflow.start"}
{"text":"发起新的采购流程","intent":"workflow.start"}
```

每条记录必须包含非空字符串 `text` 和 `intent`。同一文本不能对应多个 intent，每个 intent 至少准备两条样本；实际训练建议每类至少 50–100 条，并覆盖口语、省略、错别字和容易混淆的边界表达。

在“数据集”Tab 选择 `.jsonl` 文件并上传。服务会返回记录数、意图数和数据集 ID，训练页随后可直接选择它。上传限制为 20 MB。

也可以调用接口：

```bash
curl -X POST http://localhost:8000/v1/datasets \
  -F 'dataset=@datasets/raw/intents.jsonl'

curl http://localhost:8000/v1/datasets
```

## 3. 从前端动态训练

先进入“配置”Tab 调整参数并保存，再回到“模型训练”Tab：

1. 基础模型优先选择 `MacBERT Base`。续训时也可以选择“自定义”，填写已有本地模型目录。
2. 选择已上传数据集；不选择时使用 `datasets/raw/intents.jsonl`。
3. 确认页面展示的轮数、batch size 和学习率，点击“开始训练”。
4. 页面每 3 秒刷新真实训练进度，展示当前 Epoch、参数更新步数和百分比；模型保存与最终评估完成后才会达到 100%。完成后的模型目录会出现在“测试”Tab，无需重启服务。

前端不会拼接训练命令，而是把动态配置序列化为 JSON，通过 multipart 表单的 `parameters` 字段提交。后端使用 Pydantic 再次校验上下限，因此不能绕过限制提交危险值。

## 4. 通过 HTTP API 传递动态参数

先查询默认值、字段约束和推荐模型：

```bash
curl http://localhost:8000/v1/training/config
```

启动训练（将 `DATASET_ID` 替换成上传接口返回的 ID）：

```bash
curl -X POST http://localhost:8000/v1/training/start \
  -F 'model_source=hfl/chinese-macbert-base' \
  -F 'dataset_id=DATASET_ID' \
  -F 'parameters={"epochs":8,"batch_size":16,"learning_rate":0.00002,"max_length":128,"validation_split":0.2,"reward_strength":0.2,"penalty_strength":0.75,"early_stopping_patience":2,"early_stopping_min_delta":0.00001,"weight_decay":0.01,"warmup_ratio":0.1,"max_grad_norm":1.0,"gradient_accumulation_steps":1,"seed":42,"device":"auto"}'

curl http://localhost:8000/v1/training/status
```

为了兼容旧客户端，也可以在 `/v1/training/start` 请求中直接使用 `dataset` 文件字段；服务会先把它纳入受管数据集并完成同样的校验。

## 5. 在 Python 代码中调节参数

不启动 Web 服务时，可以直接调用训练函数：

```python
from lazy_agent_router.training.trainer import train_macbert

output = train_macbert(
    train_file="datasets/raw/intents.jsonl",
    model_path="hfl/chinese-macbert-base",
    output_dir="models/lazy-agent-router-macbert-experiment",
    epochs=8,
    batch_size=16,
    learning_rate=2e-5,
    max_length=128,
    validation_split=0.2,
    reward_strength=0.2,
    penalty_strength=0.75,
    early_stopping_patience=2,
    early_stopping_min_delta=1e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    max_grad_norm=1.0,
    gradient_accumulation_steps=1,
    seed=42,
    device="auto",
)
print(output)
```

动态参数及建议范围：

| 参数 | 后端范围 | 推荐起点 | 调整建议 |
|---|---:|---:|---|
| `epochs` | 1–50 | 5–8 | 由早停控制实际轮数 |
| `batch_size` | 1–128 | 16 | CUDA 显存不足时先降到 8 |
| `learning_rate` | 1e-7–1e-2 | 2e-5 | 从旧模型续训可降到 1e-5 |
| `max_length` | 16–512 | 128 | 根据文本 token 长度 P95 设置 |
| `validation_split` | 0.05–0.5 | 0.2 | 数据很少时仍需保证每类有验证样本 |
| `reward_strength` | 0–<1 | 0.2 | 过高会削弱已正确样本的学习 |
| `penalty_strength` | 0–5 | 0.75 | 错误高置信样本多时小步提高 |
| `cross_group_penalty_strength` | 0–5 | 0 | 路由到错误 Agent 较多时从 0.5 开始测试 |
| `early_stopping_patience` | 1–20 | 2 | 指标波动明显时改为 3 |
| `weight_decay` | 0–0.5 | 0.01 | 过拟合时可测试 0.02 |
| `warmup_ratio` | 0–0.5 | 0.1 | 小数据通常无需超过 0.1 |
| `max_grad_norm` | >0–10 | 1.0 | 一般保持默认 |
| `gradient_accumulation_steps` | 1–64 | 1 | 小显存时用 2/4 模拟大 batch |
| `device` | auto/cpu/cuda | auto | 指定 cuda 但不可用时会报错 |

有效 batch 近似为 `batch_size × gradient_accumulation_steps`。调整模型或参数时一次只改变一个主要变量，并保持数据切分和 `seed` 不变。

## 6. 测试新模型

在“测试”Tab 选择训练完成的模型，输入查询并执行。也可调用：

```bash
curl -X POST http://localhost:8000/v1/route \
  -H 'content-type: application/json' \
  -d '{"model":"lazy-agent-router-macbert-v22","query":"SAP 接口为什么一直报错"}'
```

页面测试适合快速检查单条案例，模型发布决策必须使用冻结的独立测试集。优先比较 Macro-F1，再检查每个意图的召回率、混淆样本、Agent Accuracy、延迟和资源占用。

## 训练实现原理

## 完整数据流

```text
JSONL 数据
  → 文本规范化、去重、冲突标签检查
  → 按意图分层切分训练集和验证集
  → tokenizer 分词和长度截断
  → 按批次动态 padding
  → MacBERT 前向计算
  → 训练：奖励/惩罚交叉熵
  → 验证：标准交叉熵、Accuracy、Macro-F1
  → 早停并恢复最佳权重
  → 保存模型、tokenizer、training_report.json
```

## 为什么必须去重

当前上传目录曾包含三份完全相同的数据。如果直接训练，重复文本会被计算三次梯度，看起来数据量变大，实际只是人为提高了这些样本的权重。训练入口会再次去重，因此即使调用方忘记运行预处理脚本，也不会被重复上传污染。

同一文本如果对应不同 intent，训练器会直接失败。模型无法同时学习“完全相同输入属于两个互斥类别”，这种冲突必须由标注人员决定正确标签。

## 分层切分与数据泄漏

分层切分会在每个 intent 内分别抽取验证样本，保证每个类别都参与验证。固定 `seed=42` 后，各模型版本使用相同切分，指标才可比较。

如果提供独立 `validation_file`，训练器会检查文本是否与训练集重叠。重叠意味着模型在训练时已经见过验证答案，会产生虚假的高准确率。

## 奖励与惩罚损失

基础损失仍然是逐样本交叉熵：

```text
最终训练损失 = mean(交叉熵 × 样本权重)
```

- 预测正确时：置信度越高，权重越低，但最低为 `0.5`，基础监督不会消失。
- 预测错误时：置信度越高，权重越高，重点纠正“自信地犯错”。
- 权重由 `detach()` 后的概率计算，不参与反向传播，避免模型操纵权重。

奖励/惩罚只用于训练。验证阶段使用标准交叉熵，否则不同 epoch 的验证 loss 会因权重变化而不可比较。

## Accuracy 与 Macro-F1

Accuracy 是全部样本的正确比例。Macro-F1 会先计算每个 intent 的 F1，再等权平均，因此不会让样本多的类别掩盖小类别。最佳模型优先按 Macro-F1 选择；Macro-F1 相同时，再比较标准验证 loss。

## 动态 padding

旧逻辑将所有文本补齐到 128 token。企业查询通常很短，这会让大量 padding token 参与计算。现在每个 batch 只补齐到该批最长文本，实际 v9 训练吞吐相比旧逻辑明显提升，同时减少显存占用。

## FP16、预热和梯度裁剪

- CUDA 环境启用 FP16，降低显存和计算开销；softmax 奖惩权重仍用 FP32 计算以保持稳定。
- 前 10% 训练步用于学习率预热，避免刚开始训练时较大的学习率破坏预训练权重。
- `max_grad_norm=1.0` 对梯度裁剪，避免异常 batch 产生过大的参数更新。
- `weight_decay=0.01` 用于降低小数据集上的过拟合风险。

## 早停

`early_stopping_patience=2` 表示连续两轮没有实质改善就停止。`early_stopping_min_delta=1e-5` 表示验证 loss 至少下降这么多才重置早停计数，避免浮点抖动让训练无限继续。最佳权重保存与早停阈值相互独立：即使 loss 只下降很小，仍会保存真正更好的权重。

例如 v10 请求最多训练 10 轮，但在第 3 轮自动停止，说明后续训练收益不足。

## 训练报告

每个新模型目录会生成 `training_report.json`，记录：

- 模型来源和设备；
- 训练集、验证集条数；
- 标签列表；
- 请求轮数和实际完成轮数；
- 学习率、batch size、奖惩系数；
- Accuracy、Macro-F1 和标准验证 loss；
- 训练耗时和吞吐。

模型比较应优先使用独立真实评测集，其次使用 Macro-F1，不能只看训练 loss。

## 使用 uploads 做回归测试

可以用下面的命令测试上传目录中的全部 JSONL：

```bash
uv run python -m lazy_agent_router.training.evaluate_uploaded \
  --model models/lazy-agent-router-macbert-v15 \
  --source datasets/uploads \
  --agent-config configs/intents_uploaded.yaml \
  --training-reference datasets/processed/uploaded_intents_v15.jsonl \
  --device cuda
```

报告保存在模型目录的 `upload_evaluation.json`，包含原始行数、去重文本数、
Accuracy、Macro-F1、Agent 正确率、每个意图的最低置信度以及错误样本。

注意：如果 uploads 已被用于训练，这只能算回归测试，不能证明模型能够识别从未见过的用户表达。

## 使用冻结挑战集测试泛化能力

`datasets/evaluation/upload_challenge/`、`datasets/evaluation/upload_challenge_v2/` 和
`datasets/evaluation/upload_challenge_v3/` 分别保存三套人工标注挑战集。它们与对应训练集做精确文本去重，评估器也会通过
`--training-reference` 再次检查重叠；只有 `training_overlap` 为 `0` 的报告才能标记为
`independent_challenge`。

```bash
uv run python -m lazy_agent_router.training.evaluate_uploaded \
  --model models/lazy-agent-router-macbert-v15 \
  --source datasets/evaluation/upload_challenge_v3 \
  --agent-config configs/intents_uploaded.yaml \
  --training-reference datasets/processed/uploaded_intents_v15.jsonl \
  --output models/lazy-agent-router-macbert-v15/challenge_v3_evaluation.json \
  --device cuda
```

不要根据同一挑战集的错误反复添加近似答案并继续汇报该集合成绩，否则它会逐渐变成训练集。
正确做法是冻结已揭示的集合，使用表达不同的边界样本训练，并在新的盲测集上确认提升。
