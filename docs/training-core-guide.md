# Lazy Agent Router 训练核心指南

这份文档回答三个问题：模型是怎么训练出来的、训练时真正重要的是什么、源码应该先看哪里。

## 一句话理解

这个项目把用户文本训练成一个多分类模型：输入一句查询，输出一个 `intent`；路由器再根据 intent 找到对应 Agent。

v22 不是靠换更大的模型得到提升，而是在 MacBERT 上做好四件事：

1. 清洗、去重并补充容易混淆的边界数据；
2. 使用按 intent 分层的训练集/验证集切分；
3. 对“高置信错误”以及“路由到错误 Agent”施加更高训练损失；
4. 使用冻结挑战集选择模型，不只看训练集和验证集分数。

## 完整训练链路

```text
前端配置页
  ↓ parameters JSON
POST /v1/training/start
  ↓ Pydantic 参数范围校验
后台 TrainingJob
  ↓ 数据集、模型路径、Agent 分组、进度回调
train_macbert()
  ├─ JSONL 读取、规范化、去重、冲突检查
  ├─ 按 intent 分层切分训练集和验证集
  ├─ tokenizer 分词、截断、动态 padding
  ├─ 加载 MacBERT 序列分类模型
  ├─ 奖励/惩罚损失进行反向传播
  ├─ 每轮计算 Accuracy、Macro-F1、标准验证 loss
  ├─ 早停并恢复最佳 epoch 权重
  └─ 保存模型、tokenizer、training_report.json
  ↓
冻结挑战集批量评测
  ↓
比较 Macro-F1、Agent Accuracy、低置信和错误样本
```

对应调用关系：

```text
serving/routes.py
  → serving/schemas.py
  → serving/training_jobs.py
  → training/trainer.py
  → training/evaluate_uploaded.py
```

## 数据是怎么准备的

训练文件是 UTF-8 JSONL，每行至少有两个字段：

```json
{"text":"SAP采购订单没有同步到OA","intent":"sap_interface"}
{"text":"请通过当前付款申请","intent":"workflow_approve"}
```

训练入口会再次执行以下检查，即使数据绕过前端上传也一样：

- 去掉文本和标签首尾空白；
- 删除完全重复的 `text + intent`；
- 同一文本对应两个 intent 时直接报错；
- 每个 intent 至少两条样本，确保训练集和验证集都能覆盖；
- 提供独立验证集时，检查它与训练集是否存在文本泄漏。

v22 使用的数据结构是：

```text
390 条真实上传语料
  + 既有业务增强语料
  + v20/v21 边界增强语料
  = 567 条去重训练数据
```

相关文件：

- `datasets/processed/uploaded_intents_v21.jsonl`：v22 使用的去重训练集；
- `datasets/augmentation/v20_confusion_boundaries.jsonl`：权限、规则、预测、审批等边界数据；
- `datasets/augmentation/v21_agent_boundaries.jsonl`：跨系统同步、当前节点、催办等 Agent 边界数据；
- `datasets/evaluation/upload_challenge_v1~v5/`：不进入训练的冻结评测集。

核心原则是：**增加数据不等于复制挑战集答案。** 已经看过错误答案的测试集只能作为回归集，下一轮模型必须再使用新的零重叠挑战集确认泛化。

## 最核心的代码：损失函数

第一核心是 [`reward_penalty_loss()`](../src/lazy_agent_router/training/trainer.py#L86)。

标准分类模型只计算交叉熵。这里先保留逐样本交叉熵，再根据当前预测结果乘上权重：

```text
最终损失 = mean(每条样本的交叉熵 × 样本权重)
```

设模型置信度为 `p`：

```text
预测正确：
  weight = max(0.5, 1 - reward_strength × p)

预测错误，但仍属于同一 Agent：
  weight = 1 + penalty_strength × p

预测错误，而且路由到了其他 Agent：
  weight = 1
         + penalty_strength × p
         + cross_group_penalty_strength × p
```

这三个分支分别解决：

- 已经正确且高置信的简单样本不再主导梯度；
- 越自信的错误越需要重点纠正；
- 错 intent 但仍进入正确 Agent，通常比进入错误 Agent 的业务影响小，因此跨 Agent 错误获得额外惩罚。

计算权重使用的是 `logits.detach()` 后的概率。也就是说，权重本身不参与反向传播，模型不能通过操纵权重来“逃避”基础交叉熵。

v22 的关键值是：

```text
reward_strength = 0.20
penalty_strength = 0.75
cross_group_penalty_strength = 1.50
```

## 第二核心：训练和验证使用不同损失

第二核心是 [`RewardPenaltyTrainer.compute_loss()`](../src/lazy_agent_router/training/trainer.py#L194)。

```python
if model.training:
    loss = reward_penalty_loss(...)
else:
    loss = cross_entropy(...)
```

训练阶段使用业务奖惩损失，验证阶段必须使用标准交叉熵。

如果验证也使用动态奖惩权重，每个 epoch 的验证 loss 会随着模型预测和置信度改变尺度，无法公平比较。现在验证 loss 始终处于同一尺度，早停和最佳模型选择才可信。

## 第三核心：早停和最佳权重恢复

[`RewardPenaltyTrainer.evaluate()`](../src/lazy_agent_router/training/trainer.py#L213) 每轮验证后执行：

1. 优先比较 Macro-F1；
2. Macro-F1 相同时比较标准验证 loss；
3. 指标改善时复制当前模型权重到 CPU；
4. 连续若干轮没有实质改善时停止训练；
5. 训练结束后调用 `restore_best_model()`，保存最佳 epoch，而不是最后一个 epoch。

v22 最多请求 5 轮，实际第 3 轮早停，最终恢复第 1 轮最佳权重。这避免了继续训练造成的边界过拟合。

## 第四核心：完整训练入口

完整流水线在 [`train_macbert()`](../src/lazy_agent_router/training/trainer.py#L277)。建议按下面顺序阅读：

| 代码区域 | 作用 |
|---|---|
| `deduplicate_rows()` | 去重、空值和冲突标签检查 |
| `stratified_split()` | 每个 intent 内部分层切分 |
| 标签与 Agent 分组 | 建立 `label2id` 和 `label_group_ids` |
| tokenizer + `DataCollatorWithPadding` | 截断并按当前 batch 动态补齐 |
| `TrainingArguments` | 学习率、FP16、warmup、梯度裁剪等 |
| `RewardPenaltyTrainer` | 自定义训练损失、验证和早停 |
| 保存阶段 | 保存最佳权重、tokenizer 和训练报告 |

动态 padding 很重要。企业查询通常较短，如果全部固定补到 128 token，会浪费显存和计算；现在只补到当前 batch 的最长文本。

## 前端参数如何传入训练代码

前端不会直接拼接 shell 命令，而是提交 JSON 参数：

```text
配置 Tab
  → localStorage
  → FormData.parameters
  → POST /v1/training/start
```

后端参数模型位于 [`TrainingParameters`](../src/lazy_agent_router/serving/schemas.py#L23)，负责限制 epochs、batch size、学习率、奖惩强度和设备等范围，并拒绝未知字段。

接口入口是 [`start_training()`](../src/lazy_agent_router/serving/routes.py#L56)。它解析参数、解析数据集 ID，再交给 [`TrainingJob`](../src/lazy_agent_router/serving/training_jobs.py#L17)。

`TrainingJob` 负责：

- 同一时间只允许一个训练任务；
- 在后台线程中加载 PyTorch，避免阻塞普通 API；
- 从配置文件生成 intent → Agent 分组；
- 自动生成下一个模型版本目录；
- 把真实 step/epoch 进度暴露给前端；
- 捕获后台异常并展示到训练状态区。

## v22 是怎么训练的

v22 不是从基础 MacBERT 重新开始，而是从意图表现较好的 v20 继续微调：

```text
基础权重：models/lazy-agent-router-macbert-v20
训练数据：datasets/processed/uploaded_intents_v21.jsonl
训练样本：454
验证样本：113
学习率：5e-6
batch size：16
最大轮数：5
实际轮数：3（早停）
跨 Agent 惩罚：1.5
设备：CUDA + FP16
```

核心训练调用可以简化理解为：

```python
train_macbert(
    train_file="datasets/processed/uploaded_intents_v21.jsonl",
    model_path="models/lazy-agent-router-macbert-v20",
    output_dir="models/lazy-agent-router-macbert-v22",
    epochs=5,
    batch_size=16,
    learning_rate=5e-6,
    reward_strength=0.2,
    penalty_strength=0.75,
    cross_group_penalty_strength=1.5,
    label_groups=intent_to_agent,
    early_stopping_patience=2,
    device="cuda",
)
```

实际参数和最终指标都记录在 `models/lazy-agent-router-macbert-v22/training_report.json`。

## 如何判断模型真的变好了

不能只看训练 loss，也不能只看从训练数据切出的验证集。模型选择至少比较四项：

1. **Macro-F1**：所有 intent 等权，避免大类别掩盖小类别；
2. **Agent Accuracy**：是否把请求交给正确 Agent；
3. **低置信样本数**：是否大量触发 fallback；
4. **冻结挑战集错误**：是否能识别训练中没见过的表达。

v22 在五套零文本重叠挑战集上的汇总：

| 模型 | 平均 Macro-F1 | 最差 Macro-F1 | 平均 Agent Accuracy | 总意图错误 |
|---|---:|---:|---:|---:|
| v15 | 0.9043 | 0.8716 | 0.9897 | 18 |
| v20 | 0.9181 | 0.8868 | 0.9744 | 15 |
| v22 | **0.9404** | **0.8868** | **0.9846** | **11** |

批量评测入口是 [`evaluate_uploaded()`](../src/lazy_agent_router/training/evaluate_uploaded.py#L60)。它还会统计训练文本重叠、每类准确率、最低置信度、错误明细和 Agent Accuracy。

## 调参时最重要的顺序

建议按这个顺序处理问题：

1. **先看错误样本和混淆类别**，不要先换模型；
2. 检查重复数据、冲突标签和训练/评测泄漏；
3. 补真实边界表达，保持各 intent 相对均衡；
4. 从稳定模型续训时使用较低学习率，如 `5e-6 ~ 1e-5`；
5. 高置信错误多时调整 `penalty_strength`；
6. 跨 Agent 错误多时再调整 `cross_group_penalty_strength`；
7. 每轮只改变一个主要变量，并保持 `seed` 与冻结评测集不变。

不要通过降低路由置信度阈值掩盖分类错误。阈值降低会让更多错误预测直接进入业务 Agent，而不是 fallback。

## 最值得先看的源码

如果只读五处代码，按以下顺序：

1. [`reward_penalty_loss()`](../src/lazy_agent_router/training/trainer.py#L86)：业务目标如何进入梯度；
2. [`RewardPenaltyTrainer`](../src/lazy_agent_router/training/trainer.py#L152)：训练、验证、早停和最佳权重；
3. [`train_macbert()`](../src/lazy_agent_router/training/trainer.py#L277)：完整训练流水线；
4. [`TrainingJob`](../src/lazy_agent_router/serving/training_jobs.py#L17)：API、后台线程和进度；
5. [`evaluate_uploaded()`](../src/lazy_agent_router/training/evaluate_uploaded.py#L60)：独立评测和发布门槛。

其余辅助入口：

- [`DatasetRegistry`](../src/lazy_agent_router/serving/dataset_registry.py#L30)：上传数据集校验与管理；
- [`write_deduplicated_dataset()`](../src/lazy_agent_router/training/prepare_dataset.py#L41)：组合并去重训练语料；
- [`TrainingParameters`](../src/lazy_agent_router/serving/schemas.py#L23)：动态参数范围；
- [`start_training()`](../src/lazy_agent_router/serving/routes.py#L56)：HTTP 训练入口。

## 核心要点总结

- 数据质量和边界覆盖通常比换更大的基础模型更重要；
- 训练目标要与业务目标一致：错 Agent 比同 Agent 内错 intent 更严重；
- 动态奖惩只用于训练，验证保持标准交叉熵；
- 最终保存最佳 epoch，不保存最后 epoch；
- 固定种子、分层切分和零重叠挑战集是可信比较的基础；
- 先分析错误，再补数据或改损失，不能只追训练集准确率；
- 线上还应持续保存低置信和人工纠正样本，形成下一轮真实训练数据。
