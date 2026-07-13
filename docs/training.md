# 训练逻辑说明

本文对应 `src/lazy_agent_router/training/trainer.py`，用于解释训练数据如何进入模型，以及每项优化解决什么问题。

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
