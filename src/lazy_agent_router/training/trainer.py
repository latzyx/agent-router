"""MacBERT 意图分类训练核心。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from random import Random
from typing import Any

import torch
from datasets import Dataset
from torch.nn import functional as F
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from .dataset import labels_for, load_jsonl
from ..utils.device import resolve_device


def deduplicate_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """规范化文本、删除重复记录，并拒绝同一文本对应多个标签。

    为什么训练入口还要再次去重：上传接口可能多次保存同一份数据，调用方也可能
    绕过 ``prepare_dataset`` 直接传入原始 JSONL。如果重复样本不清理，它们会在
    梯度中被重复计算，相当于人为提高这些文本的权重。

    同一文本出现不同标签时不能随机保留一个，因为模型收到的是互相矛盾的监督
    信号。这类问题必须回到人工标注阶段处理，所以这里直接抛出异常。
    """
    unique: list[dict[str, str]] = []
    labels_by_text: dict[str, str] = {}
    seen: set[tuple[str, str]] = set()
    for row in rows:
        text = row["text"].strip()
        intent = row["intent"].strip()
        if not text or not intent:
            raise ValueError("训练文本和意图标签不能为空")
        previous = labels_by_text.setdefault(text, intent)
        if previous != intent:
            raise ValueError(f"同一文本存在冲突标签：{text!r}")
        key = (text, intent)
        if key not in seen:
            seen.add(key)
            unique.append({"text": text, "intent": intent})
    return unique


def stratified_split(
    rows: list[dict[str, str]], validation_split: float, seed: int
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """按意图分别切分，确保每个类别都进入训练集和验证集。

    普通随机切分在小数据集上可能把某个低频意图全部分进训练集，导致验证指标
    完全看不到该类别。分层切分先按 intent 分组，再在每组内部抽取验证样本。
    ``seed`` 固定后，每次训练得到相同的切分，模型版本之间才可以公平比较。
    """
    if not 0 < validation_split < 1:
        raise ValueError("validation_split 必须在 0 和 1 之间")
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["intent"], []).append(row)

    train_rows: list[dict[str, str]] = []
    validation_rows: list[dict[str, str]] = []
    random = Random(seed)
    for intent, intent_rows in grouped.items():
        if len(intent_rows) < 2:
            raise ValueError(f"意图 {intent!r} 至少需要两条样本")
        shuffled = intent_rows.copy()
        random.shuffle(shuffled)
        validation_size = min(len(shuffled) - 1, max(1, round(len(shuffled) * validation_split)))
        validation_rows.extend(shuffled[:validation_size])
        train_rows.extend(shuffled[validation_size:])
    random.shuffle(train_rows)
    random.shuffle(validation_rows)
    return train_rows, validation_rows


def reward_penalty_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    reward_strength: float = 0.20,
    penalty_strength: float = 0.75,
) -> torch.Tensor:
    """训练阶段使用的置信度奖励/惩罚损失。

    标准交叉熵始终作为基础损失。正确且自信的样本最多获得有限折扣；
    错误且自信的样本会被额外放大。权重不参与梯度计算，避免模型通过
    操纵权重本身降低损失。
    """
    if not 0 <= reward_strength < 1:
        raise ValueError("reward_strength 必须在 [0, 1) 范围内")
    if penalty_strength < 0:
        raise ValueError("penalty_strength 不能为负数")

    # reduction="none" 保留每个样本的独立损失，后面才能逐样本施加奖惩权重。
    per_example = F.cross_entropy(logits, labels, reduction="none")
    # float() 避免 FP16 下 softmax 因数值范围较小产生溢出或精度损失。
    probabilities = torch.softmax(logits.detach().float(), dim=-1)
    confidence, predictions = probabilities.max(dim=-1)
    correct = predictions.eq(labels)
    # 正确预测：置信度越高，权重越低；但最低限制为 0.5，不能让基础监督消失。
    reward_weight = (1 - reward_strength * confidence).clamp_min(0.5)
    # 错误预测：越自信说明错误越严重，因此提高该样本的梯度贡献。
    penalty_weight = 1 + penalty_strength * confidence
    weights = torch.where(correct, reward_weight, penalty_weight)
    return (per_example * weights.to(per_example.dtype)).mean()


def classification_metrics(eval_prediction: Any) -> dict[str, float]:
    """计算准确率和宏平均 F1，避免只根据 loss 判断模型质量。

    accuracy 反映全部样本中预测正确的比例，但在类别不平衡时容易被大类别掩盖。
    macro_f1 先分别计算每个意图的 F1，再对所有意图等权平均，因此小类别同样重要。
    """
    logits, labels = eval_prediction
    if isinstance(logits, tuple):
        logits = logits[0]
    predictions = logits.argmax(axis=-1)
    accuracy = float((predictions == labels).mean())
    f1_scores: list[float] = []
    for label in sorted(set(labels.tolist())):
        true_positive = int(((predictions == label) & (labels == label)).sum())
        false_positive = int(((predictions == label) & (labels != label)).sum())
        false_negative = int(((predictions != label) & (labels == label)).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        f1_scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return {"accuracy": accuracy, "macro_f1": sum(f1_scores) / len(f1_scores)}


class RewardPenaltyTrainer(Trainer):
    """训练使用奖励/惩罚损失，验证使用标准交叉熵。

    这是旧逻辑最关键的修复：如果验证阶段也使用奖惩权重，eval_loss 会随着模型
    当前预测结果改变权重，导致不同 epoch 的 loss 不在同一尺度上。现在只有
    ``model.training == True`` 时使用奖惩，验证和模型选择始终使用标准交叉熵。
    """

    def __init__(
        self,
        *args,
        reward_strength: float = 0.20,
        penalty_strength: float = 0.75,
        early_stopping_patience: int = 2,
        early_stopping_min_delta: float = 1e-5,
        **kwargs,
    ):
        if early_stopping_patience < 1:
            raise ValueError("early_stopping_patience 至少为 1")
        if early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta 不能为负数")
        super().__init__(*args, **kwargs)
        self.reward_strength = reward_strength
        self.penalty_strength = penalty_strength
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.best_macro_f1 = -1.0
        self.best_eval_loss = float("inf")
        # 早停基准与最佳权重基准必须分开：微小改善仍应保存为最佳权重，
        # 但不足 min_delta 时不应让训练无限重置耐心计数。
        self.stopping_macro_f1 = -1.0
        self.stopping_eval_loss = float("inf")
        self.best_model_state: dict[str, torch.Tensor] | None = None
        self.epochs_without_improvement = 0
        self.last_eval_metrics: dict[str, float] = {}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 不修改 Trainer 传入的 inputs，避免评估和梯度累积阶段复用数据时丢失标签。
        labels = inputs["labels"]
        model_inputs = {name: value for name, value in inputs.items() if name != "labels"}
        outputs = model(**model_inputs)
        if model.training:
            loss = reward_penalty_loss(
                outputs.logits,
                labels,
                reward_strength=self.reward_strength,
                penalty_strength=self.penalty_strength,
            )
        else:
            # 验证必须使用标准交叉熵，否则奖励权重会扭曲最佳模型选择。
            loss = F.cross_entropy(outputs.logits.float(), labels)
        return (loss, outputs) if return_outputs else loss

    def evaluate(self, *args, **kwargs):
        # super().evaluate 会切换 model.eval()，计算标准验证损失和 compute_metrics。
        metrics = super().evaluate(*args, **kwargs)
        self.last_eval_metrics = dict(metrics)
        macro_f1 = float(metrics.get("eval_macro_f1", 0.0))
        eval_loss = float(metrics.get("eval_loss", float("inf")))
        # 任何真实改善都更新最佳权重，确保最终恢复的是实际最优 epoch。
        f1_improved = macro_f1 > self.best_macro_f1 + 1e-8
        loss_improved = (
            abs(macro_f1 - self.best_macro_f1) <= 1e-8
            and eval_loss < self.best_eval_loss
        )
        if f1_improved or loss_improved:
            self.best_macro_f1 = macro_f1
            self.best_eval_loss = eval_loss
            # 只在指标改善时复制一次权重，避免每轮无条件占用额外内存。
            self.best_model_state = {
                name: value.detach().cpu().clone()
                for name, value in self.model.state_dict().items()
            }

        # 只有达到 min_delta 的实质改善才重置早停计数，过滤浮点数值抖动。
        significant_f1 = macro_f1 > self.stopping_macro_f1 + 1e-8
        significant_loss = (
            abs(macro_f1 - self.stopping_macro_f1) <= 1e-8
            and eval_loss < self.stopping_eval_loss - self.early_stopping_min_delta
        )
        if significant_f1 or significant_loss:
            self.stopping_macro_f1 = macro_f1
            self.stopping_eval_loss = eval_loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
            if self.epochs_without_improvement >= self.early_stopping_patience:
                # Trainer 会在当前 epoch 完成后安全停止，不会中断正在反向传播的批次。
                self.control.should_training_stop = True
        return metrics

    def restore_best_model(self) -> None:
        """恢复宏平均 F1 最优、交叉熵最低的模型权重。"""
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)


def train_macbert(
    train_file: str | Path,
    model_path: str | Path,
    output_dir: str | Path,
    *,
    validation_file: str | Path | None = None,
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 128,
    device: str = "auto",
    reward_strength: float = 0.20,
    penalty_strength: float = 0.75,
    validation_split: float = 0.2,
    early_stopping_patience: int = 2,
    early_stopping_min_delta: float = 1e-5,
    seed: int = 42,
) -> Path:
    """执行去重、分层验证、动态填充和早停的完整微调流程。

    参数说明：
    - ``train_file``：JSONL 训练集，每行至少包含 text 和 intent。
    - ``model_path``：本地 Transformers 模型目录或 Hugging Face 模型 ID。
    - ``output_dir``：最终模型、分词器和训练报告的保存目录。
    - ``validation_file``：可选的独立验证集；不提供时从训练集分层抽取。
    - ``epochs``：最多训练轮数；启用早停后可能提前结束。
    - ``batch_size``：单个设备每步处理的样本数，越大通常越占显存。
    - ``learning_rate``：参数更新步长；续训一般应小于从基础模型微调。
    - ``max_length``：分词后的最大 token 长度，超出部分会被截断。
    - ``reward_strength``：正确高置信预测的损失折扣强度。
    - ``penalty_strength``：错误高置信预测的损失放大强度。
    - ``early_stopping_patience``：连续多少轮无实质改善后停止。
    - ``early_stopping_min_delta``：验证 loss 至少下降多少才算改善。
    """
    # 第 1 步：读取、规范化并去重。即使调用方未运行预处理脚本也能安全训练。
    rows = deduplicate_rows(load_jsonl(train_file))
    if not rows:
        raise ValueError("训练数据集为空")

    # 第 2 步：准备验证集。独立验证集更可信；没有时才使用固定种子的分层切分。
    if validation_file is None:
        train_rows, validation_rows = stratified_split(rows, validation_split, seed)
    else:
        train_rows = rows
        validation_rows = deduplicate_rows(load_jsonl(validation_file))
        # 同一文本同时出现在训练集和验证集会造成数据泄漏，使指标虚高。
        overlap = {row["text"] for row in train_rows} & {row["text"] for row in validation_rows}
        if overlap:
            raise ValueError(f"训练集与验证集存在 {len(overlap)} 条文本泄漏")

    # 第 3 步：建立稳定的标签编号。排序保证相同数据每次获得相同 label2id。
    labels = labels_for(train_rows)
    validation_labels = set(labels_for(validation_rows))
    if not validation_labels.issubset(labels):
        raise ValueError("验证集包含训练集中不存在的意图标签")
    label_to_id = {label: index for index, label in enumerate(labels)}

    def as_dataset(source_rows: list[dict[str, str]]) -> Dataset:
        return Dataset.from_list([
            {"text": row["text"], "labels": label_to_id[row["intent"]]}
            for row in source_rows
        ])

    # 第 4 步：加载分词器。本地目录强制 local_files_only，避免服务意外访问网络。
    source = str(model_path)
    local_only = Path(source).is_dir()
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=local_only)

    def tokenize(batch):
        # 动态 padding 由 DataCollator 完成，这里只截断，减少短文本的显存浪费。
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    # batched=True 会批量调用分词器，比逐条分词更快。
    tokenized_train = as_dataset(train_rows).map(tokenize, batched=True, remove_columns=["text"])
    tokenized_validation = as_dataset(validation_rows).map(tokenize, batched=True, remove_columns=["text"])
    # 第 5 步：加载分类模型。若旧模型标签数不同，复用编码器并重建分类头。
    model = AutoModelForSequenceClassification.from_pretrained(
        source,
        local_files_only=local_only,
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id=label_to_id,
        ignore_mismatched_sizes=True,
    )

    # 第 6 步：配置优化器相关参数和硬件加速。
    output = Path(output_dir)
    selected_device = resolve_device(device)
    # 显式计算预热步数，兼容已弃用 warmup_ratio 的新版 Transformers。
    steps_per_epoch = math.ceil(len(tokenized_train) / batch_size)
    warmup_steps = max(1, round(steps_per_epoch * epochs * 0.1))
    arguments = TrainingArguments(
        output_dir=str(output),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        # 权重衰减可抑制参数过大，降低小数据集过拟合风险。
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        # 梯度裁剪防止偶发大梯度造成训练不稳定。
        max_grad_norm=1.0,
        eval_strategy="epoch",
        save_strategy="no",
        report_to=[],
        use_cpu=selected_device == "cpu",
        # CUDA 上启用半精度，减少显存并提升吞吐；CPU 保持 FP32。
        fp16=selected_device == "cuda",
        seed=seed,
    )
    # 第 7 步：组装训练器。DataCollatorWithPadding 只将每批补到该批最长文本，
    # 不再把所有短句固定补到 max_length。
    trainer = RewardPenaltyTrainer(
        model=model,
        args=arguments,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_validation,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        processing_class=tokenizer,
        compute_metrics=classification_metrics,
        reward_strength=reward_strength,
        penalty_strength=penalty_strength,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
    )
    train_result = trainer.train()
    trainer.restore_best_model()
    final_metrics = trainer.evaluate()
    # 第 8 步：保存最佳模型、分词器和可审计报告，而不是保存最后一轮权重。
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)

    # 保存可审计的训练报告，前端和后续模型比较可直接读取。
    report = {
        "train_file": str(train_file),
        "model_source": source,
        "device": selected_device,
        "validation_file": str(validation_file) if validation_file else None,
        "train_examples": len(train_rows),
        "validation_examples": len(validation_rows),
        "labels": labels,
        "epochs_requested": epochs,
        "epochs_completed": train_result.metrics.get("epoch"),
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "reward_strength": reward_strength,
        "penalty_strength": penalty_strength,
        "warmup_steps": warmup_steps,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "train_metrics": {
            key: float(value)
            for key, value in train_result.metrics.items()
            if isinstance(value, (int, float))
        },
        "metrics": {key: float(value) for key, value in final_metrics.items() if isinstance(value, (int, float))},
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output
