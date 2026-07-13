"""使用训练好的模型批量回归测试 uploads 目录中的 JSONL 数据。"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..core.registry import AgentRegistry
from ..utils.config import load_yaml
from ..utils.device import resolve_device


def load_uploaded_cases(source: str | Path) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """读取上传文件，同时记录每个文件的行数和哈希值。"""
    rows: list[dict[str, str]] = []
    files: list[dict[str, Any]] = []
    paths = sorted(Path(path) for path in glob.glob(str(Path(source) / "*.jsonl")))
    if not paths:
        raise FileNotFoundError(f"{source} 中没有 JSONL 文件")
    for path in paths:
        file_rows = 0
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                required = ("text", "intent", "agent")
                if any(not isinstance(row.get(field), str) or not row[field].strip() for field in required):
                    raise ValueError(f"{path}:{line_number} 缺少有效的 text/intent/agent")
                rows.append({field: row[field].strip() for field in required})
                file_rows += 1
        files.append({
            "name": path.name,
            "rows": file_rows,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    return rows, files


def _macro_f1(expected: list[str], predicted: list[str]) -> float:
    """按类别等权计算宏平均 F1。"""
    scores = []
    for label in sorted(set(expected)):
        true_positive = sum(e == label and p == label for e, p in zip(expected, predicted, strict=True))
        false_positive = sum(e != label and p == label for e, p in zip(expected, predicted, strict=True))
        false_negative = sum(e == label and p != label for e, p in zip(expected, predicted, strict=True))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return sum(scores) / len(scores)


def evaluate_uploaded(
    model_path: str | Path,
    source: str | Path,
    agent_config: str | Path,
    *,
    batch_size: int = 64,
    confidence_threshold: float = 0.6,
    device: str = "auto",
    training_reference: str | Path | None = None,
) -> dict[str, Any]:
    """批量评估意图、置信度和 Agent 映射，并返回 JSON 可序列化报告。"""
    rows, files = load_uploaded_cases(source)

    # 三份上传文件可能完全相同。推理时只计算唯一文本，再把结果映射回原始行，
    # 可以显著减少重复 GPU 计算，同时保留原始行级准确率统计。
    labels_by_text: dict[str, str] = {}
    agents_by_text: dict[str, str] = {}
    unique_texts: list[str] = []
    seen_texts: set[str] = set()
    for row in rows:
        text = row["text"]
        previous_label = labels_by_text.setdefault(text, row["intent"])
        previous_agent = agents_by_text.setdefault(text, row["agent"])
        if previous_label != row["intent"] or previous_agent != row["agent"]:
            raise ValueError(f"文本存在冲突标注：{text!r}")
        if text not in seen_texts:
            seen_texts.add(text)
            unique_texts.append(text)

    selected_device = resolve_device(device)
    model_source = str(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_source, local_files_only=True)
    model.to(selected_device)
    model.eval()

    predictions: dict[str, dict[str, Any]] = {}
    # 批量推理比逐条调用 predict 更快；padding 只补到当前批次最长文本。
    with torch.inference_mode():
        for start in range(0, len(unique_texts), batch_size):
            texts = unique_texts[start:start + batch_size]
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )
            inputs = {name: value.to(selected_device) for name, value in inputs.items()}
            probabilities = torch.softmax(model(**inputs).logits.float(), dim=-1)
            confidence, indices = probabilities.max(dim=-1)
            for text, index, score in zip(texts, indices.tolist(), confidence.tolist(), strict=True):
                predictions[text] = {
                    "intent": model.config.id2label[index],
                    "confidence": float(score),
                }

    registry = AgentRegistry(load_yaml(agent_config))
    expected_unique = [labels_by_text[text] for text in unique_texts]
    predicted_unique = [predictions[text]["intent"] for text in unique_texts]
    unique_correct = sum(e == p for e, p in zip(expected_unique, predicted_unique, strict=True))
    raw_correct = sum(row["intent"] == predictions[row["text"]]["intent"] for row in rows)
    agent_correct = sum(
        row["agent"] == registry.get_agent(predictions[row["text"]]["intent"])["agent"]
        for row in rows
    )

    per_intent: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[str]] = defaultdict(list)
    for text in unique_texts:
        grouped[labels_by_text[text]].append(text)
    for intent, texts in sorted(grouped.items()):
        correct = sum(predictions[text]["intent"] == intent for text in texts)
        confidences = [predictions[text]["confidence"] for text in texts]
        per_intent[intent] = {
            "support": len(texts),
            "correct": correct,
            "accuracy": correct / len(texts),
            "average_confidence": sum(confidences) / len(confidences),
            "minimum_confidence": min(confidences),
        }

    errors = [
        {
            "text": text,
            "expected": labels_by_text[text],
            "predicted": predictions[text]["intent"],
            "confidence": predictions[text]["confidence"],
        }
        for text in unique_texts
        if predictions[text]["intent"] != labels_by_text[text]
    ]
    low_confidence = [
        {
            "text": text,
            "expected": labels_by_text[text],
            **predictions[text],
        }
        for text in unique_texts
        if predictions[text]["confidence"] < confidence_threshold
    ]

    # 通过精确文本交集判断评测类型，避免把真正零重叠的挑战集误标为训练回归。
    training_overlap = 0
    if training_reference is not None and Path(training_reference).is_file():
        reference_texts = {
            json.loads(line)["text"].strip()
            for line in Path(training_reference).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        training_overlap = len(set(unique_texts) & reference_texts)
    if training_overlap == 0:
        evaluation_type = "independent_challenge"
        warning = "评测文本与训练参考集零重叠，可用于衡量未见表达的泛化能力。"
    elif training_overlap == len(unique_texts):
        evaluation_type = "training_source_regression"
        warning = "全部评测文本来自训练参考集，本结果只能作为回归测试。"
    else:
        evaluation_type = "partially_overlapping"
        warning = "评测集与训练参考集部分重叠，指标包含数据泄漏。"
    return {
        "evaluation_type": evaluation_type,
        "warning": warning,
        "model": model_source,
        "device": selected_device,
        "files": files,
        "raw_rows": len(rows),
        "unique_texts": len(unique_texts),
        "duplicate_rows": len(rows) - len(unique_texts),
        "training_overlap": training_overlap,
        "training_overlap_ratio": training_overlap / len(unique_texts),
        "unique_accuracy": unique_correct / len(unique_texts),
        "raw_accuracy": raw_correct / len(rows),
        "macro_f1": _macro_f1(expected_unique, predicted_unique),
        "agent_accuracy": agent_correct / len(rows),
        "confidence_threshold": confidence_threshold,
        "low_confidence_count": len(low_confidence),
        "per_intent": per_intent,
        "errors": errors,
        "low_confidence": low_confidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="批量回归测试 uploads 中的业务语料")
    parser.add_argument("--model", default="models/lazy-agent-router-macbert-v11")
    parser.add_argument("--source", default="datasets/uploads")
    parser.add_argument("--agent-config", default="configs/intents_uploaded.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--training-reference", default="datasets/processed/uploaded_intents.jsonl")
    args = parser.parse_args()
    report = evaluate_uploaded(
        args.model,
        args.source,
        args.agent_config,
        batch_size=args.batch_size,
        device=args.device,
        training_reference=args.training_reference,
    )
    output = Path(args.output) if args.output else Path(args.model) / "upload_evaluation.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "raw_rows": report["raw_rows"],
        "unique_texts": report["unique_texts"],
        "unique_accuracy": report["unique_accuracy"],
        "macro_f1": report["macro_f1"],
        "agent_accuracy": report["agent_accuracy"],
        "errors": len(report["errors"]),
        "low_confidence_count": report["low_confidence_count"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
