"""控制台上传数据集的持久化、校验和元数据管理。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from ..training.dataset import labels_for, load_jsonl


def _validate_and_deduplicate(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    unique: list[dict[str, str]] = []
    labels_by_text: dict[str, str] = {}
    seen: set[tuple[str, str]] = set()
    for row in rows:
        text, intent = row["text"].strip(), row["intent"].strip()
        if not text or not intent:
            raise ValueError("训练文本和意图标签不能为空")
        previous = labels_by_text.setdefault(text, intent)
        if previous != intent:
            raise ValueError(f"同一文本存在冲突标签：{text!r}")
        if (text, intent) not in seen:
            seen.add((text, intent))
            unique.append({"text": text, "intent": intent})
    return unique


class DatasetRegistry:
    MAX_UPLOAD_BYTES = 20 * 1024 * 1024

    def __init__(self, project_root: Path) -> None:
        self.root = project_root / "datasets" / "managed"

    def choices(self) -> list[dict]:
        if not self.root.is_dir():
            return []
        datasets = []
        for metadata_file in sorted(self.root.glob("*.json"), reverse=True):
            try:
                metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if (self.root / f"{metadata.get('id')}.jsonl").is_file():
                datasets.append(metadata)
        return datasets

    def resolve(self, dataset_id: str) -> Path:
        if not dataset_id or Path(dataset_id).name != dataset_id:
            raise ValueError("数据集 ID 无效")
        path = self.root / f"{dataset_id}.jsonl"
        if not path.is_file():
            raise ValueError("指定数据集不存在")
        return path

    def save(self, *, filename: str, content: bytes) -> dict:
        if not filename.lower().endswith(".jsonl"):
            raise ValueError("训练数据集必须是 .jsonl 文件")
        if not content:
            raise ValueError("上传的数据集为空")
        if len(content) > self.MAX_UPLOAD_BYTES:
            raise ValueError("数据集不能超过 20 MB")

        self.root.mkdir(parents=True, exist_ok=True)
        dataset_id = uuid4().hex
        path = self.root / f"{dataset_id}.jsonl"
        path.write_bytes(content)
        try:
            rows = _validate_and_deduplicate(load_jsonl(path))
            if not rows:
                raise ValueError("数据集没有有效记录")
            labels = labels_for(rows)
            if any(sum(row["intent"] == label for row in rows) < 2 for label in labels):
                raise ValueError("每个意图至少需要两条样本，才能分层切分训练集和验证集")
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            path.unlink(missing_ok=True)
            raise ValueError(f"数据集校验失败：{exc}") from exc

        metadata = {
            "id": dataset_id,
            "name": Path(filename).name,
            "rows": len(rows),
            "intents": len(labels),
            "labels": labels,
            "size_bytes": len(content),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (self.root / f"{dataset_id}.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return metadata
