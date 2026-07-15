"""清洗上传的 JSONL 文件，生成可复现的去重训练集。"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def load_uploaded_rows(directory: str | Path) -> list[dict[str, str]]:
    """加载全部上传文件，每个文本与意图组合只保留一条。"""
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    labels_by_text: dict[str, str] = {}
    paths = sorted(Path(path) for path in glob.glob(str(Path(directory) / "*.jsonl")))
    if not paths:
        raise FileNotFoundError(f"no JSONL files found in {directory}")
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                text, intent = row.get("text"), row.get("intent")
                if not isinstance(text, str) or not text.strip() or not isinstance(intent, str) or not intent.strip():
                    raise ValueError(f"invalid text/intent at {path}:{line_number}")
                # 同一文本对应两个标签属于标注冲突，不能静默选择其中一个。
                previous = labels_by_text.setdefault(text, intent)
                if previous != intent:
                    raise ValueError(f"conflicting labels for text at {path}:{line_number}")
                key = (text, intent)
                # 多次上传可能包含完全相同的语料；去重可避免重复样本主导梯度。
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"text": text, "intent": intent})
    return rows


def write_deduplicated_dataset(
    source_dir: str | Path,
    output: str | Path,
    extra_files: list[str | Path] | None = None,
) -> int:
    rows = load_uploaded_rows(source_dir)
    # 额外人工语料与上传数据使用相同的去重和冲突检查逻辑。
    labels_by_text = {row["text"]: row["intent"] for row in rows}
    for extra_file in extra_files or []:
        with Path(extra_file).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                text, intent = row.get("text"), row.get("intent")
                if not isinstance(text, str) or not isinstance(intent, str) or not text.strip() or not intent.strip():
                    raise ValueError(f"invalid text/intent at {extra_file}:{line_number}")
                text, intent = text.strip(), intent.strip()
                previous = labels_by_text.setdefault(text, intent)
                if previous != intent:
                    raise ValueError(f"conflicting labels for text at {extra_file}:{line_number}")
                if previous == intent and not any(item["text"] == text for item in rows):
                    rows.append({"text": text, "intent": intent})
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deduplicate uploaded intent JSONL files")
    parser.add_argument("--source", default="datasets/uploads")
    parser.add_argument("--output", default="datasets/processed/uploaded_intents.jsonl")
    parser.add_argument("--extra", action="append", default=[], help="追加人工标注 JSONL，可重复传入")
    args = parser.parse_args()
    print(f"Prepared {write_deduplicated_dataset(args.source, args.output, args.extra)} rows in {args.output}")


if __name__ == "__main__":
    main()
