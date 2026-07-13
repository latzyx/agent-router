import json
from pathlib import Path


def load_jsonl(path: str | Path) -> list[dict[str, str]]:
    """Load ``text`` / ``intent`` records and validate the minimum schema."""
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("text"), str) or not isinstance(row.get("intent"), str):
                raise ValueError(f"invalid training row at line {line_number}")
            rows.append({"text": row["text"], "intent": row["intent"]})
    return rows


def labels_for(rows: list[dict[str, str]]) -> list[str]:
    return sorted({row["intent"] for row in rows})
