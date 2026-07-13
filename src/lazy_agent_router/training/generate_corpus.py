"""Generate a balanced, deterministic seed corpus for intent fine-tuning."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path


INTENT_TEMPLATES: dict[str, tuple[list[str], list[str]]] = {
    "workflow.query": (
        ["帮我查一下", "我要查看", "请查询", "能否告诉我", "麻烦获取", "我想了解"],
        ["采购申请的审批进度", "差旅报销流程", "合同审批状态", "入职申请流程", "请假单的处理情况", "订单审批记录", "付款申请的当前节点", "我的待办审批"],
    ),
    "workflow.start": (
        ["帮我发起", "我要提交", "请创建", "我需要申请", "现在新建", "协助我提交"],
        ["采购申请", "差旅报销申请", "用印申请", "请假流程", "合同审批", "付款流程", "入职申请", "费用报销单"],
    ),
    "workflow.approve": (
        ["请帮我", "我要", "现在", "麻烦", "请", "协助我"],
        ["审批这份采购申请", "批准待处理的报销单", "驳回这张付款申请", "同意合同审批", "处理我的待审批任务", "审核这份用印申请", "确认该请假单", "拒绝异常报销"],
    ),
    "knowledge.search": (
        ["帮我搜索", "请查找", "我要查看", "从知识库找", "帮我检索", "能否查询"],
        ["采购管理制度", "差旅报销规定", "员工手册", "合同归档规范", "信息安全制度", "请假政策", "供应商准入文档", "费用报销说明"],
    ),
}


def build_rows() -> list[dict[str, str]]:
    """Return 48 varied utterances per configured intent."""
    rows: list[dict[str, str]] = []
    for intent, (verbs, objects) in INTENT_TEMPLATES.items():
        for verb, obj in product(verbs, objects):
            rows.append({"text": f"{verb}{obj}", "intent": intent})
    return rows


def write_corpus(output: str | Path) -> int:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate intent-classification JSONL corpus")
    parser.add_argument("--output", default="datasets/raw/intents.jsonl")
    args = parser.parse_args()
    print(f"Generated {write_corpus(args.output)} rows in {args.output}")


if __name__ == "__main__":
    main()
