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

HARD_BOUNDARY_EXAMPLES: dict[str, list[str]] = {
    "workflow.query": [
        "查询待审批任务的状态", "看看合同审批到了哪一步", "我的请假申请现在什么状态",
        "查找付款流程的当前节点", "显示报销单审批进度", "获取采购申请处理记录",
        "查看入职流程是否完成", "待办审批还有几条", "订单审批什么时候结束",
        "查询用印申请的办理情况", "跟踪费用报销单进度", "查询合同是否已经审批",
        "我想知道申请是否被处理", "查一下流程卡在哪个节点", "查看审批历史记录", "审批进度帮我查一下",
        "合同审批流程停在哪里了", "流程为什么不动了，查下当前状态", "这张单子卡在什么环节", "看看申请目前的流转节点",
        "工号 C-302 的申请进度怎么样", "员工 D-410 的单据走到哪里了", "查看带工号的审批记录", "查询员工申请的处理状态",
        "合同流转停住了，请查询目前节点", "合同单据堵在流程里，查一下状态", "合同审批没有动，看看走到哪里", "这份合同卡在办理环节了",
        "合同申请停滞多久了", "流程挂起后当前状态是什么", "查下卡住的合同流转记录", "合同处理进度需要查询",
    ],
    "workflow.start": [
        "发起一份新的采购审批申请", "新建差旅报销流程", "提交我的请假申请", "创建合同审批单",
        "我要申请用印", "开始办理付款申请", "帮我开一个入职流程", "新增一张费用报销单",
        "现在申请采购流程", "建立新的合同申请", "我要提交报销材料", "新开一个请假单",
        "发起员工入职审批", "创建付款流程申请", "提交用印流程", "我要办理新的业务申请",
        "我要开一张采购申请单", "新增一笔付款流程", "为新员工建立入职申请", "提交新的合同流程",
        "为工号 C-302 新建报销申请", "替员工 D-410 发起请假流程", "创建带员工编号的入职单", "给新员工提交合同申请",
        "新建合同办理申请", "发起年度采购流程", "提交一份员工转正申请", "创建供应商入库申请",
        "我要开通新的付款审批", "办理新的出差申请单", "为团队建立用印流程", "提交合同盖章申请",
    ],
    "workflow.approve": [
        "批准这份采购审批单", "驳回待处理的差旅报销", "同意员工的请假申请", "审核合同审批任务",
        "处理这条待办审批", "拒绝这笔付款申请", "确认用印申请", "通过入职审批单",
        "审批费用报销", "请把这个流程驳回", "执行合同的审批操作", "批准员工 A-100 的请假单",
        "审核待办中的采购单", "同意这份付款流程", "拒绝异常费用报销", "确认审批结果",
        "请处理这条合同待办", "把这份申请审批通过", "拒绝这张不合规单据", "确认待审批事项",
        "确认工号 C-302 的采购申请", "审核员工 D-410 的用印单", "批准带员工编号的请假申请", "请确认该员工的付款审批",
        "待办合同需要我审核处理", "请处理待办合同的审核任务", "合同待办请帮忙审批", "审核待办列表中的合同单",
        "待处理合同请确认通过", "把合同审核任务办一下", "我需要审批待办合同", "请审核这份等待处理的合同",
    ],
    "knowledge.search": [
        "查询公司的审批制度说明", "搜索采购流程管理规范", "查找差旅报销政策", "知识库里有没有合同归档规则",
        "获取信息安全制度文档", "查一下员工请假规定", "检索供应商准入要求", "寻找费用报销指南",
        "公司用印规范在哪里", "搜索入职管理手册", "查询付款制度说明", "找一下采购管理办法",
        "知识库中搜索审批操作规范", "查看合同管理制度", "检索员工行为准则", "我想了解报销政策",
        "供应商资质准入需要哪些资料", "找找供应商准入的申请材料说明", "查询合作方准入要求文档", "供应商入库规范在哪里看",
        "查找供应商审核制度", "检索合作方入库说明", "哪里能看供应商合规材料", "搜索企业采购知识文档",
        "搜索合同流转操作手册", "查阅企业合同制度说明", "找一下采购审批规范文档", "检索业务流程管理办法",
        "知识库里的合同审核规则", "查询付款管理政策", "找企业流程制度材料", "搜索员工业务操作指南",
    ],
}


def build_rows() -> list[dict[str, str]]:
    """Return balanced templates plus hard examples for intent boundaries."""
    rows: list[dict[str, str]] = []
    for intent, (verbs, objects) in INTENT_TEMPLATES.items():
        for verb, obj in product(verbs, objects):
            rows.append({"text": f"{verb}{obj}", "intent": intent})
        rows.extend({"text": text, "intent": intent} for text in HARD_BOUNDARY_EXAMPLES[intent])
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
