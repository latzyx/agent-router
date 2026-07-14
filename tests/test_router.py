from lazy_agent_router.classifiers.keyword import KeywordIntentClassifier
from lazy_agent_router.core.registry import AgentRegistry
from lazy_agent_router.core.router import LazyAgentRouter
from lazy_agent_router.entities.regex import RegexEntityExtractor


def make_router() -> LazyAgentRouter:
    return LazyAgentRouter(
        KeywordIntentClassifier({"workflow.approve": ["审批"], "workflow.query": ["查询"]}),
        AgentRegistry({
            "fallback_agent": "fallback-agent",
            "intents": {
                "workflow.approve": {"agent": "workflow-agent", "risk_level": "high"},
                "workflow.query": {"agent": "workflow-agent", "risk_level": "low"},
            },
        }),
        entity_extractor=RegexEntityExtractor(),
    )


def test_routes_known_intent_and_extracts_employee_id():
    result = make_router().predict("请审批员工 A-100 的申请")
    assert result.intent == "workflow.approve"
    assert result.agent == "workflow-agent"
    assert result.decision == "confirmation_required"
    assert result.entities["employee_id"] == "A-100"


def test_unknown_intent_uses_fallback():
    result = make_router().predict("今天天气好吗")
    assert result.intent == "unknown"
    assert result.agent == "fallback-agent"
    assert result.decision == "fallback"


def test_predict_batch_preserves_order_and_applies_policy_per_query():
    results = make_router().predict_batch(["查询流程", "审批员工 A-100", "今天天气好吗"])

    assert [result.intent for result in results] == [
        "workflow.query",
        "workflow.approve",
        "unknown",
    ]
    assert results[1].decision == "confirmation_required"
    assert results[1].entities["employee_id"] == "A-100"
    assert results[2].agent == "fallback-agent"
