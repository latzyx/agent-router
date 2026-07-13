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
