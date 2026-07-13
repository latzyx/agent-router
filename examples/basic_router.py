from lazy_agent_router import LazyAgentRouter

from lazy_agent_router.classifiers.macbert import (
    MacBERTClassifier
)

from lazy_agent_router.core.registry import (
    AgentRegistry
)

classifier = MacBERTClassifier(
    "/home/lazy/models/lazy-agent-router-macbert-v0.1"
)

registry = AgentRegistry(
    "configs/intents.yaml"
)

router = LazyAgentRouter(
    classifier,
    registry
)

result = router.predict(
    "帮我查询采购流程"
)

print(result)
