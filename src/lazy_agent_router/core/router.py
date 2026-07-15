from .confidence import ConfidencePolicy
from .policy import RiskPolicy
from .types import IntentCandidate, RouterResult


class LazyAgentRouter:

    def __init__(
            self,
            classifier,
            registry,
            entity_extractor=None,
            tool_matcher=None,
            confidence_policy: ConfidencePolicy | None = None,
            risk_policy: RiskPolicy | None = None,
    ):
        self.classifier = classifier

        self.registry = registry
        self.entity_extractor = entity_extractor
        self.tool_matcher = tool_matcher
        self.confidence_policy = confidence_policy or ConfidencePolicy()
        self.risk_policy = risk_policy or RiskPolicy()

    def predict(
            self,
            query: str
    ) -> RouterResult:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        return self._result_for(query, self.classifier.predict(query))

    def predict_batch(self, queries: list[str]) -> list[RouterResult]:
        """批量执行分类，并为每条查询独立应用 Agent、风险和工具策略。"""
        if not queries or any(not isinstance(query, str) or not query.strip() for query in queries):
            raise ValueError("queries must contain non-empty strings")
        predictions = self.classifier.predict_batch(queries)
        if len(predictions) != len(queries):
            raise ValueError("classifier returned an unexpected batch size")
        return [
            self._result_for(query, prediction)
            for query, prediction in zip(queries, predictions, strict=True)
        ]

    def _result_for(self, query: str, intent_result) -> RouterResult:

        route = (
            self.registry
            .get_agent(
                intent_result["intent"]
            )
        )

        confidence = float(intent_result["confidence"])
        decision = self.confidence_policy.decision_for(confidence)
        if decision == "fallback":
            route = self.registry.get_agent("unknown")

        entities = self.entity_extractor.extract(query) if self.entity_extractor else {}
        tool_plan = self.tool_matcher.match(intent_result["intent"], entities) if self.tool_matcher else []
        if self.risk_policy.requires_confirmation(route["risk_level"]):
            decision = "confirmation_required"

        return RouterResult(

            query=query,

            intent=intent_result["intent"],

            confidence=confidence,

            agent=route["agent"],

            decision=decision,
            risk_level=route["risk_level"],
            candidate_intents=[IntentCandidate(intent=intent_result["intent"], confidence=confidence)],
            entities=entities,
            tool_plan=tool_plan,

        )

    def route(self, query: str) -> RouterResult:
        """Compatibility alias for the public API described in the README."""
        return self.predict(query)
