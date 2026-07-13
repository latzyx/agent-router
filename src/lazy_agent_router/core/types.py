from dataclasses import dataclass, field
from typing import Any


@dataclass
class IntentCandidate:
    intent: str
    confidence: float


@dataclass
class RouterResult:

    query: str

    intent: str

    confidence: float

    agent: str

    decision: str = "direct_route"

    risk_level: str = "low"

    candidate_intents: list[IntentCandidate] = field(
        default_factory=list
    )

    entities: dict[str, Any] = field(
        default_factory=dict
    )

    tool_plan: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of a route decision."""
        return {
            "query": self.query,
            "intent": self.intent,
            "confidence": self.confidence,
            "agent": self.agent,
            "decision": self.decision,
            "risk_level": self.risk_level,
            "candidate_intents": [
                {"intent": item.intent, "confidence": item.confidence}
                for item in self.candidate_intents
            ],
            "entities": self.entities,
            "tool_plan": self.tool_plan,
        }
