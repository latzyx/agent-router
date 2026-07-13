from collections.abc import Mapping


class ConfidencePolicy:
    """Normalise confidence thresholds used by the router."""

    def __init__(self, direct_route_threshold: float = 0.6):
        if not 0 <= direct_route_threshold <= 1:
            raise ValueError("direct_route_threshold must be between 0 and 1")
        self.direct_route_threshold = direct_route_threshold

    @classmethod
    def from_config(cls, config: Mapping | None) -> "ConfidencePolicy":
        return cls(float((config or {}).get("direct_route_threshold", 0.6)))

    def decision_for(self, confidence: float) -> str:
        return "direct_route" if confidence >= self.direct_route_threshold else "fallback"
