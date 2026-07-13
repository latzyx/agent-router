from collections.abc import Mapping


class RiskPolicy:
    """Map risk levels to whether a human confirmation is required."""

    def __init__(self, confirmation_levels: set[str] | None = None):
        self.confirmation_levels = confirmation_levels or {"high"}

    @classmethod
    def from_config(cls, config: Mapping | None) -> "RiskPolicy":
        levels = (config or {}).get("confirmation_required_for", ["high"])
        return cls(set(levels))

    def requires_confirmation(self, risk_level: str) -> bool:
        return risk_level in self.confirmation_levels
