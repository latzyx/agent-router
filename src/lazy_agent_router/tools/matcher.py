from typing import Any

from .registry import ToolRegistry


class ToolMatcher:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def match(self, intent: str, entities: dict[str, Any]) -> list[dict[str, Any]]:
        plan = []
        for tool in self.registry.for_intent(intent):
            missing = [key for key in tool.required_entities if key not in entities]
            plan.append({
                "name": tool.name,
                "arguments": {**tool.arguments, **{key: entities[key] for key in tool.required_entities if key in entities}},
                "ready": not missing,
                "missing_entities": missing,
            })
        return plan
