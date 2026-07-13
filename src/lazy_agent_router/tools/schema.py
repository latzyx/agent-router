from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    intents: tuple[str, ...] = ()
    description: str = ""
    required_entities: tuple[str, ...] = ()
    arguments: dict[str, Any] = field(default_factory=dict)
