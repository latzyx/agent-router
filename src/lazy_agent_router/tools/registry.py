from pathlib import Path
from typing import Any

import yaml

from .schema import ToolDefinition


class ToolRegistry:
    def __init__(self, config_file: str | Path | dict[str, Any]):
        if isinstance(config_file, dict):
            config = config_file
        else:
            with open(config_file, encoding="utf-8") as handle:
                config = yaml.safe_load(handle) or {}
        raw_tools = config.get("tools", config if isinstance(config, list) else [])
        self.tools = [
            ToolDefinition(
                name=item["name"],
                intents=tuple(item.get("intents", [])),
                description=item.get("description", ""),
                required_entities=tuple(item.get("required_entities", [])),
                arguments=dict(item.get("arguments", {})),
            )
            for item in raw_tools
        ]

    def for_intent(self, intent: str) -> list[ToolDefinition]:
        return [tool for tool in self.tools if intent in tool.intents]
