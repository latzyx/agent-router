from pathlib import Path
from typing import Any

import yaml


class AgentRegistry:

    def __init__(
            self,
            config_file: str | Path | dict[str, Any]
    ):
        if isinstance(config_file, dict):
            self.config = config_file
        else:
            with open(config_file, encoding="utf-8") as f:
                self.config = yaml.safe_load(f) or {}

    def get_agent(
            self,
            intent: str
    ):
        item = (
            self.config
            .get("intents", {})
            .get(intent)
        )

        if not item:
            return {
                "agent":
                self.config.get("fallback_agent", "fallback-agent"),

                "risk_level":
                "unknown"
            }

        return {

            "agent":
                item.get("agent", self.config.get("fallback_agent", "fallback-agent")),

            "risk_level":
                item.get(
                    "risk_level",
                    "low"
                )
        }

    def get_intent_config(self, intent: str) -> dict[str, Any]:
        return dict(self.config.get("intents", {}).get(intent, {}))
