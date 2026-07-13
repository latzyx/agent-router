"""发现并按需加载本地训练完成的路由模型。"""
from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock

from ..classifiers.macbert import MacBERTClassifier
from ..core.confidence import ConfidencePolicy
from ..core.policy import RiskPolicy
from ..core.registry import AgentRegistry
from ..core.router import LazyAgentRouter
from ..entities.regex import RegexEntityExtractor
from ..tools.matcher import ToolMatcher
from ..tools.registry import ToolRegistry
from ..utils.config import load_yaml
from ..utils.device import resolve_device


class ModelRegistry:
    """提供模型选择列表，并在当前进程内缓存已加载的路由器。"""

    def __init__(self, project_root: Path, default_router: LazyAgentRouter):
        self.project_root = project_root
        self.default_router = default_router
        self.config_root = project_root / "configs"
        self._cache = {"keyword-default": default_router}
        self._lock = Lock()

    def choices(self) -> list[dict[str, str]]:
        # 这里只扫描模型元数据；列出模型时不加载权重，也不会占用 CUDA 显存。
        choices = [{"id": "keyword-default", "label": "关键词基线（无需模型）", "type": "keyword"}]
        models_root = self.project_root / "models"
        if models_root.is_dir():
            for path in sorted(models_root.iterdir()):
                if path.is_dir() and (path / "config.json").is_file():
                    choices.append({"id": path.name, "label": path.name, "type": "macbert"})
        return choices

    def get(self, model_id: str | None) -> LazyAgentRouter:
        selected = (model_id or "keyword-default").strip()
        if selected == "keyword-default":
            return self.default_router
        model_path = self.project_root / "models" / selected
        # 模型 ID 直接来自 API 请求，因此必须阻止目录穿越。
        if Path(selected).name != selected or not (model_path / "config.json").is_file():
            raise ValueError("指定模型不存在或不在 models 目录内")
        # Transformers 模型加载成本较高；进程级缓存可以让后续请求复用权重和显存。
        with self._lock:
            if selected not in self._cache:
                metadata = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
                labels = set(metadata.get("id2label", {}).values())
                config_name = "intents_uploaded.yaml" if "sap_interface" in labels else "intents.yaml"
                intents = load_yaml(self.config_root / config_name)
                self._cache[selected] = LazyAgentRouter(
                    classifier=MacBERTClassifier(
                        model_path,
                        device=resolve_device(os.getenv("LAZY_AGENT_ROUTER_DEVICE", "auto")),
                    ),
                    registry=AgentRegistry(intents),
                    entity_extractor=RegexEntityExtractor(),
                    tool_matcher=ToolMatcher(ToolRegistry(self.config_root / "tools.yaml")),
                    confidence_policy=ConfidencePolicy.from_config(load_yaml(self.config_root / "router.yaml")),
                    risk_policy=RiskPolicy.from_config(load_yaml(self.config_root / "risk_policy.yaml")),
                )
            return self._cache[selected]
