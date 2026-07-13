from pathlib import Path

from fastapi import FastAPI

from ..classifiers.keyword import KeywordIntentClassifier
from ..core.confidence import ConfidencePolicy
from ..core.policy import RiskPolicy
from ..core.registry import AgentRegistry
from ..core.router import LazyAgentRouter
from ..entities.regex import RegexEntityExtractor
from ..tools.matcher import ToolMatcher
from ..tools.registry import ToolRegistry
from ..utils.config import load_yaml
from .routes import build_router
from .training_jobs import TrainingJob


def create_app(config_dir: str | Path | None = None, router: LazyAgentRouter | None = None) -> FastAPI:
    """Create an API app. Pass a router to inject MacBERT or a test double."""
    app = FastAPI(title="lazy-agent-router", version="0.1.0")
    if router is None:
        root = Path(config_dir) if config_dir else Path(__file__).resolve().parents[3] / "configs"
        intents = load_yaml(root / "intents.yaml")
        router_config = load_yaml(root / "router.yaml")
        classifier = KeywordIntentClassifier({
            name: item.get("keywords", []) for name, item in intents.get("intents", {}).items()
        })
        router = LazyAgentRouter(
            classifier=classifier,
            registry=AgentRegistry(intents),
            entity_extractor=RegexEntityExtractor(),
            tool_matcher=ToolMatcher(ToolRegistry(root / "tools.yaml")),
            confidence_policy=ConfidencePolicy.from_config(router_config),
            risk_policy=RiskPolicy.from_config(load_yaml(root / "risk_policy.yaml")),
        )
    app.state.lazy_router = router
    app.state.training_job = TrainingJob(Path(__file__).resolve().parents[3])
    app.include_router(build_router())
    return app


app = create_app()


def main() -> None:
    """Console entry point for local/WSL development."""
    import uvicorn
    uvicorn.run("lazy_agent_router.serving.app:app", host="0.0.0.0", port=8000)
