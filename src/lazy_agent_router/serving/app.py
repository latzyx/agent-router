import os
from contextlib import asynccontextmanager
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
from .model_registry import ModelRegistry
from .dataset_registry import DatasetRegistry
from .inference_batcher import InferenceBatcher


def create_app(config_dir: str | Path | None = None, router: LazyAgentRouter | None = None) -> FastAPI:
    """Create an API app. Pass a router to inject MacBERT or a test double."""
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        preload_model = os.getenv("LAZY_ROUTER_PRELOAD_MODEL", "").strip()
        if preload_model:
            # lifespan 启动完成前端口尚未接流量；同步加载可以确保 readiness
            # 只在模型和 CUDA 预热完成后成功，同时避免跨线程初始化 CUDA context。
            preload_router = app.state.model_registry.get(preload_model)
            preload_router.predict_batch(
                ["服务启动预热"] * app.state.inference_batcher.max_batch_size
            )
        try:
            yield
        finally:
            app.state.inference_batcher.close()

    app = FastAPI(title="lazy-agent-router", version="0.1.0", lifespan=lifespan)
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
    app.state.model_registry = ModelRegistry(Path(__file__).resolve().parents[3], router)
    project_root = Path(__file__).resolve().parents[3]
    app.state.dataset_registry = DatasetRegistry(project_root)
    app.state.training_job = TrainingJob(project_root)
    app.state.inference_batcher = InferenceBatcher(
        app.state.model_registry.get,
        max_batch_size=int(os.getenv("LAZY_ROUTER_MAX_BATCH_SIZE", "64")),
        max_wait_ms=float(os.getenv("LAZY_ROUTER_MAX_WAIT_MS", "3")),
        queue_size=int(os.getenv("LAZY_ROUTER_QUEUE_SIZE", "4096")),
        request_timeout_ms=float(os.getenv("LAZY_ROUTER_REQUEST_TIMEOUT_MS", "500")),
    )
    app.include_router(build_router())
    return app


app = create_app()


def main() -> None:
    """Console entry point for local/WSL development."""
    import uvicorn
    uvicorn.run("lazy_agent_router.serving.app:app", host="0.0.0.0", port=8000)
