import asyncio
import json

import httpx

from lazy_agent_router.serving.app import create_app


class ASGIClient:
    """兼容同步 pytest 的轻量客户端，绕开已废弃的 Starlette TestClient。"""

    def __init__(self, app):
        self.app = app

    def request(self, method, path, **kwargs):
        async def send():
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await client.request(method, path, **kwargs)

        return asyncio.run(send())

    def get(self, path, **kwargs):
        return self.request("GET", path, **kwargs)

    def post(self, path, **kwargs):
        return self.request("POST", path, **kwargs)


def test_health_and_route_endpoint():
    client = ASGIClient(create_app())
    assert client.get("/health").json() == {"status": "ok"}
    response = client.post("/v1/route", json={"query": "帮我查询采购审批流程"})
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "workflow.query"
    assert data["agent"] == "workflow-agent"


def test_models_endpoint_and_route_accept_model_id():
    client = ASGIClient(create_app())
    models = client.get("/v1/models").json()["models"]
    ids = {item["id"] for item in models}
    assert "keyword-default" in ids
    response = client.post("/v1/route", json={"query": "帮我查询采购审批流程", "model": "keyword-default"})
    assert response.status_code == 200


def test_batch_route_reduces_http_requests_and_exposes_inference_stats():
    client = ASGIClient(create_app())
    try:
        response = client.post(
            "/v1/route/batch",
            json={
                "queries": ["帮我查询采购审批流程", "今天天气好吗", "请审批这个申请"],
                "model": "keyword-default",
            },
        )

        assert response.status_code == 200
        assert [item["intent"] for item in response.json()] == [
            "workflow.query",
            "unknown",
            "workflow.start",
        ]
        stats = client.get("/v1/inference/stats").json()
        assert stats["requests"] == 3
        assert stats["batches"] >= 1
        assert stats["max_observed_batch_size"] == 3
    finally:
        client.app.state.inference_batcher.close()


def test_lifespan_preloads_and_warms_configured_model(monkeypatch):
    app = create_app()
    captured = []

    class FakeRouter:
        def predict_batch(self, queries):
            captured.extend(queries)

    class FakeRegistry:
        def get(self, model_id):
            assert model_id == "production-model"
            return FakeRouter()

    app.state.model_registry = FakeRegistry()
    monkeypatch.setenv("LAZY_ROUTER_PRELOAD_MODEL", "production-model")

    async def run_lifespan():
        async with app.router.lifespan_context(app):
            pass

    asyncio.run(run_lifespan())

    assert captured == ["服务启动预热"] * 64


def test_console_and_training_status_are_available():
    client = ASGIClient(create_app())
    page = client.get("/").text
    assert "Lazy Agent Router 控制台" in page
    assert "Hugging Face 模型" in page
    assert 'id="dataset"' in page
    assert 'data-tab="datasets"' in page
    assert 'data-tab="testing"' in page
    assert 'data-tab="settings"' in page
    assert 'id="progress-bar"' in page
    assert 'role="progressbar"' in page
    assert client.get("/v1/training/status").json()["state"] == "idle"


def test_training_config_exposes_dynamic_defaults_and_limits():
    data = ASGIClient(create_app()).get("/v1/training/config").json()

    assert data["defaults"]["epochs"] == 5
    assert data["defaults"]["device"] == "auto"
    assert data["defaults"]["cross_group_penalty_strength"] == 0.0
    assert data["schema"]["properties"]["batch_size"]["maximum"] == 128


def test_dataset_upload_is_validated_and_listed(tmp_path):
    client = ASGIClient(create_app())
    client.app.state.dataset_registry.root = tmp_path / "managed"
    content = "\n".join([
        json.dumps({"text": "查询流程", "intent": "query"}, ensure_ascii=False),
        json.dumps({"text": "流程在哪里", "intent": "query"}, ensure_ascii=False),
    ])

    response = client.post(
        "/v1/datasets",
        files={"dataset": ("sample.jsonl", content.encode(), "application/jsonl")},
    )

    assert response.status_code == 200
    assert response.json()["rows"] == 2
    assert client.get("/v1/datasets").json()["datasets"][0]["name"] == "sample.jsonl"


def test_start_training_accepts_validated_dynamic_parameters(tmp_path):
    client = ASGIClient(create_app())
    captured = {}

    class FakeTrainingJob:
        def start(self, **kwargs):
            captured.update(kwargs)
            return {"state": "running", "message": "started"}

    client.app.state.training_job = FakeTrainingJob()
    response = client.post(
        "/v1/training/start",
        data={
            "model_source": "hfl/chinese-macbert-base",
            "parameters": json.dumps({"epochs": 9, "batch_size": 8, "device": "cpu"}),
        },
    )

    assert response.status_code == 200
    assert captured["parameters"].epochs == 9
    assert captured["parameters"].batch_size == 8
    assert captured["parameters"].learning_rate == 2e-5


def test_start_training_rejects_out_of_range_parameters():
    client = ASGIClient(create_app())
    response = client.post(
        "/v1/training/start",
        data={"parameters": json.dumps({"batch_size": 0})},
    )

    assert response.status_code == 422
    assert "batch_size" in response.json()["detail"]
