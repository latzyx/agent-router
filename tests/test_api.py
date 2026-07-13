from fastapi.testclient import TestClient

from lazy_agent_router.serving.app import create_app


def test_health_and_route_endpoint():
    client = TestClient(create_app())
    assert client.get("/health").json() == {"status": "ok"}
    response = client.post("/v1/route", json={"query": "帮我查询采购审批流程"})
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "workflow.query"
    assert data["agent"] == "workflow-agent"


def test_console_and_training_status_are_available():
    client = TestClient(create_app())
    page = client.get("/").text
    assert "Lazy Agent Router 控制台" in page
    assert "Hugging Face 模型" in page
    assert 'id="dataset"' in page
    assert client.get("/v1/training/status").json()["state"] == "idle"
