from typing import Any

from pydantic import BaseModel, Field


class RouteRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4096, description="User's natural-language request")
    model: str | None = Field(default=None, description="Model id returned by GET /v1/models")


class RouteResponse(BaseModel):
    query: str
    intent: str
    confidence: float
    agent: str
    decision: str
    risk_level: str
    candidate_intents: list[dict[str, Any]]
    entities: dict[str, Any]
    tool_plan: list[dict[str, Any]]
