from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


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


class TrainingParameters(BaseModel):
    """可由 API/控制台安全调整的训练超参数。"""

    model_config = ConfigDict(extra="forbid")

    epochs: int = Field(default=5, ge=1, le=50)
    batch_size: int = Field(default=16, ge=1, le=128)
    learning_rate: float = Field(default=2e-5, ge=1e-7, le=1e-2)
    max_length: int = Field(default=128, ge=16, le=512)
    validation_split: float = Field(default=0.2, ge=0.05, le=0.5)
    reward_strength: float = Field(default=0.2, ge=0, lt=1)
    penalty_strength: float = Field(default=0.75, ge=0, le=5)
    cross_group_penalty_strength: float = Field(default=0.0, ge=0, le=5)
    early_stopping_patience: int = Field(default=2, ge=1, le=20)
    early_stopping_min_delta: float = Field(default=1e-5, ge=0, le=0.1)
    weight_decay: float = Field(default=0.01, ge=0, le=0.5)
    warmup_ratio: float = Field(default=0.1, ge=0, le=0.5)
    max_grad_norm: float = Field(default=1.0, gt=0, le=10)
    gradient_accumulation_steps: int = Field(default=1, ge=1, le=64)
    seed: int = Field(default=42, ge=0, le=2_147_483_647)
    device: Literal["auto", "cpu", "cuda"] = "auto"
