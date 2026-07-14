import asyncio
import json

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import ValidationError

from .inference_batcher import InferenceQueueFull
from .schemas import BatchRouteRequest, RouteRequest, RouteResponse, TrainingParameters
from .ui import console_page


def build_router() -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @router.get("/", include_in_schema=False)
    async def console():
        return console_page()

    @router.get("/v1/training/status")
    async def training_status(request: Request) -> dict:
        return request.app.state.training_job.snapshot()

    @router.get("/v1/training/config")
    async def training_config() -> dict:
        return {
            "defaults": TrainingParameters().model_dump(),
            "schema": TrainingParameters.model_json_schema(),
            "recommended_models": [
                {"id": "hfl/chinese-macbert-base", "label": "MacBERT Base（推荐）"},
                {"id": "hfl/chinese-roberta-wwm-ext", "label": "Chinese RoBERTa WWM"},
                {"id": "bert-base-chinese", "label": "BERT Base Chinese（基线对照）"},
            ],
        }

    @router.get("/v1/datasets")
    async def datasets(request: Request) -> dict:
        return {"datasets": request.app.state.dataset_registry.choices()}

    @router.post("/v1/datasets")
    async def upload_dataset(request: Request, dataset: UploadFile = File(...)) -> dict:
        try:
            content = await dataset.read(request.app.state.dataset_registry.MAX_UPLOAD_BYTES + 1)
            return request.app.state.dataset_registry.save(
                filename=dataset.filename or "dataset.jsonl", content=content
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.get("/v1/models")
    async def models(request: Request) -> dict:
        return {"models": request.app.state.model_registry.choices()}

    @router.get("/v1/inference/stats")
    async def inference_stats(request: Request) -> dict:
        return request.app.state.inference_batcher.snapshot()

    @router.post("/v1/training/start")
    async def start_training(
        request: Request,
        model_source: str = Form("hfl/chinese-macbert-base"),
        dataset_id: str | None = Form(None),
        parameters: str = Form("{}"),
        dataset: UploadFile | None = File(None),
    ) -> dict:
        try:
            parsed_parameters = TrainingParameters.model_validate(json.loads(parameters))
            dataset_file = (
                request.app.state.dataset_registry.resolve(dataset_id) if dataset_id else None
            )
            # 保留旧客户端“启动训练时直接上传”的调用方式，同时纳入统一数据集管理。
            if dataset and dataset.filename:
                metadata = request.app.state.dataset_registry.save(
                    filename=dataset.filename,
                    content=await dataset.read(request.app.state.dataset_registry.MAX_UPLOAD_BYTES + 1),
                )
                dataset_file = request.app.state.dataset_registry.resolve(metadata["id"])
        except (ValueError, ValidationError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=422, detail=f"训练配置无效：{exc}") from exc
        return request.app.state.training_job.start(
            model_source=model_source,
            dataset_file=dataset_file,
            parameters=parsed_parameters,
        )

    @router.post("/v1/route", response_model=RouteResponse)
    async def route(payload: RouteRequest, request: Request) -> dict:
        batcher = request.app.state.inference_batcher
        try:
            future = batcher.submit_async(payload.query, payload.model)
            result = await asyncio.wait_for(
                future,
                timeout=batcher.request_timeout_seconds,
            )
        except InferenceQueueFull as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except TimeoutError as exc:
            future.cancel()
            raise HTTPException(status_code=504, detail="路由推理超时") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return result.to_dict()

    @router.post("/v1/route/batch", response_model=list[RouteResponse])
    async def route_batch(payload: BatchRouteRequest, request: Request) -> list[dict]:
        """显式批量接口；调用方可用一次 HTTP 请求替代最多 256 次请求。"""
        if any(not query.strip() for query in payload.queries):
            raise HTTPException(status_code=422, detail="queries must contain non-empty strings")
        batcher = request.app.state.inference_batcher
        futures = []
        try:
            futures = [batcher.submit_async(query, payload.model) for query in payload.queries]
            results = await asyncio.wait_for(
                asyncio.gather(*futures),
                timeout=batcher.request_timeout_seconds,
            )
        except InferenceQueueFull as exc:
            for future in futures:
                future.cancel()
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except TimeoutError as exc:
            for future in futures:
                future.cancel()
            raise HTTPException(status_code=504, detail="批量路由推理超时") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return [result.to_dict() for result in results]

    return router
