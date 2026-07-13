from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from .schemas import RouteRequest, RouteResponse
from .ui import console_page


def build_router() -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @router.get("/", include_in_schema=False)
    def console():
        return console_page()

    @router.get("/v1/training/status")
    def training_status(request: Request) -> dict:
        return request.app.state.training_job.snapshot()

    @router.get("/v1/models")
    def models(request: Request) -> dict:
        return {"models": request.app.state.model_registry.choices()}

    @router.post("/v1/training/start")
    async def start_training(
        request: Request,
        model_source: str = Form("hfl/chinese-macbert-base"),
        dataset: UploadFile | None = File(None),
    ) -> dict:
        dataset_file: Path | None = None
        if dataset and dataset.filename:
            if not dataset.filename.lower().endswith(".jsonl"):
                raise HTTPException(status_code=422, detail="训练数据集必须是 .jsonl 文件。")
            uploads = Path(__file__).resolve().parents[3] / "datasets/uploads"
            uploads.mkdir(parents=True, exist_ok=True)
            dataset_file = uploads / f"{uuid4().hex}.jsonl"
            dataset_file.write_bytes(await dataset.read())
        return request.app.state.training_job.start(model_source=model_source, dataset_file=dataset_file)

    @router.post("/v1/route", response_model=RouteResponse)
    def route(payload: RouteRequest, request: Request) -> dict:
        try:
            # 首次请求时按需加载所选模型，后续请求直接复用；不传 model 时保持旧行为。
            result = request.app.state.model_registry.get(payload.model).predict(payload.query)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return result.to_dict()

    return router
