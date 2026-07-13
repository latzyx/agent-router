"""In-process training job management for the local operator console."""

from __future__ import annotations

import os
from pathlib import Path
from threading import Lock, Thread
from typing import Any

from ..utils.device import resolve_device



class TrainingJob:
    """Run one training job at a time and expose a JSON-safe status snapshot."""

    def __init__(self, project_root: Path) -> None:
        self._project_root = project_root
        self._lock = Lock()
        self._status: dict[str, Any] = {"state": "idle", "message": "等待开始训练"}

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def start(self, *, model_source: str | None = None, dataset_file: Path | None = None) -> dict[str, Any]:
        with self._lock:
            if self._status["state"] == "running":
                return dict(self._status)

            source = (model_source or os.getenv(
                "LAZY_AGENT_ROUTER_MODEL_PATH", str(self._project_root / "models/macbert-base")
            )).strip()
            if not source:
                self._status = {"state": "failed", "message": "请选择 Hugging Face 模型或填写本地模型路径。"}
                return dict(self._status)
            local_source = Path(source)
            if source.startswith(("/", "./", "../")) and not local_source.is_dir():
                self._status = {
                    "state": "failed",
                    "message": "未找到指定的本地模型目录。",
                }
                return dict(self._status)

            training_data = dataset_file or self._project_root / "datasets/raw/intents.jsonl"
            device = resolve_device(os.getenv("LAZY_AGENT_ROUTER_DEVICE", "auto"))
            self._status = {
                "state": "running",
                "message": f"正在加载 {source}，使用 {device.upper()} 启动模型微调…",
                "model": source,
                "dataset": str(training_data),
                "device": device,
            }
            thread = Thread(target=self._run, args=(source, training_data, device), daemon=True)
            thread.start()
            return dict(self._status)

    def _run(self, model_source: str, dataset_file: Path, device: str) -> None:
        try:
            # Import only when training starts: serving route queries must not
            # require a GPU-enabled PyTorch installation.
            from ..training.trainer import train_macbert

            output = train_macbert(
                train_file=dataset_file,
                model_path=model_source,
                output_dir=self._project_root / "models/lazy-agent-router-macbert-v1",
                device=device,
            )
        except Exception as exc:  # Surface operational failures to the console.
            with self._lock:
                self._status = {"state": "failed", "message": str(exc)}
        else:
            with self._lock:
                self._status = {"state": "completed", "message": "训练完成", "output_dir": str(output)}
