"""本地控制台使用的进程内训练任务管理器。"""

from __future__ import annotations

import os
import re
from pathlib import Path
from threading import Lock, Thread
from typing import Any

from ..utils.device import resolve_device



class TrainingJob:
    """同一时间运行一个训练任务，并提供可序列化的状态快照。"""

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
            # 只在训练启动时导入训练器，普通路由请求无需提前初始化 PyTorch/CUDA。
            from ..training.trainer import train_macbert

            output = train_macbert(
                train_file=dataset_file,
                model_path=model_source,
                output_dir=self._next_output_dir(),
                device=device,
            )
        except Exception as exc:  # 将后台线程异常展示到控制台。
            with self._lock:
                self._status = {"state": "failed", "message": str(exc)}
        else:
            with self._lock:
                self._status = {"state": "completed", "message": "训练完成", "output_dir": str(output)}

    def _next_output_dir(self) -> Path:
        """返回下一个模型版本目录，避免覆盖已经训练完成的模型。"""
        models_root = self._project_root / "models"
        versions = []
        if models_root.is_dir():
            for path in models_root.iterdir():
                if match := re.fullmatch(r"lazy-agent-router-macbert-v(\d+)", path.name):
                    versions.append(int(match.group(1)))
        return models_root / f"lazy-agent-router-macbert-v{max(versions, default=0) + 1}"
