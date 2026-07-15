"""本地控制台使用的进程内训练任务管理器。"""

from __future__ import annotations

import os
import re
from pathlib import Path
from threading import Lock, Thread
from typing import Any

from ..utils.device import resolve_device
from ..utils.config import load_yaml
from .schemas import TrainingParameters



class TrainingJob:
    """同一时间运行一个训练任务，并提供可序列化的状态快照。"""

    def __init__(self, project_root: Path) -> None:
        self._project_root = project_root
        self._lock = Lock()
        self._status: dict[str, Any] = {"state": "idle", "message": "等待开始训练"}

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def start(
        self,
        *,
        model_source: str | None = None,
        dataset_file: Path | None = None,
        parameters: TrainingParameters | None = None,
    ) -> dict[str, Any]:
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
            selected_parameters = parameters or TrainingParameters(
                device=os.getenv("LAZY_AGENT_ROUTER_DEVICE", "auto")
            )
            device = resolve_device(selected_parameters.device)
            self._status = {
                "state": "running",
                "message": f"正在加载 {source}，使用 {device.upper()} 启动模型微调…",
                "model": source,
                "dataset": str(training_data),
                "device": device,
                "parameters": selected_parameters.model_dump(),
                "progress": {"percent": 0, "step": 0, "total_steps": 0, "epoch": 0.0},
            }
            thread = Thread(
                target=self._run,
                args=(source, training_data, device, selected_parameters),
                daemon=True,
            )
            thread.start()
            return dict(self._status)

    def _run(
        self,
        model_source: str,
        dataset_file: Path,
        device: str,
        parameters: TrainingParameters,
    ) -> None:
        try:
            # 只在训练启动时导入训练器，普通路由请求无需提前初始化 PyTorch/CUDA。
            from ..training.trainer import train_macbert

            output = train_macbert(
                train_file=dataset_file,
                model_path=model_source,
                output_dir=self._next_output_dir(),
                device=device,
                **parameters.model_dump(exclude={"device"}),
                progress_callback=self._update_progress,
                label_groups=self._label_groups(),
            )
        except Exception as exc:  # 将后台线程异常展示到控制台。
            with self._lock:
                self._status = {
                    "state": "failed",
                    "message": str(exc),
                    "parameters": parameters.model_dump(),
                    "progress": self._status.get("progress", {}),
                }
        else:
            with self._lock:
                self._status = {
                    "state": "completed",
                    "message": "训练完成",
                    "output_dir": str(output),
                    "parameters": parameters.model_dump(),
                    "progress": {
                        **self._status.get("progress", {}),
                        "percent": 100,
                    },
                }

    def _update_progress(self, progress: dict[str, Any]) -> None:
        """由训练线程更新状态；snapshot 可在 API 线程安全读取。"""
        with self._lock:
            if self._status.get("state") != "running":
                return
            self._status["progress"] = dict(progress)
            self._status["message"] = (
                f"训练中：Epoch {progress.get('epoch', 0)}，"
                f"步骤 {progress.get('step', 0)}/{progress.get('total_steps', 0)}"
            )

    def _label_groups(self) -> dict[str, str]:
        """合并项目意图配置，提供标签到 Agent 的分组映射。"""
        groups: dict[str, str] = {}
        for name in ("intents.yaml", "intents_uploaded.yaml"):
            path = self._project_root / "configs" / name
            if not path.is_file():
                continue
            for intent, item in load_yaml(path).get("intents", {}).items():
                if isinstance(item, dict) and isinstance(item.get("agent"), str):
                    groups[intent] = item["agent"]
        return groups

    def _next_output_dir(self) -> Path:
        """返回下一个模型版本目录，避免覆盖已经训练完成的模型。"""
        models_root = self._project_root / "models"
        versions = []
        if models_root.is_dir():
            for path in models_root.iterdir():
                if match := re.fullmatch(r"lazy-agent-router-macbert-v(\d+)", path.name):
                    versions.append(int(match.group(1)))
        return models_root / f"lazy-agent-router-macbert-v{max(versions, default=0) + 1}"
