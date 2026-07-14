"""将并发 HTTP 路由请求合并成少量模型批量推理。"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from concurrent.futures import Future, InvalidStateError
from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Lock, Thread
from typing import Callable

from ..core.router import LazyAgentRouter
from ..core.types import RouterResult


class InferenceQueueFull(RuntimeError):
    """推理队列达到上限，调用方应执行背压或稍后重试。"""


@dataclass(slots=True)
class _Request:
    query: str
    model_id: str | None
    future: Future[RouterResult] | asyncio.Future[RouterResult]
    loop: asyncio.AbstractEventLoop | None = None


class InferenceBatcher:
    """一个模型推理线程服务大量异步 HTTP 请求。"""

    def __init__(
        self,
        router_resolver: Callable[[str | None], LazyAgentRouter],
        *,
        max_batch_size: int = 32,
        max_wait_ms: float = 3.0,
        queue_size: int = 4096,
        request_timeout_ms: float = 500.0,
    ) -> None:
        if max_batch_size < 1 or max_wait_ms < 0 or queue_size < 1 or request_timeout_ms <= 0:
            raise ValueError("invalid inference batcher configuration")
        self._router_resolver = router_resolver
        self.max_batch_size = max_batch_size
        self.max_wait_seconds = max_wait_ms / 1000
        self.request_timeout_seconds = request_timeout_ms / 1000
        self._queue: Queue[_Request | None] = Queue(maxsize=queue_size)
        self._start_lock = Lock()
        self._stats_lock = Lock()
        self._thread: Thread | None = None
        self._closed = False
        self._started_at = time.monotonic()
        self._requests = 0
        self._batches = 0
        self._errors = 0
        self._total_batch_items = 0
        self._max_observed_batch = 0
        self._total_inference_seconds = 0.0

    def submit(self, query: str, model_id: str | None = None) -> Future[RouterResult]:
        if self._closed:
            raise RuntimeError("inference batcher is closed")
        self._ensure_started()
        future: Future[RouterResult] = Future()
        try:
            self._queue.put_nowait(_Request(query=query, model_id=model_id, future=future))
        except Full as exc:
            raise InferenceQueueFull("推理队列已满，请稍后重试") from exc
        return future

    def submit_async(
        self, query: str, model_id: str | None = None
    ) -> asyncio.Future[RouterResult]:
        """提交原生 asyncio Future，避免跨线程 concurrent Future 桥接开销。"""
        if self._closed:
            raise RuntimeError("inference batcher is closed")
        self._ensure_started()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[RouterResult] = loop.create_future()

        # 某些 Python 3.14/容器组合不会立即唤醒阻塞中的 selector 来处理
        # call_soon_threadsafe。低流量时用一个很轻的定时心跳兜底；高并发下
        # 事件循环本来就持续活跃，Future 通常会在第一次心跳前完成。
        def wake_loop_until_done() -> None:
            if not future.done() and not self._closed:
                loop.call_later(0.005, wake_loop_until_done)

        loop.call_later(0.005, wake_loop_until_done)
        try:
            self._queue.put_nowait(
                _Request(query=query, model_id=model_id, future=future, loop=loop)
            )
        except Full as exc:
            raise InferenceQueueFull("推理队列已满，请稍后重试") from exc
        return future

    def snapshot(self) -> dict[str, int | float]:
        with self._stats_lock:
            uptime = max(time.monotonic() - self._started_at, 1e-9)
            return {
                "requests": self._requests,
                "batches": self._batches,
                "errors": self._errors,
                "queue_depth": self._queue.qsize(),
                "queue_capacity": self._queue.maxsize,
                "max_batch_size": self.max_batch_size,
                "max_wait_ms": self.max_wait_seconds * 1000,
                "average_batch_size": (
                    self._total_batch_items / self._batches if self._batches else 0.0
                ),
                "max_observed_batch_size": self._max_observed_batch,
                "average_inference_ms": (
                    self._total_inference_seconds / self._batches * 1000 if self._batches else 0.0
                ),
                "lifetime_throughput": self._requests / uptime,
            }

    def close(self) -> None:
        self._closed = True
        thread = self._thread
        if thread and thread.is_alive():
            try:
                self._queue.put_nowait(None)
            except Full:
                pass
            thread.join(timeout=2)

    def _ensure_started(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        with self._start_lock:
            if not self._thread or not self._thread.is_alive():
                self._thread = Thread(
                    target=self._run,
                    name="lazy-router-inference",
                    daemon=True,
                )
                self._thread.start()

    def _run(self) -> None:
        while not self._closed:
            first = self._queue.get()
            if first is None:
                break
            batch = [first]
            deadline = time.perf_counter() + self.max_wait_seconds
            while len(batch) < self.max_batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    item = self._queue.get(timeout=remaining)
                except Empty:
                    break
                if item is None:
                    self._closed = True
                    break
                batch.append(item)
            self._process(batch)

    def _process(self, batch: list[_Request]) -> None:
        active = [item for item in batch if not item.future.cancelled()]
        if not active:
            return
        started = time.perf_counter()
        errors = 0
        grouped: dict[str | None, list[_Request]] = defaultdict(list)
        for item in active:
            grouped[item.model_id].append(item)

        for model_id, items in grouped.items():
            try:
                router = self._router_resolver(model_id)
                results = router.predict_batch([item.query for item in items])
                for item, result in zip(items, results, strict=True):
                    self._deliver(item, result=result)
            except Exception as exc:
                errors += len(items)
                for item in items:
                    self._deliver(item, error=exc)

        elapsed = time.perf_counter() - started
        with self._stats_lock:
            self._requests += len(active)
            self._batches += 1
            self._errors += errors
            self._total_batch_items += len(active)
            self._max_observed_batch = max(self._max_observed_batch, len(active))
            self._total_inference_seconds += elapsed

    @staticmethod
    def _deliver(
        item: _Request,
        *,
        result: RouterResult | None = None,
        error: Exception | None = None,
    ) -> None:
        def complete() -> None:
            if item.future.done():
                return
            if error is None:
                item.future.set_result(result)  # type: ignore[arg-type]
            else:
                item.future.set_exception(error)

        if item.loop is not None:
            item.loop.call_soon_threadsafe(complete)
        else:
            try:
                complete()
            except InvalidStateError:
                pass
