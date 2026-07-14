from concurrent.futures import Future
from threading import Barrier

from lazy_agent_router.serving.inference_batcher import InferenceBatcher


class _Result:
    def __init__(self, query: str):
        self.query = query


class _BatchRouter:
    def __init__(self):
        self.calls: list[list[str]] = []

    def predict_batch(self, queries: list[str]) -> list[_Result]:
        self.calls.append(queries)
        return [_Result(query) for query in queries]


def test_concurrent_submissions_are_combined_into_one_model_call():
    router = _BatchRouter()
    batcher = InferenceBatcher(lambda _model: router, max_batch_size=8, max_wait_ms=20)
    barrier = Barrier(5)

    def submit(index: int) -> Future[_Result]:
        barrier.wait()
        return batcher.submit(f"query-{index}")

    from concurrent.futures import ThreadPoolExecutor

    try:
        with ThreadPoolExecutor(max_workers=5) as pool:
            submitted = list(pool.map(submit, range(5)))
        results = [future.result(timeout=1) for future in submitted]

        assert {result.query for result in results} == {f"query-{index}" for index in range(5)}
        assert sum(len(call) for call in router.calls) == 5
        assert len(router.calls) == 1
        assert batcher.snapshot()["average_batch_size"] == 5
    finally:
        batcher.close()
