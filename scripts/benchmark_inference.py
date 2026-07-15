"""在进程内压测 FastAPI + 微批推理链路，不包含网络传输时间。"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import time


def percentile(values: list[float], ratio: float) -> float:
    return values[min(round((len(values) - 1) * ratio), len(values) - 1)]


async def run(args: argparse.Namespace) -> None:
    os.environ["LAZY_ROUTER_MAX_BATCH_SIZE"] = str(args.batch_size)
    os.environ["LAZY_ROUTER_MAX_WAIT_MS"] = str(args.batch_wait_ms)
    os.environ["LAZY_ROUTER_REQUEST_TIMEOUT_MS"] = str(args.timeout_ms)

    import httpx

    from lazy_agent_router.serving.app import create_app

    app = create_app()
    router = app.state.model_registry.get(args.model)
    router.predict_batch(["SAP 接口为什么一直报错"] * args.batch_size)

    queries = [
        "SAP 接口为什么一直报错，请帮忙排查",
        "查询一下采购订单现在到哪个审批节点",
        "员工账号无法登录应该怎么处理",
        "帮我查一下客户付款到账情况",
    ]
    semaphore = asyncio.Semaphore(args.concurrency)
    latencies: list[float] = []

    async def request_one(client: httpx.AsyncClient, start_index: int, count: int) -> None:
        async with semaphore:
            started = time.perf_counter()
            if count == 1:
                response = await client.post(
                    "/v1/route",
                    json={"query": queries[start_index % len(queries)], "model": args.model},
                )
            else:
                response = await client.post(
                    "/v1/route/batch",
                    json={
                        "queries": [
                            queries[index % len(queries)]
                            for index in range(start_index, start_index + count)
                        ],
                        "model": args.model,
                    },
                )
            latencies.append((time.perf_counter() - started) * 1000)
            response.raise_for_status()

    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=transport, base_url="http://benchmark") as client:
            started = time.perf_counter()
            jobs = [
                (start, min(args.client_batch_size, args.requests - start))
                for start in range(0, args.requests, args.client_batch_size)
            ]
            await asyncio.gather(*(request_one(client, start, count) for start, count in jobs))
            elapsed = time.perf_counter() - started

    latencies.sort()
    stats = app.state.inference_batcher.snapshot()
    print(
        f"logical_requests={args.requests} http_requests={len(jobs)} "
        f"concurrency={args.concurrency} client_batch_size={args.client_batch_size} "
        f"server_batch_size={args.batch_size}"
    )
    print(f"elapsed={elapsed:.3f}s throughput={args.requests / elapsed:.1f} QPS")
    print(
        "latency_ms "
        f"mean={statistics.fmean(latencies):.2f} "
        f"p50={percentile(latencies, 0.50):.2f} "
        f"p95={percentile(latencies, 0.95):.2f} "
        f"p99={percentile(latencies, 0.99):.2f}"
    )
    print(
        f"batches={stats['batches']} average_batch_size={stats['average_batch_size']:.2f} "
        f"max_observed_batch_size={stats['max_observed_batch_size']} errors={stats['errors']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lazy-agent-router-macbert-v22")
    parser.add_argument("--requests", type=int, default=2000)
    parser.add_argument("--concurrency", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batch-wait-ms", type=float, default=3)
    parser.add_argument("--timeout-ms", type=float, default=1000)
    parser.add_argument("--client-batch-size", type=int, choices=range(1, 257), default=1)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
