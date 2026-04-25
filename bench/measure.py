"""Latency measurement utilities."""
from __future__ import annotations

import asyncio
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Measurement:
    stack: str
    mode: str
    samples_ms: list[float] = field(default_factory=list)

    def add(self, elapsed_s: float) -> None:
        self.samples_ms.append(elapsed_s * 1000.0)

    def percentile(self, p: int) -> float:
        if not self.samples_ms:
            return float("nan")
        sorted_s = sorted(self.samples_ms)
        idx = max(0, int(len(sorted_s) * p / 100) - 1)
        return sorted_s[idx]

    def mean(self) -> float:
        return statistics.mean(self.samples_ms) if self.samples_ms else float("nan")

    def stdev(self) -> float:
        return statistics.stdev(self.samples_ms) if len(self.samples_ms) > 1 else 0.0


async def timed_query(
    rag,
    query: str,
    mode: str,
    only_need_context: bool = True,
    top_k: int = 60,
) -> float:
    """Run one query and return elapsed seconds."""
    from lightrag import QueryParam

    start = time.perf_counter()
    await rag.aquery(
        query,
        param=QueryParam(
            mode=mode,
            top_k=top_k,
            only_need_context=only_need_context,
        ),
    )
    return time.perf_counter() - start


async def run_measurements(
    rag,
    queries: list[str],
    stack_name: str,
    modes: list[str],
    warmup: int = 3,
    bench: int = 20,
    only_need_context: bool = True,
    top_k: int = 60,
) -> list[Measurement]:
    """Warm up then collect latency samples."""
    results: list[Measurement] = []

    for mode in modes:
        # Warmup
        for q in queries[:warmup]:
            await timed_query(rag, q, mode, only_need_context, top_k)

        m = Measurement(stack=stack_name, mode=mode)
        bench_queries = (queries * (bench // len(queries) + 1))[:bench]
        for q in bench_queries:
            elapsed = await timed_query(rag, q, mode, only_need_context, top_k)
            m.add(elapsed)
        results.append(m)

    return results
