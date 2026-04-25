"""Benchmark configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

QueryMode = Literal["local", "global", "hybrid", "naive", "mix"]


@dataclass
class BenchConfig:
    # Dataset
    corpus_path: str = os.path.join(os.path.dirname(__file__), "data", "corpus.txt")
    queries_path: str = os.path.join(os.path.dirname(__file__), "data", "queries.txt")

    # What to measure
    query_modes: list[QueryMode] = field(
        default_factory=lambda: ["local", "global", "hybrid"]
    )
    top_k: int = 60
    only_need_context: bool = True  # skip LLM to isolate storage latency

    # Timing
    warmup_queries: int = 3
    bench_queries: int = 20

    # Percentiles reported
    percentiles: list[int] = field(default_factory=lambda: [50, 95, 99])

    # Storage stacks to benchmark (subset of keys from stacks.py)
    stacks: list[str] = field(
        default_factory=lambda: ["snkv", "nano", "faiss"]
    )

    # Working dirs (one per stack)
    base_dir: str = os.path.join(os.path.dirname(__file__), "_bench_wd")
