"""Run the full benchmark suite.

Usage:
    python -m bench.run_all                          # default config
    python -m bench.run_all --stacks snkv nano       # specific stacks
    python -m bench.run_all --queries 30 --warmup 5  # more samples

Results are written to bench/_results/<timestamp>.json.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import time
from pathlib import Path

from bench.config import BenchConfig
from bench.dataset import load_corpus, load_queries
from bench.measure import Measurement, run_measurements
from bench.report import print_report
from bench.stacks import STACK_FACTORIES


async def _index_stack(rag, corpus: str) -> float:
    """Insert corpus and return elapsed seconds."""
    start = time.perf_counter()
    await rag.ainsert(corpus)
    return time.perf_counter() - start


async def run_benchmark(cfg: BenchConfig) -> list[Measurement]:
    corpus = load_corpus(cfg.corpus_path)
    queries = load_queries(cfg.queries_path)

    # Resolve LLM / embed functions from environment
    from lightrag.api.lightrag_server import _get_embed_func, _get_llm_func

    llm_func = _get_llm_func()
    embed_func = _get_embed_func()

    all_results: list[Measurement] = []
    index_times: dict[str, float] = {}

    for stack_name in cfg.stacks:
        factory = STACK_FACTORIES[stack_name]
        working_dir = os.path.join(cfg.base_dir, stack_name)

        # Fresh working directory for reproducibility
        if os.path.exists(working_dir):
            shutil.rmtree(working_dir)
        os.makedirs(working_dir, exist_ok=True)

        print(f"\n[{stack_name}] Indexing corpus …")
        rag = factory(working_dir, llm_func, embed_func)
        await rag.initialize_storages()
        idx_time = await _index_stack(rag, corpus)
        index_times[stack_name] = idx_time
        print(f"[{stack_name}] Indexed in {idx_time:.1f}s")

        print(f"[{stack_name}] Running queries …")
        measurements = await run_measurements(
            rag,
            queries,
            stack_name=stack_name,
            modes=cfg.query_modes,
            warmup=cfg.warmup_queries,
            bench=cfg.bench_queries,
            only_need_context=cfg.only_need_context,
            top_k=cfg.top_k,
        )
        all_results.extend(measurements)
        await rag.finalize_storages()

    # Save raw results
    results_dir = Path(cfg.base_dir) / "_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"bench_{ts}.json"
    out_path.write_text(
        json.dumps(
            {
                "index_times_s": index_times,
                "measurements": [
                    {
                        "stack": m.stack,
                        "mode": m.mode,
                        "samples_ms": m.samples_ms,
                        "mean_ms": m.mean(),
                        "p50_ms": m.percentile(50),
                        "p95_ms": m.percentile(95),
                        "p99_ms": m.percentile(99),
                    }
                    for m in all_results
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nRaw results → {out_path}")
    return all_results


def _parse_args():
    p = argparse.ArgumentParser(description="Run lightrag-snkv benchmarks")
    p.add_argument("--stacks", nargs="+", default=None,
                   help="Stacks to benchmark (snkv nano faiss)")
    p.add_argument("--queries", type=int, default=None,
                   help="Number of timed queries per stack/mode")
    p.add_argument("--warmup", type=int, default=None,
                   help="Number of warmup queries")
    p.add_argument("--modes", nargs="+", default=None,
                   help="Query modes (local global hybrid naive)")
    p.add_argument("--context-only", action="store_true",
                   help="Skip LLM call (only_need_context=True)")
    return p.parse_args()


def main():
    args = _parse_args()
    cfg = BenchConfig()
    if args.stacks:
        cfg.stacks = args.stacks
    if args.queries:
        cfg.bench_queries = args.queries
    if args.warmup:
        cfg.warmup_queries = args.warmup
    if args.modes:
        cfg.query_modes = args.modes
    if args.context_only:
        cfg.only_need_context = True

    all_results = asyncio.run(run_benchmark(cfg))
    print_report(all_results, percentiles=cfg.percentiles)


if __name__ == "__main__":
    main()
