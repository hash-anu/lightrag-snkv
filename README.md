# lightrag-snkv

**SNKV storage backends for [LightRAG](https://github.com/HKUDS/LightRAG)** — one embedded SQLite file replaces four separate storage systems (KV, vector, graph, doc-status), eliminating network hops and operational overhead.

## Why SNKV?

| | NanoVectorDB (default) | FAISS | SNKV |
|---|---|---|---|
| KV storage | JSON files | JSON files | SQLite CF |
| Vector search | Custom numpy | HNSW (IVF) | HNSW (usearch) |
| Graph storage | NetworkX pickle | NetworkX pickle | SQLite CF |
| Doc status | JSON files | JSON files | SQLite CF |
| Network required | ✗ | ✗ | ✗ |
| Single file | ✗ | ✗ | ✓ |
| ACID transactions | ✗ | ✗ | ✓ |
| Cold-start | Fast | Fast | Fast (sidecar) |

SNKV stores everything in **one `.db` file** per working directory using SQLite column families. The HNSW index saves a `.usearch` sidecar on close so restarts skip the O(n·d) rebuild.

## Benchmark Results

> Measured on: i7-12700K, 32 GB RAM, SSD — A Christmas Carol corpus (64 chunks, 341 entities), `only_need_context=True` to isolate pure storage latency.

### Mode: `hybrid`

| Stack  | N  | Mean(ms) | p50(ms) | p95(ms) | p99(ms) |
|--------|-----|----------|---------|---------|---------|
| snkv   | 20  | 62.1     | 58.3    | 98.4    | 112.7   |
| nano   | 20  | 71.4     | 67.2    | 115.6   | 128.3   |
| faiss  | 20  | 68.9     | 64.1    | 109.2   | 121.4   |

**SNKV speedup vs nano (p50): 1.15×**

*Results will vary; run the benchmark yourself to get numbers for your hardware.*

## Installation

```bash
pip install lightrag-snkv[vector]
```

Or from source (edits both `lightrag-snkv` and the local `LightRAG` clone):

```bash
cd lightrag-snkv
pip install -e ".[vector,test]"
pip install -e "../LightRAG"   # local LightRAG copy
```

## Quick Start

```python
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag_snkv import register

# 1. Register SNKV class names into LightRAG's storage registry
register()

# 2. Configure your LLM and embedding functions (same as normal LightRAG)
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

async def main():
    rag = LightRAG(
        working_dir="./rag_storage",
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=openai_embed,
        kv_storage="SNKVKVStorage",
        vector_storage="SNKVVectorStorage",
        graph_storage="SNKVGraphStorage",
        doc_status_storage="SNKVDocStatusStorage",
        vector_db_storage_cls_kwargs={"cosine_better_than_threshold": 0.2},
    )
    await rag.initialize_storages()
    await rag.ainsert("Your document text here")
    result = await rag.aquery("Your question", param=QueryParam(mode="hybrid"))
    print(result)
    await rag.finalize_storages()

asyncio.run(main())
```

Or use the convenience helper:

```python
from lightrag_snkv import register_with_lightrag

rag = LightRAG(working_dir="./rag_storage", ...)
register_with_lightrag(rag)   # sets all 4 storage backends + registers
await rag.initialize_storages()
```

## Project Structure

```
lightrag-snkv/
├── lightrag_snkv/
│   ├── snkv_kv_impl.py          # BaseKVStorage → snkv_kv.db (column families)
│   ├── snkv_vector_impl.py      # BaseVectorStorage → snkv_vec_{ns}.db (VectorStore)
│   ├── snkv_graph_impl.py       # BaseGraphStorage → snkv_graph.db (nodes/edges/adj CFs)
│   ├── snkv_doc_status_impl.py  # DocStatusStorage → snkv_doc.db
│   └── register.py              # One-call registration into LightRAG
├── tests/
│   ├── conftest.py              # Shared fixtures (mock embedding, tmp_dir)
│   ├── test_snkv_kv.py          # 13 unit tests
│   ├── test_snkv_vector.py      # 12 unit tests
│   ├── test_snkv_graph.py       # 14 unit tests
│   ├── test_snkv_doc_status.py  # 7 unit tests
│   ├── integration/             # End-to-end LightRAG tests (requires LLM)
│   └── correctness/             # Answer quality + consistency checks
└── bench/
    ├── config.py                # Benchmark configuration
    ├── dataset.py               # Corpus + query loading
    ├── stacks.py                # Stack factory functions
    ├── measure.py               # Timed query helpers
    ├── run_all.py               # CLI entrypoint
    └── report.py                # Table + Markdown output
```

## Running Tests

```bash
# Unit tests only (no LLM required)
pytest tests/ --ignore=tests/integration --ignore=tests/correctness -v

# Integration + correctness (requires LLM credentials in env)
LIGHTRAG_RUN_INTEGRATION=true pytest tests/

# With the LightRAG .env file:
source .env && pytest tests/integration/ -v
```

## Running Benchmarks

```bash
# Quick smoke (context-only, no LLM needed for storage benchmarks)
python -m bench.run_all --stacks snkv nano --queries 10 --warmup 3 --context-only

# Full benchmark
python -m bench.run_all --stacks snkv nano faiss --queries 50 --warmup 5

# Custom corpus
python -m bench.run_all --stacks snkv nano \
  --corpus path/to/corpus.txt \
  --queries-file path/to/queries.txt
```

## Design Notes

- **One DB file per storage type**: `snkv_kv.db`, `snkv_graph.db`, `snkv_doc.db` (shared by all namespaces via SQLite column families); `snkv_vec_{namespace}.db` per vector namespace.
- **Thread safety**: Each storage instance uses a `ThreadPoolExecutor(max_workers=1)` to serialize SNKV calls on a dedicated thread. All async methods use `run_in_executor`.
- **HNSW sidecar**: VectorStore saves the usearch index to `{path}.usearch` on close, skipping O(n·d) rebuild on restart. Detects staleness via a `.nid` stamp file.
- **Cosine convention**: SNKV uses cosine *distance* (0=identical). The vector impl converts to cosine *similarity* (1=identical) to match NanoVectorDB's behaviour and LightRAG's `cosine_better_than_threshold` filter.

## License

Apache 2.0
