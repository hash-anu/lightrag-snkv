# lightrag-snkv

**SNKV storage backends for [LightRAG](https://github.com/HKUDS/LightRAG)** — two embedded SQLite files replace four separate storage systems (KV, vector, graph, doc-status), eliminating network hops and operational overhead.

## Why SNKV?

| | NanoVectorDB (default) | SNKV |
|---|---|---|
| KV storage | JSON files | SQLite column family |
| Vector search | Custom numpy | HNSW (usearch) |
| Graph storage | NetworkX pickle | SQLite column family |
| Doc status | JSON files | SQLite column family |
| Network required | ✗ | ✗ |
| Single-directory DB | ✗ (many files) | ✓ (2 `.db` files) |
| ACID transactions | ✗ | ✓ |
| Fast restarts | Slow (pickle reload) | ✓ (HNSW sidecar) |

SNKV uses **two SQLite files** per working directory:

- `snkv.db` — KV namespaces, graph (nodes/edges/adj), and doc-status, each in its own column family
- `snkv_vec.db` — all vector namespaces, each in five column families (values, raw vectors, index↔key maps, metadata)

The HNSW index is kept in-memory and saved as a `.usearch` sidecar on close so restarts skip the O(n·d) rebuild.

## Benchmark

> Measured on i7-12700K, 32 GB RAM, SSD — A Christmas Carol corpus (64 chunks, 341 entities), `only_need_context=True` to isolate pure storage latency.

### Mode: `hybrid`

| Stack  | N  | Mean(ms) | p50(ms) | p95(ms) | p99(ms) |
|--------|-----|----------|---------|---------|---------|
| snkv   | 20  | 62.1     | 58.3    | 98.4    | 112.7   |
| nano   | 20  | 71.4     | 67.2    | 115.6   | 128.3   |

**SNKV speedup vs nano (p50): 1.15×**

*Results vary by hardware and corpus size. Run the benchmark yourself — see [Running Benchmarks](#running-benchmarks).*

## Installation

```bash
pip install lightrag-snkv[vector]
```

> **Requirements**: Python ≥ 3.10, [lightrag-hku](https://pypi.org/project/lightrag-hku/) ≥ 1.4.0

## Quick Start

```python
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag_snkv import register

# Register SNKV class names into LightRAG's storage registry
register()

# Configure your LLM and embedding functions (same as normal LightRAG)
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

Or use the one-call convenience helper:

```python
from lightrag import LightRAG
from lightrag_snkv import register_with_lightrag

rag = LightRAG(working_dir="./rag_storage", ...)
register_with_lightrag(rag)   # registers SNKV + sets all 4 backend names
await rag.initialize_storages()
```

## Project Structure

```
lightrag-snkv/
├── lightrag_snkv/
│   ├── snkv_shared.py           # Per-db singleton: one KVStore + executor, ref-counted
│   ├── snkv_kv_impl.py          # BaseKVStorage  → snkv.db (one CF per namespace)
│   ├── snkv_vector_impl.py      # BaseVectorStorage → snkv_vec.db + .usearch sidecar
│   ├── snkv_graph_impl.py       # BaseGraphStorage → snkv.db (nodes/edges/adj CFs)
│   ├── snkv_doc_status_impl.py  # DocStatusStorage → snkv.db (doc_status CF)
│   └── register.py              # One-call registration into LightRAG's registry
├── tests/
│   ├── conftest.py              # Shared fixtures (mock embedding, tmp_dir)
│   ├── test_snkv_kv.py          # Unit tests — KV storage
│   ├── test_snkv_vector.py      # Unit tests — vector storage
│   ├── test_snkv_graph.py       # Unit tests — graph storage
│   ├── test_snkv_doc_status.py  # Unit tests — doc-status storage
│   ├── compat/
│   │   └── test_lightrag_graph_compat.py  # LightRAG's own 6 graph tests, SNKV-backed
│   └── integration/
│       ├── base_lightrag_test.py          # Reusable insert/query/delete suite
│       └── test_lightrag_snkv.py          # End-to-end SNKV stack (requires LLM creds)
└── bench/
    ├── config.py       # Benchmark configuration dataclass
    ├── dataset.py      # Built-in corpus + query loading
    ├── stacks.py       # LightRAG stack factory functions (snkv, nano, faiss)
    ├── measure.py      # Timed query helpers
    ├── run_all.py      # CLI entry point
    └── report.py       # Table + Markdown report rendering
```

## Running Tests

```bash
# Unit + compat tests (no LLM required)
pytest tests/ --ignore=tests/integration -v

# Integration tests (requires LLM credentials — see below)
LIGHTRAG_RUN_INTEGRATION=true pytest tests/integration/test_lightrag_snkv.py -v
```

### LLM credentials for integration tests

The integration test and benchmark use `lightrag-hku[api]` server helpers to
resolve LLM/embedding from environment variables, with an OpenAI fallback:

**Option A — server helpers (multi-backend):**
```bash
pip install "lightrag-hku[api]>=1.4.0"
export LLM_BINDING=openai LLM_MODEL=gpt-4o-mini LLM_BINDING_API_KEY=sk-...
export EMBEDDING_BINDING=openai EMBEDDING_MODEL=text-embedding-3-small EMBEDDING_DIM=1536 EMBEDDING_BINDING_API_KEY=sk-...
```

**Option B — direct OpenAI:**
```bash
export OPENAI_API_KEY=sk-...
```

## Running Benchmarks

```bash
# Context-only (no LLM call needed — isolates pure storage latency)
python -m bench.run_all --stacks snkv nano --queries 10 --warmup 3 --context-only

# Full end-to-end benchmark (requires LLM credentials)
python -m bench.run_all --stacks snkv nano --queries 20 --warmup 5

# Custom corpus
python -m bench.run_all --stacks snkv nano \
  --corpus path/to/corpus.txt \
  --queries-file path/to/queries.txt
```

## Compatibility

LightRAG's own graph storage test suite (6 tests from `tests/test_graph_storage.py`)
runs against `SNKVGraphStorage` as part of the standard test suite via
`tests/compat/test_lightrag_graph_compat.py` — all 6 pass.

## Design Notes

- **Shared singleton**: `snkv_shared.py` maintains one `KVStore` + one `ThreadPoolExecutor(max_workers=1)` per db file, reference-counted across all storage instances that share the same file. All I/O for each file is serialised through one thread.
- **Two files**: `snkv.db` holds KV, graph, and doc-status column families. `snkv_vec.db` holds all vector namespaces. Column family names are prefixed to avoid collisions.
- **HNSW sidecar**: The usearch index is kept in-memory. On `finalize()` it is saved to `snkv_vec.{namespace}.usearch` (+ a `.nid` stamp). On `initialize()` the sidecar is loaded if the stamp matches, skipping the O(n·d) rebuild.
- **Cosine convention**: SNKV stores cosine *distance* (0 = identical). The vector impl converts to cosine *similarity* (1 = identical) to match LightRAG's `cosine_better_than_threshold` filter semantics.

## License

Apache 2.0
