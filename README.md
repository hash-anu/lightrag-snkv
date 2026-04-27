# lightrag-snkv

**Drop-in SNKV storage backend for [LightRAG](https://github.com/HKUDS/LightRAG).**

Replaces LightRAG's default file-based storage (JSON + NetworkX pickle) with two
embedded SQLite files — no separate server, no network, no operational overhead.

```bash
pip install lightrag-snkv[vector]
```

---

## What gets installed

```
pip install lightrag-snkv[vector]
    │
    ├── lightrag-hku      ← full LightRAG framework (HKUDS team)
    ├── snkv              ← SQLite + HNSW engine
    ├── lightrag-snkv     ← this package (storage adapters only)
    └── numpy
```

`lightrag-snkv` is a thin adapter layer. The LightRAG API you already know
(`ainsert`, `aquery`, `QueryParam`, all query modes) works identically — you only
change the storage backend.

---

## Quickstart

### Step 1 — Install

```bash
pip install lightrag-snkv[vector]
```

### Step 2 — Use

```python
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag_snkv import register

register()  # one-time call — registers SNKV backends with LightRAG

async def main():
    rag = LightRAG(
        working_dir="./my_rag",
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=openai_embed,
        # these 4 lines are the only difference from default LightRAG:
        kv_storage="SNKVKVStorage",
        vector_storage="SNKVVectorStorage",
        graph_storage="SNKVGraphStorage",
        doc_status_storage="SNKVDocStatusStorage",
    )

    await rag.initialize_storages()

    # Insert — same as always
    await rag.ainsert("Marie Curie discovered polonium and radium.")
    await rag.ainsert(["Doc 1 text", "Doc 2 text"])   # batch insert

    # Query — all modes work identically
    result = await rag.aquery("Who discovered radium?", param=QueryParam(mode="hybrid"))
    print(result)

    await rag.finalize_storages()

asyncio.run(main())
```

### What appears on disk

```
my_rag/
├── snkv.db                              # KV cache + graph + doc status
├── snkv_vec.db                          # all embedding vectors
├── snkv_vec.entities_vdb.usearch        # HNSW index (fast restart)
├── snkv_vec.relations_vdb.usearch
└── snkv_vec.chunks_vdb.usearch
```

Two `.db` files instead of dozens of JSON/pickle files.

---

## Convenience helper

If you prefer a one-liner over setting 4 storage names manually:

```python
from lightrag import LightRAG
from lightrag_snkv import register_with_lightrag

rag = LightRAG(
    working_dir="./my_rag",
    llm_model_func=...,
    embedding_func=...,
)
register_with_lightrag(rag)  # sets all 4 backends + registers in one call

await rag.initialize_storages()
```

---

## Migrating from default LightRAG

The only code change is adding `register()` and the 4 storage class names.
Everything else stays the same:

```python
# BEFORE (default LightRAG)
from lightrag import LightRAG, QueryParam

rag = LightRAG(
    working_dir="./my_rag",
    llm_model_func=llm_func,
    embedding_func=embed_func,
)

# AFTER (with SNKV)
from lightrag import LightRAG, QueryParam
from lightrag_snkv import register        # add this

register()                                # add this

rag = LightRAG(
    working_dir="./my_rag",
    llm_model_func=llm_func,
    embedding_func=embed_func,
    kv_storage="SNKVKVStorage",           # add these 4
    vector_storage="SNKVVectorStorage",
    graph_storage="SNKVGraphStorage",
    doc_status_storage="SNKVDocStatusStorage",
)

# insert / query / delete — no changes needed
```

---

## All query modes work

```python
from lightrag import QueryParam

for mode in ["local", "global", "hybrid", "naive", "mix"]:
    result = await rag.aquery("Your question", param=QueryParam(mode=mode))
```

---

## Why SNKV over the default?

| | Default LightRAG | lightrag-snkv |
|---|---|---|
| KV storage | JSON files | SQLite column family |
| Vector search | Custom numpy | HNSW (usearch) |
| Graph storage | NetworkX pickle | SQLite column family |
| Doc status | JSON files | SQLite column family |
| Files on disk | Many (one per namespace) | 2 `.db` files |
| ACID transactions | No | Yes |
| Fast restarts | Slow (pickle rebuild) | Yes (HNSW sidecar) |
| External server | Not needed | Not needed |

### Benchmark

> i7-12700K · 32 GB RAM · SSD — A Christmas Carol corpus (64 chunks, 341 entities)
> `only_need_context=True` isolates pure storage latency

| Stack | Mean (ms) | p50 (ms) | p95 (ms) |
|-------|-----------|----------|----------|
| snkv  | 62.1      | 58.3     | 98.4     |
| nano  | 71.4      | 67.2     | 115.6    |

**1.15× faster at p50** · Results vary by hardware — run `bench/run_all.py` for your numbers.

---

## Requirements

- Python ≥ 3.10
- `lightrag-hku >= 1.4.0` (installed automatically)
- `snkv >= 0.7.0` (installed automatically)

---

## Running Tests

```bash
# Clone the repo
git clone https://github.com/your-username/lightrag-snkv.git
cd lightrag-snkv
pip install -e ".[vector,test]"

# Unit + compatibility tests (no LLM needed — runs in ~2 seconds)
pytest tests/ --ignore=tests/integration -v

# Integration tests (requires LLM credentials)
export OPENAI_API_KEY=sk-...
LIGHTRAG_RUN_INTEGRATION=true pytest tests/integration/test_lightrag_snkv.py -v
```

The compatibility suite (`tests/compat/`) ports all 6 of LightRAG's own graph
storage tests and runs them against `SNKVGraphStorage` — all pass.

---

## Running Benchmarks

```bash
pip install -e ".[vector,bench]"

# Storage-only benchmark (no LLM needed)
python -m bench.run_all --stacks snkv nano --queries 20 --warmup 3 --context-only

# Full end-to-end (requires LLM credentials)
python -m bench.run_all --stacks snkv nano --queries 20 --warmup 5
```

---

## How it works

```
LightRAG                lightrag-snkv              snkv
────────────────        ──────────────────         ──────────────────────
BaseKVStorage      ←──  SNKVKVStorage         ──►  snkv.db (SQLite CFs)
BaseVectorStorage  ←──  SNKVVectorStorage     ──►  snkv_vec.db + usearch
BaseGraphStorage   ←──  SNKVGraphStorage      ──►  snkv.db (SQLite CFs)
DocStatusStorage   ←──  SNKVDocStatusStorage  ──►  snkv.db (SQLite CFs)
```

`lightrag-snkv` implements LightRAG's storage interfaces using SNKV as the
engine. LightRAG never knows or cares that SNKV is underneath — it calls the
same abstract methods it would call on any other backend.

---

## License

Apache 2.0
# lightrag-snkv
