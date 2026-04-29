# lightrag-snkv

A storage backend for [LightRAG](https://github.com/HKUDS/LightRAG) built on [SNKV](https://github.com/hash-anu/snkv)

---

## What is it?

LightRAG is a framework that builds a knowledge graph from your documents and uses it to answer questions. Out of the box it stores everything in a mix of JSON files and a binary pickle file — one file per namespace, scattered across your working directory.

`lightrag-snkv` swaps those files for two embedded SNKV databases. Everything LightRAG does — inserting documents, building the graph, searching vectors, tracking document status — still works exactly the same way. You change four lines of configuration; nothing else in your code changes.

---

## Why use it?

**Fewer files, less mess.**
Default LightRAG creates a growing collection of JSON and pickle files. SNKV collapses all of that into two files: `snkv.db` and `snkv_vec.db`. Easier to back up, move, and reason about.

**Safe writes.**
Every multi-step write (graph update, vector upsert, batch delete) runs inside a single SNKV transaction. Either the whole operation commits or nothing changes. The default backend has no such guarantee — a crash mid-write can leave your data in an inconsistent state.

**Fast restarts.**
The default LightRAG uses a NetworkX pickle for graph storage that must be fully deserialized on startup, and a custom numpy index for vectors that must be rebuilt from scratch. SNKV saves the HNSW vector index to a sidecar file and validates it with a stamp on open — if it matches, the index loads in milliseconds rather than being rebuilt entry by entry.

**No external server.**
Everything runs in-process. No Redis, no Postgres, no Qdrant to install, configure, or keep running. Just a file on disk.

**Drop-in replacement.**
You do not rewrite your application. The full LightRAG API — `ainsert`, `aquery`, `QueryParam`, all five query modes — works identically. The only change is four storage class names in the constructor.

---

## How to use it

### Install

```bash
pip install lightrag-snkv[vector]
```

This installs `lightrag-hku` (the full LightRAG framework), `snkv` (the storage engine), and this adapter package.

### Use

```python
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag_snkv import register

register()  # registers SNKV backends with LightRAG — call once at startup

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

    await rag.ainsert("Marie Curie discovered polonium and radium.")
    await rag.ainsert(["Doc 1 text", "Doc 2 text"])  # batch insert works too

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

### Migrating from default LightRAG

The diff is exactly five lines:

```python
# BEFORE
from lightrag import LightRAG, QueryParam

rag = LightRAG(
    working_dir="./my_rag",
    llm_model_func=llm_func,
    embedding_func=embed_func,
)

# AFTER
from lightrag import LightRAG, QueryParam
from lightrag_snkv import register        # + add this

register()                                # + add this

rag = LightRAG(
    working_dir="./my_rag",
    llm_model_func=llm_func,
    embedding_func=embed_func,
    kv_storage="SNKVKVStorage",           # + add these 4
    vector_storage="SNKVVectorStorage",
    graph_storage="SNKVGraphStorage",
    doc_status_storage="SNKVDocStatusStorage",
)

# ainsert / aquery / aquery — no other changes needed
```

### All query modes work

```python
from lightrag import QueryParam

for mode in ["local", "global", "hybrid", "naive", "mix"]:
    result = await rag.aquery("Your question", param=QueryParam(mode=mode))
```

### Convenience helper

If you prefer a one-liner over passing four class names manually:

```python
from lightrag import LightRAG
from lightrag_snkv import register_with_lightrag

rag = LightRAG(
    working_dir="./my_rag",
    llm_model_func=...,
    embedding_func=...,
)
register_with_lightrag(rag)  # sets all 4 backends and calls register() in one step

await rag.initialize_storages()
```

### Requirements

- Python 3.10 or later
- `lightrag-hku >= 1.4.0` — installed automatically
- `snkv >= 0.7.0` — installed automatically

---

## License

Apache 2.0
