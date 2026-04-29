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

Clone the repository and install in editable mode:

```bash
git clone https://github.com/hash-anu/lightrag-snkv.git
cd lightrag-snkv

# Programmatic use only
pip install -e ".[vector]"

# Web UI + server
pip install -e ".[vector,server]"
```

This installs all dependencies: `lightrag-hku` (the full LightRAG framework) and `snkv` (the storage engine). The `server` extra adds `uvicorn` and the LightRAG API server components needed to run the web UI.

### Use

If you already have LightRAG set up, plug in SNKV by adding two things — one import and four constructor params. Everything else in your code stays the same.

```python
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag_snkv import register          # ← add this import

register()                                  # ← call once at startup

async def main():
    rag = LightRAG(
        working_dir="./my_rag",
        llm_model_func=your_llm_func,       # your existing LLM function — unchanged
        embedding_func=your_embed_func,     # your existing embed function — unchanged
        kv_storage="SNKVKVStorage",         # ← add these 4 lines
        vector_storage="SNKVVectorStorage",
        graph_storage="SNKVGraphStorage",
        doc_status_storage="SNKVDocStatusStorage",
    )

    await rag.initialize_storages()

    await rag.ainsert("Marie Curie discovered polonium and radium.")
    await rag.ainsert(["Doc 1 text", "Doc 2 text"])

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

# ainsert / aquery / finalize_storages — no other changes needed
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
    llm_model_func=your_llm_func,
    embedding_func=your_embed_func,
)
register_with_lightrag(rag)  # sets all 4 storage backends and calls register() in one step

await rag.initialize_storages()
```

### Web UI

LightRAG ships a full web interface. To run it with SNKV as the storage backend, use the included `server.py` instead of the standard `lightrag-server` command — it registers SNKV before the server starts:

```bash
python server.py
```

Then open `http://localhost:9621` in your browser.

`server.py` loads your `.env` automatically and overrides the storage settings to use SNKV. The host and port are read from `HOST` and `PORT` in your `.env` (defaulting to `0.0.0.0:9621`).

### Reusing your existing LightRAG `.env`

If you already run the LightRAG server and have a `.env` configured, you can reuse it directly — no need to rewire your LLM or embedding setup.

Install `python-dotenv` if you don't have it:

```bash
pip install python-dotenv
```

Then load your `.env` before building the `LightRAG` instance:

```python
import asyncio
from dotenv import load_dotenv
from lightrag import LightRAG, QueryParam
from lightrag_snkv import register, get_llm_and_embed_funcs

load_dotenv()  # loads .env from the current directory (or pass a path: load_dotenv("/path/to/.env"))

llm_func, embed_func = get_llm_and_embed_funcs()  # reads LLM_BINDING, LLM_MODEL, EMBEDDING_BINDING, etc.

register()

async def main():
    rag = LightRAG(
        working_dir="./my_rag",
        llm_model_func=llm_func,
        embedding_func=embed_func,
        kv_storage="SNKVKVStorage",
        vector_storage="SNKVVectorStorage",
        graph_storage="SNKVGraphStorage",
        doc_status_storage="SNKVDocStatusStorage",
    )

    await rag.initialize_storages()
    result = await rag.aquery("Your question", param=QueryParam(mode="hybrid"))
    print(result)
    await rag.finalize_storages()

asyncio.run(main())
```

`get_llm_and_embed_funcs()` reads the same environment variables that the LightRAG server uses:

| Variable | Purpose |
|---|---|
| `LLM_BINDING` | LLM backend — `openai`, `azure_openai`, `ollama` |
| `LLM_MODEL` | Model name (e.g. `gpt-4o-mini`) |
| `LLM_BINDING_HOST` | Custom endpoint (optional) |
| `LLM_BINDING_API_KEY` | API key (falls back to `OPENAI_API_KEY`) |
| `EMBEDDING_BINDING` | Embedding backend — `openai`, `azure_openai`, `ollama` |
| `EMBEDDING_MODEL` | Embedding model name |
| `EMBEDDING_DIM` | Vector dimension (default: 1536) |
| `EMBEDDING_BINDING_HOST` | Custom endpoint (optional) |
| `EMBEDDING_BINDING_API_KEY` | API key (falls back to `OPENAI_API_KEY`) |
| `AZURE_OPENAI_API_VERSION` | Azure API version (default: `2024-08-01-preview`) |
| `AZURE_OPENAI_DEPLOYMENT` | Azure deployment name override (falls back to `LLM_MODEL`) |
| `AZURE_EMBEDDING_DEPLOYMENT` | Azure embedding deployment name (falls back to `EMBEDDING_MODEL`) |

### Requirements

- Python 3.10 or later
- `lightrag-hku >= 1.4.0` — installed as a dependency
- `snkv >= 0.7.0` — installed as a dependency

---

## License

Apache 2.0
