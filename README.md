# lightrag-snkv

**Run [LightRAG](https://github.com/HKUDS/LightRAG) with zero extra databases.**

`lightrag-snkv` is a storage backend for LightRAG built on the embedded [SNKV](https://github.com/hash-anu/snkv) database. It covers **all four** LightRAG storage types — KV, vector, graph, and document status — in a single folder of local files.

No Postgres. No Neo4j. No Qdrant, Milvus, Redis, or MongoDB. No Docker. Just `pip install` and go.

---

## Why lightrag-snkv?

A production LightRAG setup usually means running a separate service for each storage type:

| Storage type | Typical choice | With lightrag-snkv |
|---|---|---|
| KV storage | Redis / Postgres / MongoDB | `snkv.db` |
| Vector storage | Qdrant / Milvus / pgvector | `snkv_vec_*.db` (SNKV native vector store) |
| Graph storage | Neo4j / Memgraph / Postgres AGE | `snkv.db` |
| Doc status storage | Postgres / MongoDB | `snkv.db` |
| **Services to run** | 2–4 | **0** |

**One install, nothing to operate.**
Everything runs in-process. No servers to install, configure, secure, upgrade, or keep alive.

**Safer than the default file backend.**
LightRAG's built-in default writes JSON files plus a NetworkX pickle. Every multi-step write in lightrag-snkv (graph update, vector upsert, batch delete) runs inside a single SNKV transaction — it either commits fully or not at all. A crash mid-write can't leave your data half-written.

**Fast restarts.**
The HNSW vector index is saved to a sidecar file and checked on open. If it matches, it loads in milliseconds instead of being rebuilt entry by entry.

**Easy to back up and move.**
Your whole knowledge base is one folder of SNKV files. Copy it, ship it, version it.

**Drop-in.**
The full LightRAG API — `ainsert`, `aquery`, `QueryParam`, all query modes, the web UI — works unchanged. You change four storage class names.

---

## Install

Requires Python 3.10+.

```bash
git clone https://github.com/hash-anu/lightrag-snkv.git
cd lightrag-snkv

# Library use
pip install -e ".[vector]"

# Library + LightRAG web UI / API server
pip install -e ".[vector,server]"

# LLM / embedding credentials
cp .env.example .env
```

Dependencies: `lightrag-hku >= 1.5.7` and `snkv >= 0.9.0`.

---

## Use it in code

One import, one `register()` call, four constructor arguments. Everything else in your LightRAG code stays the same.

```python
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag_snkv import register          # ← add

register()                                  # ← call once at startup

async def main():
    rag = LightRAG(
        working_dir="./my_rag",
        llm_model_func=your_llm_func,       # unchanged
        embedding_func=your_embed_func,     # unchanged
        kv_storage="SNKVKVStorage",         # ← add these 4 lines
        vector_storage="SNKVVectorStorage",
        graph_storage="SNKVGraphStorage",
        doc_status_storage="SNKVDocStatusStorage",
    )

    await rag.initialize_storages()

    await rag.ainsert("Marie Curie discovered polonium and radium.")
    result = await rag.aquery("Who discovered radium?", param=QueryParam(mode="hybrid"))
    print(result)

    await rag.finalize_storages()

asyncio.run(main())
```

### What ends up on disk

```
my_rag/
├── snkv.db                              # KV cache + graph + doc status
├── snkv_vec_entities.db                 # entity embeddings
├── snkv_vec_relationships.db            # relationship embeddings
├── snkv_vec_chunks.db                   # text chunk embeddings
└── snkv_vec_*.db.usearch                # saved HNSW indexes (fast restart)
```

---

## Web UI

LightRAG ships a full web interface. Run it on SNKV with the included `server.py` instead of `lightrag-server` — it registers SNKV and sets all four storage types before the server starts:

```bash
python server.py
```

Open `http://localhost:9621`. Host and port come from `HOST` / `PORT` in your `.env` (default `0.0.0.0:9621`).

---

## Reuse your existing LightRAG `.env`

Already running LightRAG? Load your existing `.env` and build the LLM and embedding functions from it:

```python
import asyncio
from dotenv import load_dotenv
from lightrag import LightRAG, QueryParam
from lightrag_snkv import register, get_llm_and_embed_funcs

load_dotenv()
llm_func, embed_func = get_llm_and_embed_funcs()
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
    print(await rag.aquery("Your question", param=QueryParam(mode="hybrid")))
    await rag.finalize_storages()

asyncio.run(main())
```

`get_llm_and_embed_funcs()` reads the same variables as the LightRAG server:

| Variable | Purpose |
|---|---|
| `LLM_BINDING` | `openai`, `azure_openai`, `ollama` |
| `LLM_MODEL` | Model name (e.g. `gpt-4o-mini`) |
| `LLM_BINDING_HOST` | Custom endpoint (optional) |
| `LLM_BINDING_API_KEY` | API key (falls back to `OPENAI_API_KEY`) |
| `EMBEDDING_BINDING` | `openai`, `azure_openai`, `ollama` |
| `EMBEDDING_MODEL` | Embedding model name |
| `EMBEDDING_DIM` | Vector dimension (default: 1536) |
| `EMBEDDING_BINDING_HOST` | Custom endpoint (optional) |
| `EMBEDDING_BINDING_API_KEY` | API key (falls back to `OPENAI_API_KEY`) |
| `AZURE_OPENAI_API_VERSION` | Azure API version (default: `2024-08-01-preview`) |
| `AZURE_OPENAI_DEPLOYMENT` | Azure deployment name (falls back to `LLM_MODEL`) |
| `AZURE_EMBEDDING_DEPLOYMENT` | Azure embedding deployment (falls back to `EMBEDDING_MODEL`) |

---

## Good to know

- **Upgrading from an older version?** Vectors used to live in one shared `snkv_vec.db`. On first start, each namespace is copied into its new `snkv_vec_<namespace>.db` automatically. The old file is left untouched; delete it once everything looks right.
- **One writer process per folder.** SNKV is embedded, so don't point two separate processes at the same `working_dir` at the same time. Inside a single process (including `server.py`) it's safe.
- **Benchmarks.** `bench/` compares SNKV against LightRAG's default storage: `pip install -e ".[bench]"` then `lightrag-snkv-bench`.
- **Tests.** `pip install -e ".[vector,test]"` then `pytest`.

---

## License

Apache 2.0
