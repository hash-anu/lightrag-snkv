# lightrag-snkv

A storage backend for [LightRAG](https://github.com/HKUDS/LightRAG) built on [SNKV](https://github.com/hash-anu/snkv) database.

---

## Featured: Hacker News knowledge graph

Ask questions across thousands of Ask HN and Show HN posts — including every comment thread — without leaving your terminal.

<video src="https://github.com/user-attachments/assets/23a95319-ee66-49bc-811d-79fea94779b2" controls width="100%"></video>

```bash
pip install -e ".[vector,server,hn]"

# Backfill the last 4 weeks of HN discussions, then start the server
python ingest_hn.py --lookback 4w
python server.py
```

Open `http://localhost:9621` and ask:

- *What problems are developers struggling with right now?*
- *Which trending projects launched on Show HN this week?*
- *What are people building with LLMs?*

The knowledge graph captures not just what was posted but **what the community said about it**. Each post is ingested with its full comment tree, so the answers draw on real developer conversations.

For continuous updates, set `HN_SCHEDULE=daily` in `.env` and run only `server.py` — the ingestion daemon runs inside the same process automatically.

→ [Full HN ingestion docs](#hacker-news-ingestion)

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

# Copy and fill in your LLM / embedding credentials
cp .env.example .env
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

### Web UI

LightRAG ships a full web interface. To run it with SNKV as the storage backend, use the included `server.py` instead of the standard `lightrag-server` command — it registers SNKV before the server starts:

```bash
python server.py
```

Then open `http://localhost:9621` in your browser.

`server.py` loads your `.env` automatically and overrides the storage settings to use SNKV. The host and port are read from `HOST` and `PORT` in your `.env` (defaulting to `0.0.0.0:9621`).

### Hacker News Ingestion

`lightrag-snkv` can automatically pull **Ask HN** and **Show HN** posts from Hacker News — including their full comment discussions — and build a searchable knowledge graph from them.

Each ingested post becomes a rich document containing:
- Post title, author, score, and date
- Post body (if present)
- All comments and replies (fetched from the HN Algolia items API)

This means you can ask questions not just about what was posted, but what the community said about it.

#### Install

```bash
pip install -e ".[vector,server,hn]"
```

#### Quick start — run once (backfill)

```bash
# Fetch last 1 week of Show HN + Ask HN posts (score ≥ 10), then exit
python ingest_hn.py --lookback 1w

# Show HN only
python ingest_hn.py --lookback 1w --tags show_hn

# For recent posts (< 3 days), lower min-score since new posts haven't accumulated upvotes yet
python ingest_hn.py --lookback 2d --tags show_hn --min-score 3
```

#### Quick start — auto-ingest with the server

Add these lines to your `.env`:

```ini
HN_SCHEDULE=daily          # fetch new posts every 24 hours automatically
HN_LOOKBACK=1w             # on first run, go back 1 week
HN_MIN_SCORE=10            # only keep posts with 10+ upvotes
WORKING_DIR=./hn_rag_storage   # point server at HN knowledge graph
```

Then start the server:

```bash
python server.py
```

On first boot it backfills the past week of posts. Every 24 hours after that it fetches only new ones automatically in the background. Open `http://localhost:9621` to query the knowledge graph.

#### Run manually (without the server)

```bash
# Fetch last 1 week, then exit
python ingest_hn.py --lookback 1w

# Fetch last 2 years, then exit
python ingest_hn.py --lookback 2y

# Show HN only, last 3 days
python ingest_hn.py --lookback 3d --tags show_hn

# Run as a daily daemon (keeps running forever, fetches new posts every 24h)
python ingest_hn.py --schedule daily

# Weekly daemon, high-quality posts only
python ingest_hn.py --schedule weekly --min-score 50
```

#### Options

| Option | `.env` variable | Default | What it does |
|---|---|---|---|
| `--lookback` | `HN_LOOKBACK` | `4w` | How far back to fetch on the **first run only**. After that, only new posts are fetched incrementally. Use `1y`, `2y`, `4w`, `3d`, `90d` etc. |
| `--schedule` | `HN_SCHEDULE` | *(none)* | `daily` — fetch new posts every 24 hours. `weekly` — fetch every 7 days. Leave unset to run once and exit. When set in `.env`, the daemon starts automatically with `server.py`. |
| `--min-score` | `HN_MIN_SCORE` | `10` | Only ingest posts with at least this many upvotes. For posts less than 3 days old, use `1`–`5` since they haven't had time to accumulate votes. |
| `--tags` | `HN_TAGS` | `ask_hn,show_hn` | Which post types to fetch. `ask_hn` = Ask HN discussions. `show_hn` = Show HN project announcements. Pass one or both comma-separated. |
| `--batch-size` | `HN_BATCH_SIZE` | `10` | How many posts are sent to the knowledge graph at once. Lower if you hit LLM rate limits. |
| `--working-dir` | `HN_WORKING_DIR` | `./hn_rag_storage` | Where to store the HN knowledge graph. Kept separate from the main RAG storage. |

#### How much data to expect

| Lookback | Tags | Min score | Stories |
|---|---|---|---|
| 1 day | show_hn | 10 | ~5–15 |
| 2 days | show_hn | 10 | ~13–30 |
| 1 week | ask_hn + show_hn | 10 | ~900 |
| 4 weeks | ask_hn + show_hn | 10 | ~4,000 |

**Cost estimate (gpt-4o-mini):**
- 1-week backfill: ~$1–2 one-time
- Daily incremental runs: cents per day (LLM cache further reduces repeat costs)

#### What gets stored

```
hn_rag_storage/
├── snkv.db                    # knowledge graph + post metadata
├── snkv_vec.db                # post embeddings for semantic search
├── snkv_vec.*.usearch         # fast-restart vector index files
└── hn_state.json              # tracks ingested posts and last run timestamp
```

#### Important: avoid running two processes on the same database

Do **not** run `ingest_hn.py` and `server.py` pointing to the same `hn_rag_storage` directory at the same time. SQLite does not support concurrent writes from separate processes and will corrupt the database.

Two safe approaches:

**Option A — finish backfill first, then start server:**
```bash
# Step 1: run backfill to completion
python ingest_hn.py --lookback 1w

# Step 2: start server once ingestion is done
python server.py
```

**Option B — integrated daemon (recommended for ongoing use):**
Set `HN_SCHEDULE=daily` in `.env` and run only `server.py`. The ingestion daemon starts inside the same process, sharing the same database connection safely.

```ini
HN_SCHEDULE=daily
WORKING_DIR=./hn_rag_storage
```

```bash
python server.py   # serves + ingests in one process — no conflict
```

#### What to ask

Once ingested, try questions like:

- *What are the most talked about projects on Show HN this week?*
- *What are developers asking about on Ask HN recently?*
- *What are people building with AI/LLMs right now?*
- *What new open source tools launched this week?*
- *What problems are developers struggling with?*
- *What did the community say about recent Show HN projects?*

---

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
