# lightrag-snkv Code Overview

This document describes how the `lightrag-snkv` package works, with a function-by-function overview of each module.

## Package Purpose

`lightrag-snkv` provides SNKV-powered storage backends for LightRAG. It replaces LightRAG's default file-based storage (JSON files + NetworkX pickle) with two embedded SQLite files and an in-memory HNSW index. The package exposes:

- `SNKVKVStorage` — key-value storage
- `SNKVVectorStorage` — vector embeddings and similarity search
- `SNKVGraphStorage` — graph node/edge storage
- `SNKVDocStatusStorage` — document processing status
- `register()` and `register_with_lightrag()` — one-call wiring helpers

The package is intentionally thin: it adapts LightRAG's abstract storage interfaces to the SNKV API.

---

## Files on disk

```
<working_dir>/
├── snkv.db                          # KV cache + graph + doc status (shared)
├── snkv_vec.db                      # All vector namespaces (separate file)
├── snkv_vec.<namespace>.usearch     # HNSW sidecar (fast restart)
└── snkv_vec.<namespace>.usearch.nid # next_id stamp for sidecar validation
```

Two db files instead of dozens of JSON/pickle files. The separate vector file lets the KV/graph executor and the vector executor run on separate threads, enabling parallel insert + query.

---

## `lightrag_snkv.__init__`

Exports the public API:

```python
from lightrag_snkv import register, SNKVKVStorage, SNKVVectorStorage, SNKVGraphStorage, SNKVDocStatusStorage
```

---

## `lightrag_snkv.register`

### `register()`

Registers all four SNKV backends into LightRAG's internal storage registry (`STORAGES`, `STORAGE_IMPLEMENTATIONS`, `STORAGE_ENV_REQUIREMENTS`). Guards against double-registration. Must be called once before `LightRAG` is instantiated.

### `register_with_lightrag(rag)`

Convenience wrapper: calls `register()` then sets all four backend names on an existing `LightRAG` instance. Intended to be called before `await rag.initialize_storages()`.

---

## `lightrag_snkv.snkv_shared`

Manages one `KVStore` + one `ThreadPoolExecutor(max_workers=1)` per db file path, reference-counted across all storage adapters sharing that file.

### `SharedStore`

```
kv          snkv.KVStore — single SQLite connection
executor    ThreadPoolExecutor — serialises all SNKV access
ref_count   int — number of active storage adapters using this store
```

### `acquire(db_path) → SharedStore`

- Protected by a process-level `threading.Lock`
- Creates `KVStore` + executor on first call for a given path
- Increments `ref_count` on every call
- Returns the shared store

### `release(db_path)`

- Decrements `ref_count`
- When count reaches zero: closes `KVStore`, shuts down executor, removes registry entry

The single-thread executor guarantees that all reads and writes to a given db file are serialised — no locking needed inside the storage adapters themselves.

---

## `lightrag_snkv.snkv_kv_impl`

### `SNKVKVStorage`

Implements `BaseKVStorage`. One column family per namespace, stored in `snkv.db`.

#### `__post_init__()`

- Builds `db_dir` = `working_dir/workspace` if workspace is set, else `working_dir`
- `_db_path` = `<db_dir>/snkv.db`
- `_cf_name` = namespace name

#### `initialize()` / `finalize()`

Acquire/release the shared store; open/close the column family on the executor thread.

#### `get_by_id(id)` / `get_by_ids(ids)`

Straight reads from the column family; return decoded JSON or `None`.

#### `filter_keys(keys)`

Returns the subset of keys that do not yet exist in the store.

#### `upsert(data)`

Single write transaction — all keys committed atomically; rolls back on exception.

#### `delete(ids)`

Single write transaction — all deletes committed atomically; ignores missing keys.

#### `is_empty()`

Returns `True` when the column family count is zero.

#### `index_done_callback()`

Calls `kv.sync()` to flush WAL to disk.

#### `drop()`

Clears the entire column family. Returns `{"status": "success", ...}`.

---

## `lightrag_snkv.snkv_doc_status_impl`

### Helper functions

#### `_to_dict(obj)` / `_from_dict(d)`

Serialise/deserialise `DocProcessingStatus` dataclasses, converting the `status` enum to/from its string value.

### `SNKVDocStatusStorage`

Implements `DocStatusStorage`. Uses a single `doc_status` column family in `snkv.db`. Workspace isolation is handled by the subdirectory path, not by CF name prefixing.

#### `_iter_all()`

Full scan of the `doc_status` CF; returns a list of `(id, DocProcessingStatus)` pairs. Used by all status-query methods.

#### `upsert(data)` / `delete(ids)`

Both use single write transactions (same pattern as KV storage).

#### DocStatus-specific methods

| Method | Behaviour |
|--------|-----------|
| `get_status_counts()` | Full scan; counts by `status.value` |
| `get_all_status_counts()` | Alias for `get_status_counts()` |
| `get_docs_by_status(status)` | Full scan; filter by status |
| `get_docs_by_statuses(statuses)` | Full scan; filter by status set |
| `get_docs_by_track_id(track_id)` | Full scan; filter by track_id |
| `get_docs_paginated(...)` | Full scan + in-memory sort/slice |
| `get_doc_by_file_path(file_path)` | Full scan; return first match |

All queries do a full scan — acceptable for LightRAG's document counts (typically hundreds to low thousands).

---

## `lightrag_snkv.snkv_graph_impl`

### Constants and helpers

#### `_SEP = "||"`

Separator used in edge keys.

#### `_edge_key(src, tgt) → bytes`

Returns a canonical, undirected edge key: `min(src,tgt) || max(src,tgt)`. Ensures `edge(A,B) == edge(B,A)`.

### `SNKVGraphStorage`

Implements `BaseGraphStorage` using three column families in `snkv.db`:

| CF | Contents |
|----|----------|
| `nodes` | node_id → JSON metadata |
| `edges` | canonical edge key → JSON metadata |
| `adj` | node_id → JSON list of neighbour IDs |

#### Adjacency helpers

##### `_get_adj(node_id) → list[str]`

Reads the adjacency list for a node. Returns `[]` if none stored.

##### `_set_adj(node_id, neighbours)`

Writes the adjacency list as JSON. If `neighbours` is empty, deletes the key entirely (no empty-list tombstones).

> `_add_to_adj` and `_remove_from_adj` no longer exist. All adj mutations use the pre-load-then-write pattern described below.

#### Pre-load pattern (all write operations)

Every write operation that touches adjacency data follows this sequence to avoid SQLite read-within-write isolation issues:

1. **Pre-load** all relevant adj lists into Python `dict[str, set[str]]` — before any transaction begins
2. **Apply changes in-memory** on those dicts
3. **Begin transaction** — write edges, write updated adj lists, write node records
4. **Commit** (or rollback on exception)

This guarantees that reads inside a write transaction never see stale committed-but-not-yet-visible data.

#### Node operations

##### `has_node(node_id)` / `get_node(node_id)`

Point lookups on the `nodes` CF.

##### `upsert_node(node_id, node_data)`

Single put; inherently atomic.

##### `delete_node(node_id)`

Pre-load pattern:
1. Read this node's adj list (its neighbours)
2. Pre-load each neighbour's adj list into a dict; discard `node_id` from each in-memory
3. Begin transaction: write all updated neighbour adj lists, delete their edges with `node_id`, delete `node_id`'s own adj entry and node record
4. Commit

Fully cascading — no ghost edges or stale adj references left behind.

##### `upsert_nodes_batch(nodes)`

Single transaction over all nodes.

##### `has_nodes_batch(node_ids)` / `get_nodes_batch(node_ids)`

Batch reads; no transaction needed.

#### Edge operations

##### `has_edge(src, tgt)` / `get_edge(src, tgt)`

Point lookups using the canonical edge key.

##### `get_node_edges(source_node_id)`

Returns `[(source_node_id, nb) for nb in adj]` or `None` if the node doesn't exist.

##### `upsert_edge(src, tgt, edge_data)`

Pre-load pattern:
1. Pre-load `src_adj` and `tgt_adj`; add each to the other's set
2. Begin transaction: write edge, write both updated adj lists
3. Commit

##### `upsert_edges_batch(edges)`

Pre-load pattern:
1. Collect all affected node IDs; pre-load all their adj into `adj_map`
2. Apply all additions in-memory across the whole batch
3. Begin transaction: write all edges + all updated adj lists
4. Commit

##### `remove_nodes(nodes)`

Pre-load pattern:
1. Pre-load adj for every node being deleted
2. Identify external neighbours (nodes not in the delete set); pre-load their adj
3. Apply removals in-memory on external neighbours' adj sets
4. Begin transaction: delete all incident edges, delete adj + node records for deleted nodes, write updated adj for external neighbours
5. Commit

##### `remove_edges(edges)`

Pre-load pattern:
1. Pre-load adj for all affected nodes
2. Apply discards in-memory
3. Begin transaction: delete edge entries, write updated adj lists
4. Commit

##### `get_edges_batch(pairs)` / `get_nodes_edges_batch(node_ids)`

Batch reads using canonical edge keys / adj lists.

#### Degree operations

| Method | Returns |
|--------|---------|
| `node_degree(node_id)` | `len(adj)` for one node |
| `edge_degree(src, tgt)` | `len(adj_src) + len(adj_tgt)` |
| `node_degrees_batch(node_ids)` | dict of degrees |
| `edge_degrees_batch(edge_pairs)` | dict of degree sums |

#### Label / search operations

##### `get_all_labels()`

Full scan of `nodes` CF; returns all IDs sorted lexicographically.

##### `get_popular_labels(limit)`

Full scan; sorts nodes by adj-list length descending; returns top `limit`.

##### `search_labels(query, limit)`

Case-insensitive substring match first; falls back to `difflib.get_close_matches` for fuzzy results.

#### `get_knowledge_graph(node_label, max_depth, max_nodes)`

BFS from seed nodes (substring-matched or `"*"` for all). Uses `collections.deque` with `popleft()` for O(1) pop. Visits up to `max_nodes` nodes within `max_depth` hops. Returns `KnowledgeGraph(nodes, edges, is_truncated)`.

#### `index_done_callback()` / `drop()`

`index_done_callback` syncs the KVStore. `drop` clears all three CFs.

---

## `lightrag_snkv.snkv_vector_impl`

### Helper functions

#### `_pack_i64(n)` / `_unpack_i64(b)`

Pack/unpack 64-bit signed integers (big-endian). Map between SNKV byte keys and numeric usearch labels.

#### `_get_or_create_cf(kv, name)`

Opens or creates a named column family.

### `SNKVVectorStorage`

Implements `BaseVectorStorage`. Uses `snkv_vec.db` and an in-memory usearch HNSW index. Each namespace gets **six** column families:

| CF name | Contents |
|---------|----------|
| `vec_val_{ns}` | key → JSON metadata (`content`, `meta_fields`, `__created_at__`) |
| `vec_raw_{ns}` | key → raw float32 bytes |
| `vec_idk_{ns}` | key → int64 usearch label |
| `vec_idi_{ns}` | int64 usearch label → key |
| `vec_meta_{ns}` | persistent config (`next_id`) |
| `vec_rev_{ns}` | entity_name → JSON list of relation keys (reverse index) |

The reverse index (`vec_rev_{ns}`) is the key optimisation: it lets `delete_entity_relation` find all relations for an entity in O(1) instead of scanning every stored vector.

#### `__post_init__()`

- Reads `cosine_better_than_threshold` from `vector_db_storage_cls_kwargs` (required)
- Builds `_db_path` = `<db_dir>/snkv_vec.db`
- Builds `_sidecar_path` = `<db_dir>/snkv_vec.<namespace>.usearch`
- Stores dimension from `embedding_func.embedding_dim`
- Stores `_max_batch_size` from `global_config["embedding_batch_num"]`
- Declares all six CF name strings; initialises CF handles + index to `None`

#### Reverse-index helpers

##### `_get_rev(entity_name) → list[str]`

Reads `vec_rev_{ns}` for the given entity name. Returns `[]` if not present.

##### `_write_rev_map(rev_map: dict[str, list[str]])`

Writes a pre-computed in-memory reverse-index dict to `vec_rev_{ns}`. Called inside an open write transaction. If the list for an entity is empty, the key is deleted rather than storing an empty list.

#### `_open_store()`

1. Opens or creates all six column families
2. Loads `next_id` from `vec_meta_{ns}`
3. Builds a fresh `UsearchIndex`
4. Tries sidecar fast-path: if `snkv_vec.<ns>.usearch` exists and its `.nid` stamp matches `next_id`, restore the index directly (O(1))
5. If sidecar is missing, stale, or corrupt: rebuilds index by iterating `vec_raw_{ns}` + `vec_idk_{ns}`

#### `_close_store()`

Saves usearch index to the sidecar file, writes `.nid` stamp, closes all six CFs, clears in-memory index.

#### `initialize()` / `finalize()`

Acquire/release the shared `snkv_vec.db` store; run `_open_store`/`_close_store` on the executor thread.

### BaseVectorStorage methods

#### `upsert(data)`

Pre-load pattern for reverse index:

1. **Embedding**: batch-compute embeddings asynchronously
2. **Pre-read** (before transaction):
   - For each key: old int64 label from `vec_idk`; old val from `vec_val` to extract old `src_id`/`tgt_id`
   - Collect all affected entity names (old and new src/tgt)
   - Pre-load `rev_map` for all affected entities from `vec_rev`
3. **Apply rev changes in-memory**: remove key from old entities' lists; add key to new entities' lists (deduplicated)
4. **Transaction**:
   - Write `vec_val`, `vec_raw`, `vec_idk`, `vec_idi` for each key
   - Delete old `vec_idi` entry if key previously existed
   - Write updated `rev_map` via `_write_rev_map`
   - Write new `next_id` to `vec_meta`
   - Commit (rollback + restore `_next_id` on exception)
5. **Post-commit**: remove old usearch labels; add new labels + vectors to index; reserve capacity if >90% full

#### `query(query, top_k, query_embedding=None)`

Computes embedding if not supplied. Searches usearch index. Resolves labels → keys via `vec_idi`, keys → metadata via `vec_val`. Filters by `cosine_better_than_threshold`. Returns list of `{id, distance, created_at, ...metadata}`.

#### `get_by_id(id)` / `get_by_ids(ids)`

Point reads from `vec_val`; return metadata dicts or `None`.

#### `delete(ids)`

Pre-load pattern:

1. Pre-read int_id and `src_id`/`tgt_id` for each key
2. Pre-load `rev_map` for affected entities; apply removals in-memory
3. **Single transaction**: delete all `vec_val`/`vec_raw`/`vec_idk`/`vec_idi` entries + write updated `rev_map`
4. Post-commit: remove usearch labels

Previously used N separate transactions (one per key). Now one transaction for all keys.

#### `delete_entity(entity_name)`

Computes `entity_id = compute_mdhash_id(entity_name, prefix="ent-")`. Deletes that key's val/raw/idk/idi entries in a single transaction. No reverse index involvement — entity vectors do not carry `src_id`/`tgt_id`.

#### `delete_entity_relation(entity_name)`

**O(1) via reverse index** — previously O(n) full scan:

1. `_get_rev(entity_name)` → list of relation keys (O(1))
2. Pre-read int_ids and `src_id`/`tgt_id` for those keys
3. Pre-load `rev_map` for all affected entities; apply removals in-memory
4. **Single transaction**: delete all vector entries + write updated `rev_map`
5. Post-commit: remove usearch labels

#### `get_vectors_by_ids(ids)`

Returns `{id: [float, ...]}` for raw embedding vectors.

#### `index_done_callback()`

Calls `kv.sync()` to flush WAL to disk.

#### `drop()`

Clears all six column families, resets `_next_id = 0`, rebuilds a fresh empty usearch index, removes sidecar files. Returns status dict.

---

## Crash safety

| Scenario | Outcome |
|----------|---------|
| Crash mid-transaction | SQLite WAL rolls back automatically on next open |
| Crash after commit, before usearch update | Index rebuilt from `vec_raw_{ns}` on next open — deleted entries absent, consistent |
| Stale sidecar after crash | `.nid` stamp mismatch triggers full rebuild |
| Reverse index stale entry | `delete_entity_relation` handles empty `pre_data` by cleaning up the stale rev entry in a transaction |

---

## Summary of storage layout

| File | Column families | Used by |
|------|----------------|---------|
| `snkv.db` | one CF per KV namespace | `SNKVKVStorage` |
| `snkv.db` | `nodes`, `edges`, `adj` | `SNKVGraphStorage` |
| `snkv.db` | `doc_status` | `SNKVDocStatusStorage` |
| `snkv_vec.db` | 6 CFs per vector namespace | `SNKVVectorStorage` |

All adapters sharing the same db file share one `KVStore` connection and one serialising executor thread (via `snkv_shared`). The two db files run on separate executor threads, allowing parallel graph + vector operations.

---

## How to use

```python
from lightrag import LightRAG, QueryParam
from lightrag_snkv import register

register()   # once per process

rag = LightRAG(
    working_dir="./my_rag",
    llm_model_func=...,
    embedding_func=...,
    kv_storage="SNKVKVStorage",
    vector_storage="SNKVVectorStorage",
    graph_storage="SNKVGraphStorage",
    doc_status_storage="SNKVDocStatusStorage",
)
await rag.initialize_storages()
await rag.ainsert("Your text")
result = await rag.aquery("Your question", param=QueryParam(mode="hybrid"))
await rag.finalize_storages()
```

---

## Design constraints

- **Single process only**: the usearch HNSW index is in-memory per process. Two processes on the same `working_dir` would have diverging indices.
- **Serialised writes**: all writes to a given db file go through one thread. Throughput is bounded by single-thread performance — appropriate for LightRAG's workload.
- **`cosine_better_than_threshold` is required**: must be set in `vector_db_storage_cls_kwargs` or initialisation will fail with `ValueError`.
