"""SNKV-backed BaseVectorStorage for LightRAG.

Built on SNKV's native vector store (``snkv.vector.VectorStore``): storage,
the HNSW index and its ``.usearch`` sidecar are all handled by the SNKV C
library, so no separate vector package is needed.

Each LightRAG vector namespace (entities, relationships, chunks) gets its own
file, ``snkv_vec_{namespace}.db``, because one VectorStore holds one index.

Stored per key:
  value     JSON payload: ``__created_at__``, the namespace's meta fields, content
  vector    float32 embedding, indexed for cosine search

Reverse index (relationships namespace): plain KV entries
``_REV_PREFIX + entity_name -> JSON list of relation keys`` in the same file,
so deleting an entity's relations needs no full scan. Relation keys are hashes
of their two entity names, so a key's entities never change. Entries are
written before the vectors and cleaned after deletes; after a crash the index
can hold extra keys (ignored on use) but never misses one.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, final

import numpy as np
from lightrag.base import BaseVectorStorage
from lightrag.utils import compute_mdhash_id, logger
from snkv import NotFoundError
from snkv.vector import VectorStore

_REV_PREFIX = b"\x00rev\x00"  # cannot collide with LightRAG ids (ent-/rel-/chunk- hashes)
_MIGRATE_BATCH = 500


def _rev_key(entity_name: str) -> bytes:
    return _REV_PREFIX + entity_name.encode()


def _decode_payload(raw: bytes | None) -> dict[str, Any] | None:
    if raw is None:
        return None
    try:
        return json.loads(raw.decode())
    except (ValueError, UnicodeDecodeError):
        return {}


def _vector_files(db_path: str) -> list[str]:
    return [db_path, f"{db_path}-wal", f"{db_path}-shm",
            f"{db_path}.usearch", f"{db_path}.usearch.nid"]


@final
@dataclass
class SNKVVectorStorage(BaseVectorStorage):
    def __post_init__(self) -> None:
        self._validate_embedding_func()

        kwargs = self.global_config.get("vector_db_storage_cls_kwargs", {})
        cosine_threshold = kwargs.get("cosine_better_than_threshold")
        if cosine_threshold is None:
            raise ValueError(
                "cosine_better_than_threshold must be set in vector_db_storage_cls_kwargs"
            )
        self.cosine_better_than_threshold = cosine_threshold

        working_dir = self.global_config["working_dir"]
        if self.workspace:
            db_dir = os.path.join(working_dir, self.workspace)
            self.final_namespace = f"{self.workspace}_{self.namespace}"
        else:
            db_dir = working_dir
            self.workspace = ""
            self.final_namespace = self.namespace

        os.makedirs(db_dir, exist_ok=True)
        self._db_dir = db_dir
        self._db_path = os.path.join(db_dir, f"snkv_vec_{self.namespace}.db")
        self._dim = self.embedding_func.embedding_dim
        self._max_batch_size = self.global_config["embedding_batch_num"]

        self._vs: VectorStore | None = None
        # One thread per storage: keeps reverse-index read-modify-writes ordered.
        self._executor: ThreadPoolExecutor | None = None

    async def _run(self, fn):
        return await asyncio.get_running_loop().run_in_executor(self._executor, fn)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _open_store(self) -> None:
        is_new = not os.path.exists(self._db_path)
        self._vs = VectorStore(
            self._db_path, dim=self._dim, space="cosine", expansion_search=64
        )
        if is_new:
            self._migrate_legacy()

    async def initialize(self) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"snkv_vec_{self.namespace}"
        )
        await self._run(self._open_store)

    async def finalize(self) -> None:
        if self._executor is None:
            return

        def _close():
            if self._vs is not None:
                self._vs.close()  # also saves the .usearch sidecar
                self._vs = None

        await self._run(_close)
        self._executor.shutdown(wait=True)
        self._executor = None

    def _migrate_legacy(self) -> None:
        """Copy this namespace out of the pre-0.2 shared ``snkv_vec.db``
        (column families ``vec_val_{ns}`` / ``vec_raw_{ns}``), if present.
        The old file is left untouched."""
        legacy_path = os.path.join(self._db_dir, "snkv_vec.db")
        if not os.path.exists(legacy_path):
            return
        from snkv import KVStore

        val_name, raw_name = f"vec_val_{self.namespace}", f"vec_raw_{self.namespace}"
        old = KVStore(legacy_path)
        try:
            if not {val_name, raw_name} <= set(old.list_column_families()):
                return
            val_cf = old.open_column_family(val_name)
            raw_cf = old.open_column_family(raw_name)
            batch: list[tuple[bytes, bytes, np.ndarray]] = []
            rev: dict[str, list[str]] = {}
            moved = 0
            with val_cf.iterator() as it:
                for key_b, val_b in it:
                    vec_b = raw_cf.get(key_b)
                    if vec_b is None:
                        continue
                    vec = np.frombuffer(vec_b, dtype=np.float32)
                    if vec.shape[0] != self._dim:
                        logger.warning(
                            f"[{self.workspace}] Not migrating {self.namespace}: stored "
                            f"vectors have {vec.shape[0]} dims, embedding model has {self._dim}"
                        )
                        return
                    batch.append((key_b, bytes(val_b), vec))
                    payload = _decode_payload(val_b) or {}
                    for e in (payload.get("src_id"), payload.get("tgt_id")):
                        if e:
                            rev.setdefault(e, []).append(key_b.decode())
                    if len(batch) >= _MIGRATE_BATCH:
                        self._vs.vector_put_batch(batch)
                        moved += len(batch)
                        batch = []
            if batch:
                self._vs.vector_put_batch(batch)
                moved += len(batch)
            for entity, keys in rev.items():
                self._vs[_rev_key(entity)] = json.dumps(keys).encode()
            val_cf.close()
            raw_cf.close()
            logger.info(
                f"[{self.workspace}] Migrated {moved} {self.namespace} vectors from "
                f"snkv_vec.db to {os.path.basename(self._db_path)}"
            )
        finally:
            old.close()

    # ------------------------------------------------------------------
    # Reverse index helpers (executor thread)
    # ------------------------------------------------------------------

    def _get_rev(self, entity_name: str) -> list[str]:
        raw = self._vs.get(_rev_key(entity_name))
        return json.loads(raw.decode()) if raw else []

    def _set_rev(self, entity_name: str, keys: list[str]) -> None:
        if keys:
            self._vs[_rev_key(entity_name)] = json.dumps(keys).encode()
        else:
            try:
                del self._vs[_rev_key(entity_name)]
            except NotFoundError:
                pass

    def _delete_keys(self, keys: list[str]) -> int:
        """Delete vectors and drop them from their entities' reverse index."""
        drop_from: dict[str, set[str]] = {}
        deleted = 0
        for key in keys:
            key_b = key.encode()
            payload = _decode_payload(self._vs.get(key_b))
            if payload is None:
                continue
            for e in (payload.get("src_id"), payload.get("tgt_id")):
                if e:
                    drop_from.setdefault(e, set()).add(key)
            try:
                self._vs.delete(key_b)
                deleted += 1
            except NotFoundError:
                pass
        for entity, gone in drop_from.items():
            self._set_rev(entity, [k for k in self._get_rev(entity) if k not in gone])
        return deleted

    @staticmethod
    def _row(key: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {"id": key, "created_at": payload.pop("__created_at__", None), **payload}

    # ------------------------------------------------------------------
    # BaseVectorStorage methods
    # ------------------------------------------------------------------

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if not data:
            return

        current_time = int(time.time())
        keys = list(data.keys())
        contents = [v["content"] for v in data.values()]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]
        embeddings_list = await asyncio.gather(*[self.embedding_func(b) for b in batches])
        embeddings = np.concatenate(embeddings_list)

        if len(embeddings) != len(keys):
            logger.error(
                f"[{self.workspace}] embedding count mismatch "
                f"{len(embeddings)} != {len(keys)} for {self.namespace}"
            )
            return

        def _put():
            # Reverse index first, so a crash can only leave extra entries.
            additions: dict[str, list[str]] = {}
            for key, val in data.items():
                for e in (val.get("src_id"), val.get("tgt_id")):
                    if e:
                        additions.setdefault(e, []).append(key)
            for entity, new_keys in additions.items():
                current = self._get_rev(entity)
                merged = current + [k for k in dict.fromkeys(new_keys) if k not in current]
                if len(merged) != len(current):
                    self._set_rev(entity, merged)

            items = []
            for key, val, emb in zip(keys, data.values(), embeddings):
                stored = {
                    "__created_at__": current_time,
                    **{k: v for k, v in val.items() if k in self.meta_fields},
                    "content": val.get("content", ""),
                }
                items.append((
                    key.encode(),
                    json.dumps(stored, ensure_ascii=False).encode(),
                    np.asarray(emb, dtype=np.float32),
                ))
            self._vs.vector_put_batch(items)  # one transaction

        await self._run(_put)

    async def query(
        self, query: str, top_k: int, query_embedding: list[float] = None
    ) -> list[dict[str, Any]]:
        if query_embedding is not None:
            q_vec = np.asarray(query_embedding, dtype=np.float32)
        else:
            emb = await self.embedding_func([query], _priority=5)
            q_vec = np.asarray(emb[0], dtype=np.float32)

        max_dist = 1.0 - self.cosine_better_than_threshold

        def _search():
            if len(self._vs) == 0:
                return []
            out = []
            for r in self._vs.search(q_vec, top_k=top_k, max_distance=max_dist):
                payload = _decode_payload(r.value) or {}
                row = self._row(r.key.decode(), payload)
                row["distance"] = 1.0 - r.distance  # LightRAG expects similarity
                out.append(row)
            return out

        return await self._run(_search)

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        def _get():
            payload = _decode_payload(self._vs.get(id.encode()))
            return None if payload is None else self._row(id, payload)

        return await self._run(_get)

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any] | None]:
        def _get_many():
            out: list[dict[str, Any] | None] = []
            for id_str in ids:
                payload = _decode_payload(self._vs.get(id_str.encode()))
                out.append(None if payload is None else self._row(id_str, payload))
            return out

        return await self._run(_get_many)

    async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        def _get_vecs():
            out: dict[str, list[float]] = {}
            for id_str in ids:
                try:
                    out[id_str] = self._vs.vector_get(id_str.encode()).tolist()
                except NotFoundError:
                    pass
            return out

        return await self._run(_get_vecs)

    async def delete(self, ids: list[str]) -> None:
        if ids:
            await self._run(lambda: self._delete_keys(list(ids)))

    async def delete_entity(self, entity_name: str) -> None:
        entity_id = compute_mdhash_id(entity_name, prefix="ent-")
        await self._run(lambda: self._delete_keys([entity_id]))

    async def delete_entity_relation(self, entity_name: str) -> None:
        def _delete_relations():
            keys = self._get_rev(entity_name)
            if not keys:
                return
            deleted = self._delete_keys(keys)
            self._set_rev(entity_name, [])  # also clears keys that were already gone
            logger.debug(
                f"[{self.workspace}] deleted {deleted} relations for {entity_name}"
            )

        await self._run(_delete_relations)

    async def index_done_callback(self) -> None:
        # Every write above commits its own transaction; the HNSW sidecar is
        # saved on close and rebuilt from the database if it is ever stale.
        return None

    async def drop(self) -> dict[str, str]:
        def _drop():
            if self._vs is not None:
                self._vs.close()
                self._vs = None
            for p in _vector_files(self._db_path):
                try:
                    os.unlink(p)
                except FileNotFoundError:
                    pass
            self._vs = VectorStore(
                self._db_path, dim=self._dim, space="cosine", expansion_search=64
            )

        try:
            await self._run(_drop)
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(f"[{self.workspace}] Error dropping {self.namespace}: {e}")
            return {"status": "error", "message": str(e)}
