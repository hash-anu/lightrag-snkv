"""SNKV-backed BaseVectorStorage implementation for LightRAG.

One ``snkv_vec_{namespace}.db`` file per vector namespace (entities, relations,
chunks).  Each file is a VectorStore with HNSW index (cosine, f32).

The full data dict is stored as the VectorStore value (JSON-encoded) so query
results include all meta fields that LightRAG expects without an extra KV round-
trip.
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

try:
    from snkv.vector import VectorStore, VectorIndexError
except ImportError as exc:
    raise ImportError(
        "snkv[vector] is required for SNKVVectorStorage.\n"
        "Install with:  pip install snkv[vector]"
    ) from exc

from snkv import NotFoundError


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
        self._db_path = os.path.join(db_dir, f"snkv_vec_{self.namespace}.db")
        self._dim = self.embedding_func.embedding_dim
        self._max_batch_size = self.global_config["embedding_batch_num"]
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"snkv_vec_{self.final_namespace}",
        )
        self._store: VectorStore | None = None

    # ------------------------------------------------------------------
    # Sync helpers
    # ------------------------------------------------------------------

    def _open_store(self) -> None:
        self._store = VectorStore(
            self._db_path,
            dim=self._dim,
            space="cosine",
            dtype="f32",
        )

    def _close_store(self) -> None:
        if self._store is not None:
            try:
                self._store.close()
            except Exception:
                pass
            self._store = None

    def _ex(self):
        return asyncio.get_running_loop()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        await self._ex().run_in_executor(self._executor, self._open_store)

    async def finalize(self) -> None:
        await self._ex().run_in_executor(self._executor, self._close_store)
        self._executor.shutdown(wait=False)

    # ------------------------------------------------------------------
    # BaseVectorStorage abstract methods
    # ------------------------------------------------------------------

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Embed and store all items in a single batch."""
        if not data:
            return

        current_time = int(time.time())
        keys = list(data.keys())
        contents = [v["content"] for v in data.values()]

        # Batch embed (outside executor to keep async concurrency)
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]
        embedding_tasks = [self.embedding_func(batch) for batch in batches]
        embeddings_list = await asyncio.gather(*embedding_tasks)
        embeddings = np.concatenate(embeddings_list)

        if len(embeddings) != len(keys):
            logger.error(
                f"[{self.workspace}] embedding count mismatch "
                f"{len(embeddings)} != {len(keys)} for {self.namespace}"
            )
            return

        def _batch_put():
            items = []
            for key, val, emb in zip(keys, data.values(), embeddings):
                stored = {
                    "__created_at__": current_time,
                    **{k: v for k, v in val.items() if k in self.meta_fields},
                    "content": val.get("content", ""),
                }
                items.append(
                    (key.encode(), json.dumps(stored, ensure_ascii=False).encode(),
                     emb.astype(np.float32))
                )
            self._store.vector_put_batch(items)

        await self._ex().run_in_executor(self._executor, _batch_put)

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
            try:
                results = self._store.search(q_vec, top_k=top_k, max_distance=max_dist)
            except VectorIndexError:
                return []
            out = []
            for r in results:
                try:
                    val = json.loads(r.value.decode())
                except Exception:
                    val = {}
                out.append({
                    "id": r.key.decode(),
                    "distance": 1.0 - r.distance,  # return as cosine similarity
                    "created_at": val.pop("__created_at__", None),
                    **val,
                })
            return out

        return await self._ex().run_in_executor(self._executor, _search)

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        def _get():
            raw = self._store.get(id.encode())
            if raw is None:
                return None
            val = json.loads(raw.decode())
            return {
                "id": id,
                "created_at": val.pop("__created_at__", None),
                **val,
            }

        return await self._ex().run_in_executor(self._executor, _get)

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any] | None]:
        def _get_many():
            out: list[dict[str, Any] | None] = []
            for id_str in ids:
                raw = self._store.get(id_str.encode())
                if raw is None:
                    out.append(None)
                else:
                    val = json.loads(raw.decode())
                    out.append({
                        "id": id_str,
                        "created_at": val.pop("__created_at__", None),
                        **val,
                    })
            return out

        return await self._ex().run_in_executor(self._executor, _get_many)

    async def delete(self, ids: list[str]) -> None:
        if not ids:
            return

        def _delete():
            for id_str in ids:
                try:
                    self._store.delete(id_str.encode())
                except NotFoundError:
                    pass
                except Exception as e:
                    logger.warning(f"[{self.workspace}] delete {id_str}: {e}")

        await self._ex().run_in_executor(self._executor, _delete)

    async def delete_entity(self, entity_name: str) -> None:
        entity_id = compute_mdhash_id(entity_name, prefix="ent-")

        def _delete():
            try:
                self._store.delete(entity_id.encode())
                logger.debug(f"[{self.workspace}] deleted entity {entity_name}")
            except NotFoundError:
                logger.debug(f"[{self.workspace}] entity {entity_name} not found")
            except Exception as e:
                logger.error(f"[{self.workspace}] delete_entity {entity_name}: {e}")

        await self._ex().run_in_executor(self._executor, _delete)

    async def delete_entity_relation(self, entity_name: str) -> None:
        """Scan all entries and delete those involving entity_name as src/tgt."""
        def _delete_relations():
            to_delete: list[bytes] = []
            # Iterate default CF of underlying KVStore to find matching entries
            try:
                for key_b, val_b in self._store._kv.iterator():
                    try:
                        val = json.loads(val_b.decode())
                        if (val.get("src_id") == entity_name
                                or val.get("tgt_id") == entity_name):
                            to_delete.append(key_b)
                    except Exception:
                        pass
            except Exception as e:
                logger.error(f"[{self.workspace}] scanning relations: {e}")
                return

            for key_b in to_delete:
                try:
                    self._store.delete(key_b)
                except Exception:
                    pass

            logger.debug(
                f"[{self.workspace}] deleted {len(to_delete)} relations for {entity_name}"
            )

        await self._ex().run_in_executor(self._executor, _delete_relations)

    async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        def _get_vecs():
            out: dict[str, list[float]] = {}
            for id_str in ids:
                try:
                    vec = self._store.vector_get(id_str.encode())
                    out[id_str] = vec.tolist()
                except (NotFoundError, Exception):
                    pass
            return out

        return await self._ex().run_in_executor(self._executor, _get_vecs)

    async def index_done_callback(self) -> None:
        def _sync():
            self._store._kv.sync()

        await self._ex().run_in_executor(self._executor, _sync)

    async def drop(self) -> dict[str, str]:
        def _drop():
            self._close_store()
            try:
                os.remove(self._db_path)
            except FileNotFoundError:
                pass
            # remove sidecar if present
            for ext in (".usearch", ".usearch.nid"):
                try:
                    os.remove(self._db_path + ext)
                except FileNotFoundError:
                    pass
            self._open_store()

        try:
            await self._ex().run_in_executor(self._executor, _drop)
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(f"[{self.workspace}] Error dropping {self.namespace}: {e}")
            return {"status": "error", "message": str(e)}
