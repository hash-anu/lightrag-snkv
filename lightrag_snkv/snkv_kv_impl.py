"""SNKV-backed BaseKVStorage implementation for LightRAG.

One shared ``snkv_kv.db`` file per working-dir/workspace; each LightRAG
KV namespace gets its own SQLite column family inside that file.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, final

from lightrag.base import BaseKVStorage
from lightrag.utils import logger
from snkv import KVStore, NotFoundError


@final
@dataclass
class SNKVKVStorage(BaseKVStorage):
    def __post_init__(self) -> None:
        working_dir = self.global_config["working_dir"]
        if self.workspace:
            db_dir = os.path.join(working_dir, self.workspace)
            self.final_namespace = f"{self.workspace}_{self.namespace}"
        else:
            db_dir = working_dir
            self.workspace = ""
            self.final_namespace = self.namespace

        os.makedirs(db_dir, exist_ok=True)
        self._db_path = os.path.join(db_dir, "snkv_kv.db")
        self._cf_name = self.namespace
        # max_workers=1 guarantees thread-safety of the KVStore object
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"snkv_kv_{self.final_namespace}",
        )
        self._db: KVStore | None = None
        self._cf = None

    # ------------------------------------------------------------------
    # Sync helpers (run inside the executor thread)
    # ------------------------------------------------------------------

    def _open_db(self) -> None:
        self._db = KVStore(self._db_path)
        try:
            self._cf = self._db.open_column_family(self._cf_name)
        except NotFoundError:
            self._cf = self._db.create_column_family(self._cf_name)

    def _close_db(self) -> None:
        if self._cf is not None:
            try:
                self._cf.close()
            except Exception:
                pass
            self._cf = None
        if self._db is not None:
            try:
                self._db.close()
            except Exception:
                pass
            self._db = None

    def _ex(self):
        """Return the running loop; helper for less verbosity."""
        return asyncio.get_running_loop()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        await self._ex().run_in_executor(self._executor, self._open_db)

    async def finalize(self) -> None:
        await self._ex().run_in_executor(self._executor, self._close_db)
        self._executor.shutdown(wait=False)

    # ------------------------------------------------------------------
    # BaseKVStorage abstract methods
    # ------------------------------------------------------------------

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        def _get():
            raw = self._cf.get(id.encode())
            return json.loads(raw.decode()) if raw is not None else None

        return await self._ex().run_in_executor(self._executor, _get)

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any] | None]:
        def _get_many():
            out: list[dict[str, Any] | None] = []
            for doc_id in ids:
                raw = self._cf.get(doc_id.encode())
                out.append(json.loads(raw.decode()) if raw is not None else None)
            return out

        return await self._ex().run_in_executor(self._executor, _get_many)

    async def filter_keys(self, keys: set[str]) -> set[str]:
        """Return keys that are NOT present in storage."""
        def _filter():
            return {k for k in keys if not self._cf.exists(k.encode())}

        return await self._ex().run_in_executor(self._executor, _filter)

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if not data:
            return

        def _upsert():
            self._db.begin(write=True)
            try:
                for key, val in data.items():
                    self._cf.put(key.encode(), json.dumps(val, ensure_ascii=False).encode())
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise

        await self._ex().run_in_executor(self._executor, _upsert)

    async def delete(self, ids: list[str]) -> None:
        if not ids:
            return

        def _delete():
            self._db.begin(write=True)
            try:
                for doc_id in ids:
                    try:
                        self._cf.delete(doc_id.encode())
                    except NotFoundError:
                        pass
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise

        await self._ex().run_in_executor(self._executor, _delete)

    async def is_empty(self) -> bool:
        def _check():
            return self._cf.count() == 0

        return await self._ex().run_in_executor(self._executor, _check)

    async def index_done_callback(self) -> None:
        def _sync():
            self._db.sync()

        await self._ex().run_in_executor(self._executor, _sync)

    async def drop(self) -> dict[str, str]:
        def _drop():
            self._cf.clear()

        try:
            await self._ex().run_in_executor(self._executor, _drop)
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(f"[{self.workspace}] Error dropping {self.namespace}: {e}")
            return {"status": "error", "message": str(e)}
