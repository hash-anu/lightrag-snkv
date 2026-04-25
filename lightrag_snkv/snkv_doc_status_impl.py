"""SNKV-backed DocStatusStorage implementation for LightRAG.

One ``snkv_doc.db`` per working-dir/workspace with a single ``doc_status``
column family.  DocProcessingStatus objects are stored as JSON.
"""
from __future__ import annotations

import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any, final

from lightrag.base import DocProcessingStatus, DocStatus, DocStatusStorage
from lightrag.utils import logger
from snkv import KVStore, NotFoundError


def _to_dict(obj: DocProcessingStatus) -> dict:
    d = asdict(obj)
    d["status"] = obj.status.value
    return d


def _from_dict(d: dict) -> DocProcessingStatus:
    d = dict(d)
    d["status"] = DocStatus(d["status"])
    return DocProcessingStatus(**d)


@final
@dataclass
class SNKVDocStatusStorage(DocStatusStorage):
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
        self._db_path = os.path.join(db_dir, "snkv_doc.db")
        self._cf_name = "doc_status"
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"snkv_doc_{self.final_namespace}",
        )
        self._db: KVStore | None = None
        self._cf = None

    # ------------------------------------------------------------------
    # Sync helpers
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

    def _iter_all(self) -> list[tuple[str, DocProcessingStatus]]:
        out: list[tuple[str, DocProcessingStatus]] = []
        with self._cf.iterator() as it:
            for key_b, val_b in it:
                try:
                    d = json.loads(val_b.decode())
                    out.append((key_b.decode(), _from_dict(d)))
                except Exception:
                    pass
        return out

    def _ex(self):
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
            logger.error(f"[{self.workspace}] Error dropping doc_status: {e}")
            return {"status": "error", "message": str(e)}

    # ------------------------------------------------------------------
    # DocStatusStorage abstract methods
    # ------------------------------------------------------------------

    async def get_status_counts(self) -> dict[str, int]:
        def _count():
            counts: dict[str, int] = {}
            for _, doc in self._iter_all():
                key = doc.status.value
                counts[key] = counts.get(key, 0) + 1
            return counts

        return await self._ex().run_in_executor(self._executor, _count)

    async def get_all_status_counts(self) -> dict[str, int]:
        return await self.get_status_counts()

    async def get_docs_by_status(
        self, status: DocStatus
    ) -> dict[str, DocProcessingStatus]:
        def _filter():
            return {
                doc_id: doc
                for doc_id, doc in self._iter_all()
                if doc.status == status
            }

        return await self._ex().run_in_executor(self._executor, _filter)

    async def get_docs_by_statuses(
        self, statuses: list[DocStatus]
    ) -> dict[str, DocProcessingStatus]:
        status_set = set(statuses)

        def _filter():
            return {
                doc_id: doc
                for doc_id, doc in self._iter_all()
                if doc.status in status_set
            }

        return await self._ex().run_in_executor(self._executor, _filter)

    async def get_docs_by_track_id(
        self, track_id: str
    ) -> dict[str, DocProcessingStatus]:
        def _filter():
            return {
                doc_id: doc
                for doc_id, doc in self._iter_all()
                if doc.track_id == track_id
            }

        return await self._ex().run_in_executor(self._executor, _filter)

    async def get_docs_paginated(
        self,
        status_filter: DocStatus | None = None,
        page: int = 1,
        page_size: int = 50,
        sort_field: str = "updated_at",
        sort_direction: str = "desc",
    ) -> tuple[list[tuple[str, DocProcessingStatus]], int]:
        def _paginate():
            all_docs = self._iter_all()
            if status_filter is not None:
                all_docs = [(i, d) for i, d in all_docs if d.status == status_filter]

            reverse = sort_direction.lower() == "desc"
            if sort_field == "id":
                all_docs.sort(key=lambda x: x[0], reverse=reverse)
            else:
                all_docs.sort(
                    key=lambda x: getattr(x[1], sort_field, ""),
                    reverse=reverse,
                )

            total = len(all_docs)
            start = (page - 1) * page_size
            return all_docs[start : start + page_size], total

        return await self._ex().run_in_executor(self._executor, _paginate)

    async def get_doc_by_file_path(self, file_path: str) -> dict[str, Any] | None:
        def _find():
            for doc_id, doc in self._iter_all():
                if doc.file_path == file_path:
                    d = _to_dict(doc)
                    d["id"] = doc_id
                    return d
            return None

        return await self._ex().run_in_executor(self._executor, _find)
