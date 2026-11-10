"""SNKV-backed DocStatusStorage for LightRAG.

Shares ``snkv.db`` with KV and graph storage via snkv_shared.
Single column family ``doc_status``.

Rows are stored as raw JSON dicts (the same shape LightRAG's JSON backend
keeps). All access goes through the shared single-thread executor, so every
method below sees a consistent view of the column family.
"""
from __future__ import annotations

import asyncio
import heapq
import json
import os
from dataclasses import dataclass
from typing import Any, ClassVar, Iterator, Sequence, final

from lightrag.base import (
    CURSOR_END,
    CURSOR_START,
    CursorAfter,
    CursorPosition,
    DocProcessingStatus,
    DocSchedulingRecord,
    DocStatus,
    DocStatusPage,
    DocStatusStorage,
    SourceAbsent,
    SourceConflict,
    SourceResolution,
    SourceUnique,
)
from lightrag.constants import CUSTOM_CHUNK_PATCH_METADATA_KEY
from lightrag.exceptions import (
    StorageControlPlaneError,
    StorageNotInitializedError,
    StorageRecordNotFoundError,
)
from lightrag.utils import get_pinyin_sort_key, logger
from snkv import NotFoundError

from . import snkv_shared


def _status_value(raw: Any) -> Any:
    """Status as its string value, whether stored as a DocStatus or a str."""
    return raw.value if isinstance(raw, DocStatus) else raw


def _is_duplicate_row(row: Any) -> bool:
    """True for duplicate-marker rows (``metadata.is_duplicate == true``)."""
    if not isinstance(row, dict):
        return False
    metadata = row.get("metadata")
    return bool(isinstance(metadata, dict) and metadata.get("is_duplicate"))


def _row_sort_key(doc_id: str, row: dict[str, Any]) -> tuple[str, str]:
    """``(created_at, id)`` keyset key; a missing created_at sorts first."""
    created = row.get("created_at")
    return (created if isinstance(created, str) else "", doc_id)


def _encode_cursor(key: tuple[str, str]) -> str:
    return json.dumps(list(key), ensure_ascii=False)


def _decode_cursor(opaque: str) -> tuple[str, str]:
    try:
        created, doc_id = json.loads(opaque)
        if not isinstance(created, str) or not isinstance(doc_id, str):
            raise ValueError("cursor fields must be strings")
    except (ValueError, TypeError) as e:
        raise StorageControlPlaneError(
            f"Malformed scheduling cursor for SNKVDocStatusStorage: {e}"
        ) from e
    return (created, doc_id)


def _to_status(row: dict[str, Any]) -> DocProcessingStatus:
    """Raw row -> DocProcessingStatus, with the same defaults as LightRAG's
    JSON backend. Raises KeyError/TypeError/ValueError on a malformed row."""
    data = dict(row)
    data.pop("content", None)  # deprecated inline content
    if not data.get("file_path"):
        data["file_path"] = "no-file-path"
    data.setdefault("metadata", {})
    data.setdefault("error_msg", None)
    data["status"] = DocStatus(_status_value(data["status"]))
    return DocProcessingStatus.from_stored(data)


@final
@dataclass
class SNKVDocStatusStorage(DocStatusStorage):
    # get_by_id is a direct key lookup: a miss is a confirmed absence.
    supports_strict_point_reads: ClassVar[bool] = True

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
        self._db_path = os.path.join(db_dir, "snkv.db")
        self._shared: snkv_shared.SharedStore | None = None
        self._cf = None

    async def _run(self, fn):
        if self._shared is None or self._cf is None:
            raise StorageNotInitializedError("SNKVDocStatusStorage")
        return await asyncio.get_running_loop().run_in_executor(self._shared.executor, fn)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        self._shared = snkv_shared.acquire(self._db_path)

        def _open():
            try:
                self._cf = self._shared.kv.open_column_family("doc_status")
            except NotFoundError:
                self._cf = self._shared.kv.create_column_family("doc_status")

        await asyncio.get_running_loop().run_in_executor(self._shared.executor, _open)

    async def finalize(self) -> None:
        if self._shared is None:
            return

        def _close():
            if self._cf is not None:
                try:
                    self._cf.close()
                except Exception:
                    pass
                self._cf = None

        await asyncio.get_running_loop().run_in_executor(self._shared.executor, _close)
        snkv_shared.release(self._db_path)
        self._shared = None

    # ------------------------------------------------------------------
    # Internal helpers (run inside the executor)
    # ------------------------------------------------------------------

    def _get_row(self, doc_id: str) -> dict[str, Any] | None:
        raw = self._cf.get(doc_id.encode())
        return json.loads(raw.decode()) if raw is not None else None

    def _put_row(self, doc_id: str, row: dict[str, Any]) -> None:
        self._cf.put(doc_id.encode(), json.dumps(row, ensure_ascii=False).encode())

    def _scan(self, *, strict: bool = False) -> Iterator[tuple[str, dict[str, Any]]]:
        """Yield every (doc_id, row). Undecodable rows raise under strict,
        otherwise they are logged and skipped."""
        with self._cf.iterator() as it:
            for key_b, val_b in it:
                doc_id = key_b.decode()
                try:
                    row = json.loads(val_b.decode())
                    if not isinstance(row, dict):
                        raise TypeError("doc_status record is not a mapping")
                except (ValueError, TypeError) as e:
                    logger.error(f"[{self.workspace}] Unreadable doc_status row {doc_id}: {e}")
                    if strict:
                        raise
                    continue
                yield doc_id, row

    def _scheduling_record(
        self, doc_id: str, row: dict[str, Any], *, strict: bool
    ) -> DocSchedulingRecord | None:
        """Lightweight projection; strict raises on unusable rows, relaxed
        returns None."""
        try:
            created_at = row["created_at"]
            updated_at = row.get("updated_at", created_at)
            if not isinstance(created_at, str) or not isinstance(updated_at, str):
                raise TypeError("created_at/updated_at must be strings")
            metadata = row.get("metadata")
            return DocSchedulingRecord(
                id=doc_id,
                status=DocStatus(_status_value(row["status"])),
                created_at=created_at,
                updated_at=updated_at,
                file_path=row.get("file_path") or "no-file-path",
                track_id=row.get("track_id"),
                has_custom_chunk_journal=isinstance(metadata, dict)
                and isinstance(metadata.get(CUSTOM_CHUNK_PATCH_METADATA_KEY), dict),
            )
        except (KeyError, TypeError, ValueError) as e:
            logger.error(f"[{self.workspace}] Unusable scheduling row {doc_id}: {e}")
            if strict:
                raise
            return None

    # ------------------------------------------------------------------
    # BaseKVStorage methods
    # ------------------------------------------------------------------

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        return await self._run(lambda: self._get_row(id))

    async def get_by_id_strict(self, id: str) -> dict[str, Any] | None:
        return await self._run(lambda: self._get_row(id))

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any] | None]:
        return await self._run(lambda: [self._get_row(i) for i in ids])

    async def filter_keys(self, keys: set[str]) -> set[str]:
        return await self._run(lambda: {k for k in keys if not self._cf.exists(k.encode())})

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if not data:
            return

        def _upsert():
            self._shared.kv.begin(write=True)
            try:
                for key, val in data.items():
                    self._put_row(key, val)
                self._shared.kv.commit()
            except Exception:
                self._shared.kv.rollback()
                raise

        await self._run(_upsert)

    async def delete(self, ids: list[str]) -> None:
        if not ids:
            return

        def _delete():
            self._shared.kv.begin(write=True)
            try:
                for doc_id in ids:
                    try:
                        self._cf.delete(doc_id.encode())
                    except NotFoundError:
                        pass
                self._shared.kv.commit()
            except Exception:
                self._shared.kv.rollback()
                raise

        await self._run(_delete)

    async def is_empty(self) -> bool:
        return await self._run(lambda: self._cf.count() == 0)

    async def index_done_callback(self) -> None:
        await self._run(lambda: self._shared.kv.sync())

    async def drop(self) -> dict[str, str]:
        try:
            await self._run(lambda: self._cf.clear())
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(f"[{self.workspace}] Error dropping doc_status: {e}")
            return {"status": "error", "message": str(e)}

    # ------------------------------------------------------------------
    # Status listing / counting
    # ------------------------------------------------------------------

    async def get_status_counts(self) -> dict[str, int]:
        def _count():
            counts = {s.value: 0 for s in DocStatus}
            for _, row in self._scan():
                status = _status_value(row.get("status"))
                if status is not None:
                    counts[status] = counts.get(status, 0) + 1
            return counts

        return await self._run(_count)

    async def get_all_status_counts(self) -> dict[str, int]:
        counts = await self.get_status_counts()
        counts["all"] = sum(counts.values())
        return counts

    async def count_docs_by_statuses(
        self, statuses: list[DocStatus], *, strict: bool = True
    ) -> int:
        if not statuses:
            return 0
        wanted = {s.value for s in statuses}

        def _count():
            n = 0
            for doc_id, row in self._scan(strict=strict):
                if "status" not in row:
                    if strict:
                        raise TypeError(f"doc_status record {doc_id} has no readable status")
                    continue
                if _status_value(row["status"]) in wanted:
                    n += 1
            return n

        return await self._run(_count)

    def _filter_to_status(
        self, predicate, *, strict: bool
    ) -> dict[str, DocProcessingStatus]:
        out: dict[str, DocProcessingStatus] = {}
        for doc_id, row in self._scan(strict=strict):
            try:
                if not predicate(row):
                    continue
                out[doc_id] = _to_status(row)
            except (KeyError, TypeError, ValueError) as e:
                logger.error(f"[{self.workspace}] Unusable doc_status row {doc_id}: {e}")
                if strict:
                    raise
        return out

    async def get_docs_by_status(self, status: DocStatus) -> dict[str, DocProcessingStatus]:
        return await self.get_docs_by_statuses([status])

    async def get_docs_by_statuses(
        self, statuses: list[DocStatus], strict: bool = False
    ) -> dict[str, DocProcessingStatus]:
        if not statuses:
            return {}
        wanted = {s.value for s in statuses}
        return await self._run(
            lambda: self._filter_to_status(
                lambda r: _status_value(r["status"]) in wanted, strict=strict
            )
        )

    async def get_docs_by_track_id(self, track_id: str) -> dict[str, DocProcessingStatus]:
        return await self._run(
            lambda: self._filter_to_status(
                lambda r: r.get("track_id") == track_id, strict=False
            )
        )

    async def get_docs_paginated(
        self,
        status_filter: DocStatus | None = None,
        status_filters: list[DocStatus] | None = None,
        page: int = 1,
        page_size: int = 50,
        sort_field: str = "updated_at",
        sort_direction: str = "desc",
    ) -> tuple[list[tuple[str, DocProcessingStatus]], int]:
        wanted = self.resolve_status_filter_values(
            status_filter=status_filter, status_filters=status_filters
        )
        page = max(page, 1)
        page_size = min(max(page_size, 10), 200)
        if sort_field not in ("created_at", "updated_at", "id", "file_path"):
            sort_field = "updated_at"
        reverse = sort_direction.lower() != "asc"

        def _paginate():
            items: list[tuple[Any, str, DocProcessingStatus]] = []
            for doc_id, row in self._scan():
                if wanted is not None and _status_value(row.get("status")) not in wanted:
                    continue
                try:
                    doc = _to_status({k: v for k, v in row.items() if k != "chunks_list"})
                except (KeyError, TypeError, ValueError) as e:
                    logger.error(f"[{self.workspace}] Unusable doc_status row {doc_id}: {e}")
                    continue
                if sort_field == "id":
                    key = doc_id
                elif sort_field == "file_path":
                    key = get_pinyin_sort_key(doc.file_path)
                else:
                    key = getattr(doc, sort_field, "") or ""
                items.append((key, doc_id, doc))
            items.sort(key=lambda t: t[0], reverse=reverse)
            start = (page - 1) * page_size
            return [(i, d) for _, i, d in items[start : start + page_size]], len(items)

        return await self._run(_paginate)

    # ------------------------------------------------------------------
    # Lookups used for dedup and source resolution
    # ------------------------------------------------------------------

    async def get_doc_by_file_path(self, file_path: str) -> dict[str, Any] | None:
        def _find():
            for _, row in self._scan():
                if row.get("file_path") == file_path:
                    return row
            return None

        return await self._run(_find)

    async def get_doc_by_file_basename(
        self, basename: str
    ) -> tuple[str, dict[str, Any]] | None:
        """The PRIMARY (non-duplicate) row whose canonical file_path matches."""
        if not basename or basename == "unknown_source":
            return None

        def _find():
            for doc_id, row in self._scan():
                if row.get("file_path") == basename and not _is_duplicate_row(row):
                    return doc_id, row
            return None

        return await self._run(_find)

    async def get_doc_by_content_hash(
        self, content_hash: str, *, exclude_doc_id: str | None = None
    ) -> tuple[str, dict[str, Any]] | None:
        """Earliest row by (created_at, id) with this content_hash, skipping
        ``exclude_doc_id`` and any duplicate row that points at it."""
        if not content_hash:
            return None

        def _find():
            best: tuple[tuple[str, str], str, dict[str, Any]] | None = None
            for doc_id, row in self._scan(strict=True):
                if doc_id == exclude_doc_id:
                    continue
                if self._row_points_at_as_duplicate(row, exclude_doc_id):
                    continue
                if row.get("content_hash") != content_hash:
                    continue
                key = _row_sort_key(doc_id, row)
                if best is None or key < best[0]:
                    best = (key, doc_id, row)
            return None if best is None else (best[1], best[2])

        return await self._run(_find)

    async def resolve_doc_source_strict(self, canonical_source_key: str) -> SourceResolution:
        """0 / 1 / 2+ primary rows for a source -> Absent / Unique / Conflict."""
        if not canonical_source_key or canonical_source_key == "unknown_source":
            await self._run(lambda: None)  # still fail closed if uninitialized
            return SourceAbsent()

        def _find():
            found: list[tuple[str, dict[str, Any]]] = []
            for doc_id, row in self._scan(strict=True):
                if row.get("file_path") == canonical_source_key and not _is_duplicate_row(row):
                    found.append((doc_id, row))
                    if len(found) >= 2:
                        break
            return found

        candidates = await self._run(_find)
        if not candidates:
            return SourceAbsent()
        if len(candidates) == 1:
            doc_id, row = candidates[0]
            return SourceUnique(doc_id=doc_id, doc=self._scheduling_record_from_raw(doc_id, row))
        return SourceConflict(
            candidate_count=None,
            sample_doc_ids=tuple(sorted(doc_id for doc_id, _ in candidates)),
        )

    # ------------------------------------------------------------------
    # Scheduling API (paged sweep, batch reads, targeted updates)
    # ------------------------------------------------------------------

    async def get_docs_by_statuses_page(
        self,
        statuses: list[DocStatus],
        *,
        limit: int,
        position: CursorPosition = CURSOR_START,
        strict: bool = False,
    ) -> DocStatusPage:
        """One keyset page ordered by (created_at, id). Page memory is bounded
        by ``limit`` via a heap; every selected row counts as consumed."""
        if limit <= 0:
            raise ValueError(f"page limit must be positive, got {limit}")
        if not statuses or position is CURSOR_END:
            await self._run(lambda: None)  # still fail closed if uninitialized
            return DocStatusPage(docs={}, next_position=CURSOR_END)
        last_key = _decode_cursor(position.opaque) if isinstance(position, CursorAfter) else None
        wanted = {s.value for s in statuses}

        def _page():
            def _candidates():
                for doc_id, row in self._scan(strict=strict):
                    if _status_value(row.get("status")) not in wanted:
                        continue
                    key = _row_sort_key(doc_id, row)
                    if last_key is not None and key <= last_key:
                        continue
                    yield key, doc_id, row

            selected = heapq.nsmallest(limit, _candidates(), key=lambda t: t[0])
            docs: dict[str, DocSchedulingRecord] = {}
            for _, doc_id, row in selected:
                record = self._scheduling_record(doc_id, row, strict=strict)
                if record is not None:
                    docs[doc_id] = record
            return selected, docs

        selected, docs = await self._run(_page)
        if len(selected) < limit:
            next_position: CursorPosition = CURSOR_END
        else:
            next_position = CursorAfter(_encode_cursor(selected[-1][0]))
        return DocStatusPage(docs=docs, next_position=next_position)

    async def get_docs_by_ids(
        self, doc_ids: Sequence[str], *, strict: bool = False
    ) -> dict[str, DocSchedulingRecord]:
        def _read():
            out: dict[str, DocSchedulingRecord] = {}
            for doc_id in doc_ids:
                row = self._get_row(doc_id)
                if row is None:
                    continue  # confirmed absent
                record = self._scheduling_record(doc_id, row, strict=strict)
                if record is not None:
                    out[doc_id] = record
            return out

        return await self._run(_read)

    async def get_full_docs_by_ids(
        self, doc_ids: Sequence[str], *, strict: bool = False
    ) -> dict[str, DocProcessingStatus]:
        def _read():
            out: dict[str, DocProcessingStatus] = {}
            for doc_id in doc_ids:
                row = self._get_row(doc_id)
                if row is None:
                    continue  # confirmed absent
                try:
                    out[doc_id] = _to_status(row)
                except (KeyError, TypeError, ValueError) as e:
                    logger.error(f"[{self.workspace}] Unusable doc_status row {doc_id}: {e}")
                    if strict:
                        raise
            return out

        return await self._run(_read)

    async def update_doc_status_fields(
        self, doc_id: str, fields: dict[str, Any], *, missing_ok: bool = False
    ) -> None:
        """Merge ``fields`` into one row in a single transaction."""
        if "created_at" in fields:
            raise ValueError(
                "created_at is an immutable scheduling sort key and cannot "
                "be changed via update_doc_status_fields"
            )

        def _update():
            self._shared.kv.begin(write=True)
            try:
                row = self._get_row(doc_id)
                if row is None:
                    self._shared.kv.rollback()
                    if missing_ok:
                        return
                    raise StorageRecordNotFoundError(doc_id)
                self._put_row(doc_id, {**row, **fields})
                self._shared.kv.commit()
            except StorageRecordNotFoundError:
                raise
            except Exception:
                self._shared.kv.rollback()
                raise

        await self._run(_update)
