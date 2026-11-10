"""Unit tests for SNKVDocStatusStorage."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from lightrag.base import DocProcessingStatus, DocStatus
from lightrag_snkv.snkv_doc_status_impl import SNKVDocStatusStorage


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_doc(status: DocStatus = DocStatus.PROCESSED, file_path: str = "test.txt",
             track_id: str | None = None) -> dict:
    return {
        "content_summary": "Test document content",
        "content_length": 100,
        "file_path": file_path,
        "status": status.value,
        "created_at": _now(),
        "updated_at": _now(),
        "track_id": track_id,
        "chunks_count": 5,
        "chunks_list": ["c1", "c2"],
        "error_msg": None,
        "metadata": {},
        "multimodal_processed": None,
    }


async def make_storage(global_config):
    ef = AsyncMock()
    s = SNKVDocStatusStorage(
        namespace="doc_status",
        workspace="",
        global_config=global_config,
        embedding_func=ef,
    )
    await s.initialize()
    return s


@pytest.mark.asyncio
async def test_upsert_and_get_by_id(global_config):
    s = await make_storage(global_config)
    await s.upsert({"doc1": make_doc()})
    result = await s.get_by_id("doc1")
    assert result is not None
    assert result["status"] == DocStatus.PROCESSED.value
    await s.finalize()


@pytest.mark.asyncio
async def test_get_status_counts(global_config):
    s = await make_storage(global_config)
    await s.upsert({
        "d1": make_doc(DocStatus.PROCESSED),
        "d2": make_doc(DocStatus.PROCESSED),
        "d3": make_doc(DocStatus.FAILED),
    })
    counts = await s.get_status_counts()
    assert counts.get("processed", 0) == 2
    assert counts.get("failed", 0) == 1
    await s.finalize()


@pytest.mark.asyncio
async def test_get_docs_by_status(global_config):
    s = await make_storage(global_config)
    await s.upsert({
        "d1": make_doc(DocStatus.PROCESSED),
        "d2": make_doc(DocStatus.PENDING),
        "d3": make_doc(DocStatus.PROCESSED),
    })
    processed = await s.get_docs_by_status(DocStatus.PROCESSED)
    assert len(processed) == 2
    assert all(d.status == DocStatus.PROCESSED for d in processed.values())
    await s.finalize()


@pytest.mark.asyncio
async def test_get_docs_by_statuses(global_config):
    s = await make_storage(global_config)
    await s.upsert({
        "d1": make_doc(DocStatus.PROCESSED),
        "d2": make_doc(DocStatus.FAILED),
        "d3": make_doc(DocStatus.PENDING),
    })
    results = await s.get_docs_by_statuses([DocStatus.PROCESSED, DocStatus.FAILED])
    assert len(results) == 2
    await s.finalize()


@pytest.mark.asyncio
async def test_get_docs_by_track_id(global_config):
    s = await make_storage(global_config)
    await s.upsert({
        "d1": make_doc(track_id="batch-1"),
        "d2": make_doc(track_id="batch-1"),
        "d3": make_doc(track_id="batch-2"),
    })
    result = await s.get_docs_by_track_id("batch-1")
    assert len(result) == 2
    await s.finalize()


@pytest.mark.asyncio
async def test_get_docs_paginated(global_config):
    s = await make_storage(global_config)
    docs = {f"doc{i:03d}": make_doc() for i in range(25)}
    await s.upsert(docs)
    page1, total = await s.get_docs_paginated(page=1, page_size=10)
    assert total == 25
    assert len(page1) == 10
    page3, _ = await s.get_docs_paginated(page=3, page_size=10)
    assert len(page3) == 5
    await s.finalize()


@pytest.mark.asyncio
async def test_get_doc_by_file_path(global_config):
    s = await make_storage(global_config)
    await s.upsert({
        "d1": make_doc(file_path="/path/to/report.pdf"),
        "d2": make_doc(file_path="/other/doc.txt"),
    })
    result = await s.get_doc_by_file_path("/path/to/report.pdf")
    assert result is not None
    assert result["file_path"] == "/path/to/report.pdf"
    none_result = await s.get_doc_by_file_path("/path/to/nonexistent.txt")
    assert none_result is None
    await s.finalize()


# ---------------------------------------------------------------------------
# LightRAG 1.5 scheduling / dedup API
# ---------------------------------------------------------------------------

def _doc_at(created_at: str, status: DocStatus = DocStatus.PENDING, **extra) -> dict:
    d = make_doc(status=status, file_path=extra.pop("file_path", "test.txt"))
    d["created_at"] = created_at
    d["updated_at"] = created_at
    d.update(extra)
    return d


@pytest.mark.asyncio
async def test_get_docs_by_statuses_page_sweeps_in_order(global_config):
    from lightrag.base import CURSOR_END, CURSOR_START

    s = await make_storage(global_config)
    await s.upsert({
        f"d{i}": _doc_at(f"2026-01-0{i}T00:00:00", file_path=f"f{i}.txt")
        for i in range(1, 6)
    })
    await s.upsert({"done": _doc_at("2026-01-01T00:00:00", status=DocStatus.PROCESSED)})

    seen, position = [], CURSOR_START
    while True:
        page = await s.get_docs_by_statuses_page(
            [DocStatus.PENDING], limit=2, position=position, strict=True
        )
        seen.extend(page.docs)
        if page.next_position is CURSOR_END:
            break
        position = page.next_position
    assert seen == ["d1", "d2", "d3", "d4", "d5"]
    await s.finalize()


@pytest.mark.asyncio
async def test_get_docs_by_ids_and_full_docs(global_config):
    s = await make_storage(global_config)
    await s.upsert({"a": make_doc(), "b": make_doc(status=DocStatus.FAILED)})

    light = await s.get_docs_by_ids(["a", "b", "missing"], strict=True)
    assert set(light) == {"a", "b"}
    assert light["b"].status == DocStatus.FAILED

    full = await s.get_full_docs_by_ids(["a", "missing"], strict=True)
    assert set(full) == {"a"}
    assert full["a"].chunks_list == ["c1", "c2"]
    await s.finalize()


@pytest.mark.asyncio
async def test_get_doc_by_content_hash_earliest_and_exclude(global_config):
    s = await make_storage(global_config)
    await s.upsert({
        "late": _doc_at("2026-01-03T00:00:00", content_hash="h1"),
        "early": _doc_at("2026-01-01T00:00:00", content_hash="h1"),
        "pointer": _doc_at(
            "2026-01-00T00:00:00", content_hash="h1",
            metadata={"is_duplicate": True, "original_doc_id": "early"},
        ),
    })
    assert (await s.get_doc_by_content_hash("h1"))[0] == "pointer"
    # Excluding "early" also skips the row that points at it.
    assert (await s.get_doc_by_content_hash("h1", exclude_doc_id="early"))[0] == "late"
    assert await s.get_doc_by_content_hash("nope") is None
    await s.finalize()


@pytest.mark.asyncio
async def test_file_basename_and_source_resolution(global_config):
    from lightrag.base import SourceAbsent, SourceConflict, SourceUnique

    s = await make_storage(global_config)
    await s.upsert({
        "one": make_doc(file_path="one.pdf"),
        "dup": make_doc(file_path="one.pdf") | {"metadata": {"is_duplicate": True}},
        "two_a": make_doc(file_path="two.pdf"),
        "two_b": make_doc(file_path="two.pdf"),
    })
    assert (await s.get_doc_by_file_basename("one.pdf"))[0] == "one"
    assert await s.get_doc_by_file_basename("none.pdf") is None

    assert isinstance(await s.resolve_doc_source_strict("none.pdf"), SourceAbsent)
    unique = await s.resolve_doc_source_strict("one.pdf")
    assert isinstance(unique, SourceUnique) and unique.doc_id == "one"
    conflict = await s.resolve_doc_source_strict("two.pdf")
    assert isinstance(conflict, SourceConflict)
    assert conflict.sample_doc_ids == ("two_a", "two_b")
    await s.finalize()


@pytest.mark.asyncio
async def test_update_fields_and_count(global_config):
    from lightrag.exceptions import StorageRecordNotFoundError

    s = await make_storage(global_config)
    await s.upsert({"a": make_doc(status=DocStatus.PENDING)})
    await s.update_doc_status_fields("a", {"status": DocStatus.PROCESSING.value})
    assert (await s.get_by_id("a"))["status"] == DocStatus.PROCESSING.value
    assert await s.count_docs_by_statuses([DocStatus.PROCESSING]) == 1

    with pytest.raises(ValueError):
        await s.update_doc_status_fields("a", {"created_at": "x"})
    with pytest.raises(StorageRecordNotFoundError):
        await s.update_doc_status_fields("missing", {"status": "failed"})
    await s.update_doc_status_fields("missing", {"status": "failed"}, missing_ok=True)
    await s.finalize()
