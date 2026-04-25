"""Unit tests for SNKVDocStatusStorage (7 test cases)."""
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
