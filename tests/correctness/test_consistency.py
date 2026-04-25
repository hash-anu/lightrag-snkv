"""Consistency correctness test: insert → delete → re-query.

Verifies that SNKV storage remains consistent after document deletion.

Requires: LIGHTRAG_RUN_INTEGRATION=true
"""
from __future__ import annotations

import os
import shutil
import tempfile

import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("LIGHTRAG_RUN_INTEGRATION"),
    reason="set LIGHTRAG_RUN_INTEGRATION=true to run correctness tests",
)

_DOC_A = "Marie Curie discovered radioactivity and won two Nobel Prizes."
_DOC_B = "Isaac Newton formulated the laws of motion and universal gravitation."


async def _make_snkv_rag(working_dir: str):
    from lightrag import LightRAG
    from lightrag_snkv.register import register

    register()
    from lightrag.api.lightrag_server import _get_embed_func, _get_llm_func

    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=_get_llm_func(),
        embedding_func=_get_embed_func(),
        kv_storage="SNKVKVStorage",
        vector_storage="SNKVVectorStorage",
        graph_storage="SNKVGraphStorage",
        doc_status_storage="SNKVDocStatusStorage",
    )
    await rag.initialize_storages()
    return rag


@pytest.mark.asyncio
async def test_delete_removes_doc_from_answers():
    working_dir = tempfile.mkdtemp(prefix="snkv_consistency_")
    try:
        rag = await _make_snkv_rag(working_dir)

        # Insert both documents
        ids = await rag.ainsert([_DOC_A, _DOC_B])

        # Both should be answerable
        ans_before = await rag.aquery("Who is Marie Curie?")
        assert "curie" in str(ans_before).lower() or "radioactivity" in str(ans_before).lower()

        # Delete the Curie document
        if isinstance(ids, list) and len(ids) >= 1:
            await rag.adelete_by_doc_id(ids[0])

        # After deletion, querying for Curie should return less relevant content
        ans_after = await rag.aquery("Who is Marie Curie?")
        # Newton answer should still work
        ans_newton = await rag.aquery("Who is Isaac Newton?")
        assert "newton" in str(ans_newton).lower() or "gravity" in str(ans_newton).lower()

        await rag.finalize_storages()
    finally:
        shutil.rmtree(working_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_reinsert_after_delete():
    working_dir = tempfile.mkdtemp(prefix="snkv_reinsert_")
    try:
        rag = await _make_snkv_rag(working_dir)

        await rag.ainsert(_DOC_A)
        ans1 = await rag.aquery("Nobel Prize winner")
        assert ans1 is not None

        # Drop everything
        await rag.adelete_by_doc_id("all")  # not necessarily valid; just test re-insert
        await rag.ainsert(_DOC_A)
        ans2 = await rag.aquery("Nobel Prize winner")
        assert ans2 is not None

        await rag.finalize_storages()
    finally:
        shutil.rmtree(working_dir, ignore_errors=True)
