"""Answer-quality correctness test.

Inserts a fixed corpus, then asserts that query answers from SNKV and
NanoVectorDB backends have ≥80% token overlap — proving SNKV does not
degrade retrieval quality.

Requires: LIGHTRAG_RUN_INTEGRATION=true
"""
from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from collections import Counter

import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("LIGHTRAG_RUN_INTEGRATION"),
    reason="set LIGHTRAG_RUN_INTEGRATION=true to run correctness tests",
)

_CORPUS = """
Marie Curie was a pioneering physicist who discovered polonium and radium.
She was the first woman to win a Nobel Prize.
Pierre Curie was her husband and research partner.
They worked at the University of Paris.
Curie won the Nobel Prize in Physics in 1903 and Chemistry in 1911.
Radioactivity was the central theme of her research.
"""

_QUERIES = [
    "Who discovered polonium?",
    "What Nobel Prizes did Marie Curie win?",
    "Where did the Curies work?",
]


def _token_overlap(a: str, b: str) -> float:
    a_tokens = Counter(a.lower().split())
    b_tokens = Counter(b.lower().split())
    shared = sum((a_tokens & b_tokens).values())
    total = max(sum(a_tokens.values()), sum(b_tokens.values()), 1)
    return shared / total


async def _make_nano_rag(working_dir: str):
    from lightrag import LightRAG
    from lightrag.api.lightrag_server import _get_embed_func, _get_llm_func

    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=_get_llm_func(),
        embedding_func=_get_embed_func(),
        kv_storage="JsonKVStorage",
        vector_storage="NanoVectorDBStorage",
        graph_storage="NetworkXStorage",
        doc_status_storage="JsonDocStatusStorage",
    )
    await rag.initialize_storages()
    return rag


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
async def test_snkv_answer_quality_vs_nano():
    nano_dir = tempfile.mkdtemp(prefix="nano_quality_")
    snkv_dir = tempfile.mkdtemp(prefix="snkv_quality_")
    try:
        nano_rag = await _make_nano_rag(nano_dir)
        snkv_rag = await _make_snkv_rag(snkv_dir)

        await nano_rag.ainsert(_CORPUS)
        await snkv_rag.ainsert(_CORPUS)

        for query in _QUERIES:
            nano_ans = await nano_rag.aquery(query)
            snkv_ans = await snkv_rag.aquery(query)
            overlap = _token_overlap(str(nano_ans), str(snkv_ans))
            assert overlap >= 0.80, (
                f"Query '{query}': token overlap {overlap:.2%} < 80%\n"
                f"  Nano: {nano_ans}\n  SNKV: {snkv_ans}"
            )

        await nano_rag.finalize_storages()
        await snkv_rag.finalize_storages()
    finally:
        shutil.rmtree(nano_dir, ignore_errors=True)
        shutil.rmtree(snkv_dir, ignore_errors=True)
