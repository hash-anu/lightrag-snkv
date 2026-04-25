"""Integration tests: LightRAG with FAISS vector backend.

Run with: LIGHTRAG_RUN_INTEGRATION=true pytest tests/integration/test_lightrag_faiss.py
"""
from __future__ import annotations

import os

import pytest

from tests.integration.base_lightrag_test import BaseLightRAGTest

pytestmark = pytest.mark.skipif(
    not os.getenv("LIGHTRAG_RUN_INTEGRATION"),
    reason="set LIGHTRAG_RUN_INTEGRATION=true to run integration tests",
)


class TestLightRAGFaiss(BaseLightRAGTest):
    async def _make_rag(self, working_dir: str):
        from lightrag import LightRAG
        from lightrag.api.lightrag_server import _get_embed_func, _get_llm_func

        rag = LightRAG(
            working_dir=working_dir,
            llm_model_func=_get_llm_func(),
            embedding_func=_get_embed_func(),
            kv_storage="JsonKVStorage",
            vector_storage="FaissVectorDBStorage",
            graph_storage="NetworkXStorage",
            doc_status_storage="JsonDocStatusStorage",
        )
        return rag
