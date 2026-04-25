"""Integration tests: LightRAG with all-SNKV backend.

Run with a real LLM+embedding configured via env vars:
    LIGHTRAG_RUN_INTEGRATION=true pytest tests/integration/test_lightrag_snkv.py
"""
from __future__ import annotations

import os

import pytest

from tests.integration.base_lightrag_test import BaseLightRAGTest

pytestmark = pytest.mark.skipif(
    not os.getenv("LIGHTRAG_RUN_INTEGRATION"),
    reason="set LIGHTRAG_RUN_INTEGRATION=true to run integration tests",
)


class TestLightRAGSNKV(BaseLightRAGTest):
    async def _make_rag(self, working_dir: str):
        from lightrag import LightRAG
        from lightrag_snkv.register import register

        register()

        # Import LLM/embed from env config (same approach as the server)
        from lightrag.api.lightrag_server import _get_llm_func, _get_embed_func

        rag = LightRAG(
            working_dir=working_dir,
            llm_model_func=_get_llm_func(),
            embedding_func=_get_embed_func(),
            kv_storage="SNKVKVStorage",
            vector_storage="SNKVVectorStorage",
            graph_storage="SNKVGraphStorage",
            doc_status_storage="SNKVDocStatusStorage",
        )
        return rag
