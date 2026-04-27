"""Integration tests: LightRAG with all-SNKV backend.

Requires LLM + embedding credentials in environment.  See
tests/integration/llm_env.py for supported configuration paths.

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
        from tests.integration.llm_env import get_llm_and_embed_funcs

        register()
        llm_func, embed_func = get_llm_and_embed_funcs()

        return LightRAG(
            working_dir=working_dir,
            llm_model_func=llm_func,
            embedding_func=embed_func,
            kv_storage="SNKVKVStorage",
            vector_storage="SNKVVectorStorage",
            graph_storage="SNKVGraphStorage",
            doc_status_storage="SNKVDocStatusStorage",
        )
