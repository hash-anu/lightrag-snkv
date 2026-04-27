"""Load LLM and embedding functions from environment variables.

Tries the LightRAG server helpers first (requires ``lightrag-hku[api]``
which reads LLM_BINDING / EMBEDDING_BINDING etc.).  Falls back to a
direct OpenAI setup when ``OPENAI_API_KEY`` is set and the api extra is
absent.
"""
from __future__ import annotations

import os


def _try_server_helpers():
    try:
        from lightrag.api.lightrag_server import _get_embed_func, _get_llm_func
        return _get_llm_func(), _get_embed_func()
    except (ImportError, Exception):
        return None, None


def get_llm_and_embed_funcs():
    """Return *(llm_func, embed_func)* configured from environment variables.

    **Server-helper path** (preferred, multi-backend):
        Install ``lightrag-hku[api]`` and set:
        ``LLM_BINDING``, ``LLM_MODEL``, ``LLM_BINDING_HOST``, ``LLM_BINDING_API_KEY``
        ``EMBEDDING_BINDING``, ``EMBEDDING_MODEL``, ``EMBEDDING_DIM``

    **OpenAI fallback** (when api extra is not installed):
        Set ``OPENAI_API_KEY``.
    """
    llm_func, embed_func = _try_server_helpers()
    if llm_func is not None:
        return llm_func, embed_func

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Cannot configure LLM/embed functions. Either:\n"
            "  1. pip install 'lightrag-hku[api]' and set LLM_BINDING/EMBEDDING_BINDING, or\n"
            "  2. Set OPENAI_API_KEY for the default OpenAI backend."
        )

    from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

    return gpt_4o_mini_complete, openai_embed
