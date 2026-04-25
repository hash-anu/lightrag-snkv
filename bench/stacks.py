"""Factory functions that return a configured, uninitialised LightRAG instance
for each storage stack.

Each function accepts ``working_dir`` and ``llm_func``/``embed_func`` and
returns a LightRAG instance.  Call ``await rag.initialize_storages()`` after.
"""
from __future__ import annotations

from typing import Any, Callable


def _base_rag(working_dir: str, llm_func: Any, embed_func: Any, **kwargs):
    from lightrag import LightRAG

    return LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_func,
        embedding_func=embed_func,
        **kwargs,
    )


def snkv_stack(working_dir: str, llm_func: Any, embed_func: Any):
    """All four SNKV backends."""
    from lightrag_snkv.register import register

    register()
    return _base_rag(
        working_dir,
        llm_func,
        embed_func,
        kv_storage="SNKVKVStorage",
        vector_storage="SNKVVectorStorage",
        graph_storage="SNKVGraphStorage",
        doc_status_storage="SNKVDocStatusStorage",
    )


def nano_stack(working_dir: str, llm_func: Any, embed_func: Any):
    """Default LightRAG stack: JSON KV + NanoVectorDB + NetworkX."""
    return _base_rag(
        working_dir,
        llm_func,
        embed_func,
        kv_storage="JsonKVStorage",
        vector_storage="NanoVectorDBStorage",
        graph_storage="NetworkXStorage",
        doc_status_storage="JsonDocStatusStorage",
    )


def faiss_stack(working_dir: str, llm_func: Any, embed_func: Any):
    """JSON KV + FAISS vector + NetworkX."""
    return _base_rag(
        working_dir,
        llm_func,
        embed_func,
        kv_storage="JsonKVStorage",
        vector_storage="FaissVectorDBStorage",
        graph_storage="NetworkXStorage",
        doc_status_storage="JsonDocStatusStorage",
    )


STACK_FACTORIES: dict[str, Callable] = {
    "snkv": snkv_stack,
    "nano": nano_stack,
    "faiss": faiss_stack,
}
