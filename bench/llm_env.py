"""Resolve LLM + embedding functions from environment variables.

Builds functions directly — never imports lightrag.api.lightrag_server,
which calls argparse at module import time and crashes under pytest/bench argv.

Supported bindings
------------------
LLM_BINDING      : openai (default), openai-ollama, azure_openai, ollama
EMBEDDING_BINDING: openai (default), azure_openai, ollama

Environment variables
---------------------
LLM_BINDING              backend for LLM (default: openai)
LLM_MODEL                model name (default: gpt-4o-mini)
LLM_BINDING_HOST         optional base URL override
LLM_BINDING_API_KEY      API key (falls back to OPENAI_API_KEY)
EMBEDDING_BINDING        backend for embeddings (default: openai)
EMBEDDING_MODEL          model name (default: text-embedding-3-small)
EMBEDDING_DIM            vector dimension (default: 1536)
EMBEDDING_BINDING_HOST   optional base URL override
EMBEDDING_BINDING_API_KEY API key (falls back to OPENAI_API_KEY)
OPENAI_API_KEY           standard OpenAI key — fallback for both sides

Azure-specific variables (when LLM_BINDING=azure_openai)
---------------------------------------------------------
AZURE_OPENAI_API_VERSION   API version (default: 2024-08-01-preview)
AZURE_OPENAI_DEPLOYMENT    deployment name override (falls back to LLM_MODEL)
"""
from __future__ import annotations

import os


def get_llm_and_embed_funcs():
    """Return *(llm_func, embed_func)* built from environment variables.

    Raises ``MissingCredentialsError`` (a subclass of RuntimeError) when the
    selected binding requires an API key that is not set.  Callers that want
    to skip gracefully (e.g. pytest fixtures) can catch that specific type.
    """
    llm_binding   = os.environ.get("LLM_BINDING",        "openai").lower()
    embed_binding = os.environ.get("EMBEDDING_BINDING",  "openai").lower()
    llm_model     = os.environ.get("LLM_MODEL",          "gpt-4o-mini")
    embed_model   = os.environ.get("EMBEDDING_MODEL",    "text-embedding-3-small")
    embed_dim     = int(os.environ.get("EMBEDDING_DIM",  "1536"))
    llm_host      = os.environ.get("LLM_BINDING_HOST")          or None
    llm_api_key   = (os.environ.get("LLM_BINDING_API_KEY")      or
                     os.environ.get("OPENAI_API_KEY"))           or None
    embed_host    = os.environ.get("EMBEDDING_BINDING_HOST")     or None
    embed_api_key = (os.environ.get("EMBEDDING_BINDING_API_KEY") or
                     os.environ.get("OPENAI_API_KEY"))           or None

    # Fail fast — better than a cryptic 401 deep in the call stack.
    if llm_binding in ("openai", "openai-ollama", "azure_openai") and not llm_api_key and not llm_host:
        raise MissingCredentialsError(
            f"{llm_binding} LLM needs OPENAI_API_KEY or LLM_BINDING_API_KEY."
        )
    if embed_binding in ("openai", "azure_openai") and not embed_api_key and not embed_host:
        raise MissingCredentialsError(
            f"{embed_binding} embedding needs OPENAI_API_KEY or EMBEDDING_BINDING_API_KEY."
        )

    llm_func   = _make_llm(llm_binding, llm_model, llm_host, llm_api_key)
    embed_func = _make_embed(embed_binding, embed_model, embed_dim, embed_host, embed_api_key)
    return llm_func, embed_func


class MissingCredentialsError(RuntimeError):
    """Raised when required API credentials are absent from the environment."""


def _make_llm(binding: str, model: str, host: str | None, api_key: str | None):
    if binding in ("openai", "openai-ollama"):
        from lightrag.llm.openai import openai_complete_if_cache

        _kw: dict = {}
        if host:
            _kw["base_url"] = host
        if api_key:
            _kw["api_key"] = api_key

        async def _llm(prompt, system_prompt=None, history_messages=[], **kw):
            return await openai_complete_if_cache(
                model, prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                **_kw, **kw,
            )
        return _llm

    if binding == "azure_openai":
        from lightrag.llm.openai import openai_complete_if_cache

        api_version = (
            os.environ.get("AZURE_OPENAI_API_VERSION") or
            os.environ.get("OPENAI_API_VERSION") or
            "2024-08-01-preview"
        )
        deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT") or model

        _kw: dict = {
            "use_azure": True,
            "azure_deployment": deployment,
            "api_version": api_version,
        }
        if host:
            _kw["base_url"] = host
        if api_key:
            _kw["api_key"] = api_key

        async def _llm(prompt, system_prompt=None, history_messages=[], **kw):
            return await openai_complete_if_cache(
                deployment, prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                **_kw, **kw,
            )
        return _llm

    if binding == "ollama":
        from lightrag.llm.ollama import ollama_model_complete
        _host = host or "http://localhost:11434"

        async def _llm(prompt, system_prompt=None, history_messages=[], **kw):
            return await ollama_model_complete(
                model, prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                host=_host, **kw,
            )
        return _llm

    raise RuntimeError(
        f"Unsupported LLM_BINDING={binding!r}. "
        "Supported: openai, openai-ollama, azure_openai, ollama"
    )


def _make_embed(binding: str, model: str, dim: int, host: str | None, api_key: str | None):
    from lightrag.utils import wrap_embedding_func_with_attrs

    if binding == "openai":
        from lightrag.llm.openai import openai_embed

        _kw: dict = {"model": model}
        if host:
            _kw["base_url"] = host
        if api_key:
            _kw["api_key"] = api_key

        @wrap_embedding_func_with_attrs(embedding_dim=dim, max_token_size=8192)
        async def _embed(texts):
            return await openai_embed.func(texts, **_kw)
        return _embed

    if binding == "azure_openai":
        from lightrag.llm.openai import openai_embed

        api_version = (
            os.environ.get("AZURE_EMBEDDING_API_VERSION") or
            os.environ.get("AZURE_OPENAI_API_VERSION") or
            os.environ.get("OPENAI_API_VERSION") or
            "2024-08-01-preview"
        )
        deployment = os.environ.get("AZURE_EMBEDDING_DEPLOYMENT") or model

        _kw: dict = {
            "model": deployment,
            "use_azure": True,
            "azure_deployment": deployment,
            "api_version": api_version,
        }
        if host:
            _kw["base_url"] = host
        if api_key:
            _kw["api_key"] = api_key

        @wrap_embedding_func_with_attrs(embedding_dim=dim, max_token_size=8192)
        async def _embed(texts):
            return await openai_embed.func(texts, **_kw)
        return _embed

    if binding == "ollama":
        from lightrag.llm.ollama import ollama_embedding
        _host = host or "http://localhost:11434"

        @wrap_embedding_func_with_attrs(embedding_dim=dim, max_token_size=8192)
        async def _embed(texts):
            return await ollama_embedding(texts, embed_model=model, host=_host)
        return _embed

    raise RuntimeError(
        f"Unsupported EMBEDDING_BINDING={binding!r}. "
        "Supported: openai, azure_openai, ollama"
    )
