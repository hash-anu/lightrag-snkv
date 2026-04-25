"""Base class for end-to-end LightRAG integration tests.

Subclasses supply ``_make_rag()`` which returns an initialised LightRAG
instance configured with the desired storage backend.  The base class then
runs a standard suite of insert/query/delete checks.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from typing import AsyncIterator

import pytest

_SAMPLE_TEXT = (
    "Marie Curie was a pioneering physicist and chemist who conducted "
    "groundbreaking research on radioactivity. She was the first woman to win "
    "a Nobel Prize and the only person to win Nobel Prizes in two different "
    "sciences. Pierre Curie, her husband, collaborated with her on this research. "
    "They discovered two elements: polonium and radium. The Curies worked at the "
    "University of Paris."
)

_QUERY = "Who was Marie Curie?"


class BaseLightRAGTest(ABC):
    """Mixin for pytest classes that test a specific LightRAG storage stack."""

    @abstractmethod
    async def _make_rag(self, working_dir: str):
        """Return an initialised (but not yet ``initialize_storages``'d) LightRAG."""
        ...

    @pytest.fixture(autouse=True)
    def _setup_tmp(self, tmp_path):
        self._working_dir = str(tmp_path / "rag_storage")
        os.makedirs(self._working_dir, exist_ok=True)

    async def _get_rag(self):
        rag = await self._make_rag(self._working_dir)
        await rag.initialize_storages()
        return rag

    @pytest.mark.asyncio
    async def test_insert_and_query(self):
        rag = await self._get_rag()
        try:
            await rag.ainsert(_SAMPLE_TEXT)
            result = await rag.aquery(_QUERY)
            assert result is not None
            assert len(str(result)) > 0
        finally:
            await rag.finalize_storages()

    @pytest.mark.asyncio
    async def test_insert_multiple_docs(self):
        rag = await self._get_rag()
        try:
            docs = [
                "Albert Einstein developed the theory of relativity.",
                "Isaac Newton formulated the laws of motion and universal gravitation.",
                "Nikola Tesla invented the alternating current electrical system.",
            ]
            await rag.ainsert(docs)
            result = await rag.aquery("Who worked on physics?")
            assert result is not None
        finally:
            await rag.finalize_storages()

    @pytest.mark.asyncio
    async def test_query_modes(self):
        from lightrag import QueryParam
        rag = await self._get_rag()
        try:
            await rag.ainsert(_SAMPLE_TEXT)
            for mode in ("local", "global", "hybrid", "naive"):
                result = await rag.aquery(_QUERY, param=QueryParam(mode=mode))
                assert result is not None, f"mode={mode} returned None"
        finally:
            await rag.finalize_storages()

    @pytest.mark.asyncio
    async def test_only_need_context(self):
        """``only_need_context=True`` skips the LLM call — pure storage retrieval."""
        from lightrag import QueryParam
        rag = await self._get_rag()
        try:
            await rag.ainsert(_SAMPLE_TEXT)
            result = await rag.aquery(
                _QUERY,
                param=QueryParam(mode="hybrid", only_need_context=True),
            )
            assert result is not None
        finally:
            await rag.finalize_storages()
