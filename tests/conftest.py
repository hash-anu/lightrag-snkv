"""Shared fixtures for all unit tests."""
from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from unittest.mock import AsyncMock

import numpy as np
import pytest
import pytest_asyncio

EMBED_DIM = 128


def make_embedding_func():
    """Return a mock embedding function returning dim=128 random vectors."""
    func = AsyncMock(
        side_effect=lambda texts, **kw: np.random.rand(len(texts), EMBED_DIM).astype(np.float32)
    )
    func.embedding_dim = EMBED_DIM
    func.max_token_size = 512
    return func


def make_global_config(working_dir: str) -> dict:
    return {
        "working_dir": working_dir,
        "embedding_batch_num": 32,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
    }


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="snkv_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def embedding_func():
    return make_embedding_func()


@pytest.fixture
def global_config(tmp_dir):
    return make_global_config(tmp_dir)
