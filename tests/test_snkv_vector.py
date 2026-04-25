"""Unit tests for SNKVVectorStorage (12 test cases)."""
from __future__ import annotations

import numpy as np
import pytest
import pytest_asyncio

from lightrag_snkv.snkv_vector_impl import SNKVVectorStorage

EMBED_DIM = 128


def make_embedding_func_fixed(dim=EMBED_DIM):
    """Embedding func that returns a deterministic vector for each text."""
    from unittest.mock import AsyncMock
    rng = np.random.default_rng(42)
    async def _embed(texts, **kw):
        vecs = rng.random((len(texts), dim)).astype(np.float32)
        # Normalise so cosine distance is well-defined
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / (norms + 1e-8)
    _embed.embedding_dim = dim
    _embed.max_token_size = 512
    return _embed


async def make_storage(global_config, namespace="test_vec"):
    ef = make_embedding_func_fixed()
    s = SNKVVectorStorage(
        namespace=namespace,
        workspace="",
        global_config=global_config,
        embedding_func=ef,
        meta_fields={"src_id", "tgt_id", "entity_name"},
    )
    await s.initialize()
    return s


@pytest.mark.asyncio
async def test_upsert_and_query_returns_results(global_config):
    s = await make_storage(global_config)
    await s.upsert({
        "e1": {"content": "cat", "entity_name": "CAT"},
        "e2": {"content": "dog", "entity_name": "DOG"},
    })
    results = await s.query("cat", top_k=2)
    assert len(results) >= 1
    assert all("id" in r and "distance" in r for r in results)
    await s.finalize()


@pytest.mark.asyncio
async def test_query_empty_store_returns_empty(global_config):
    s = await make_storage(global_config)
    results = await s.query("anything", top_k=5)
    assert results == []
    await s.finalize()


@pytest.mark.asyncio
async def test_query_with_precomputed_embedding(global_config):
    s = await make_storage(global_config)
    await s.upsert({"e1": {"content": "hello", "entity_name": "HELLO"}})
    q_emb = np.random.rand(EMBED_DIM).astype(np.float32).tolist()
    results = await s.query("irrelevant", top_k=5, query_embedding=q_emb)
    # Should not crash; may return 0 or more results
    assert isinstance(results, list)
    await s.finalize()


@pytest.mark.asyncio
async def test_get_by_id(global_config):
    s = await make_storage(global_config)
    await s.upsert({"e1": {"content": "hello", "entity_name": "HELLO"}})
    result = await s.get_by_id("e1")
    assert result is not None
    assert result["id"] == "e1"
    assert result["content"] == "hello"
    await s.finalize()


@pytest.mark.asyncio
async def test_get_by_id_missing(global_config):
    s = await make_storage(global_config)
    assert await s.get_by_id("nonexistent") is None
    await s.finalize()


@pytest.mark.asyncio
async def test_get_by_ids_ordered(global_config):
    s = await make_storage(global_config)
    await s.upsert({"a": {"content": "A"}, "b": {"content": "B"}})
    results = await s.get_by_ids(["b", "nonexistent", "a"])
    assert results[0]["id"] == "b"
    assert results[1] is None
    assert results[2]["id"] == "a"
    await s.finalize()


@pytest.mark.asyncio
async def test_delete(global_config):
    s = await make_storage(global_config)
    await s.upsert({"e1": {"content": "x"}, "e2": {"content": "y"}})
    await s.delete(["e1"])
    assert await s.get_by_id("e1") is None
    assert await s.get_by_id("e2") is not None
    await s.finalize()


@pytest.mark.asyncio
async def test_delete_entity(global_config):
    from lightrag.utils import compute_mdhash_id
    s = await make_storage(global_config)
    entity_id = compute_mdhash_id("ALICE", prefix="ent-")
    await s.upsert({entity_id: {"content": "alice description"}})
    await s.delete_entity("ALICE")
    assert await s.get_by_id(entity_id) is None
    await s.finalize()


@pytest.mark.asyncio
async def test_delete_entity_relation(global_config):
    s = await make_storage(global_config)
    await s.upsert({
        "rel1": {"content": "rel", "src_id": "ALICE", "tgt_id": "BOB"},
        "rel2": {"content": "rel", "src_id": "BOB", "tgt_id": "CAROL"},
        "rel3": {"content": "rel", "src_id": "CAROL", "tgt_id": "DAVE"},
    })
    await s.delete_entity_relation("BOB")
    # rel1 and rel2 should be gone; rel3 should remain
    assert await s.get_by_id("rel1") is None
    assert await s.get_by_id("rel2") is None
    assert await s.get_by_id("rel3") is not None
    await s.finalize()


@pytest.mark.asyncio
async def test_get_vectors_by_ids(global_config):
    s = await make_storage(global_config)
    await s.upsert({"e1": {"content": "hello"}})
    vecs = await s.get_vectors_by_ids(["e1", "missing"])
    assert "e1" in vecs
    assert len(vecs["e1"]) == EMBED_DIM
    assert "missing" not in vecs
    await s.finalize()


@pytest.mark.asyncio
async def test_drop_clears_all(global_config):
    s = await make_storage(global_config)
    await s.upsert({"e1": {"content": "x"}})
    result = await s.drop()
    assert result["status"] == "success"
    assert await s.get_by_id("e1") is None
    await s.finalize()


@pytest.mark.asyncio
async def test_upsert_overwrites_existing(global_config):
    s = await make_storage(global_config)
    await s.upsert({"e1": {"content": "original", "entity_name": "OLD"}})
    await s.upsert({"e1": {"content": "updated", "entity_name": "NEW"}})
    result = await s.get_by_id("e1")
    assert result["entity_name"] == "NEW"
    await s.finalize()
