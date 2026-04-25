"""Unit tests for SNKVKVStorage (13 test cases)."""
from __future__ import annotations

import pytest
import pytest_asyncio

from lightrag_snkv.snkv_kv_impl import SNKVKVStorage


async def make_storage(global_config, embedding_func, namespace="test_kv"):
    s = SNKVKVStorage(
        namespace=namespace,
        workspace="",
        global_config=global_config,
        embedding_func=embedding_func,
    )
    await s.initialize()
    return s


@pytest.mark.asyncio
async def test_upsert_and_get_by_id(global_config, embedding_func):
    s = await make_storage(global_config, embedding_func)
    await s.upsert({"doc1": {"content": "hello world", "extra": 42}})
    result = await s.get_by_id("doc1")
    assert result is not None
    assert result["content"] == "hello world"
    assert result["extra"] == 42
    await s.finalize()


@pytest.mark.asyncio
async def test_get_by_id_missing(global_config, embedding_func):
    s = await make_storage(global_config, embedding_func)
    result = await s.get_by_id("nonexistent")
    assert result is None
    await s.finalize()


@pytest.mark.asyncio
async def test_get_by_ids_ordered(global_config, embedding_func):
    s = await make_storage(global_config, embedding_func)
    await s.upsert({"a": {"v": 1}, "b": {"v": 2}, "c": {"v": 3}})
    results = await s.get_by_ids(["c", "nonexistent", "a"])
    assert results[0]["v"] == 3
    assert results[1] is None
    assert results[2]["v"] == 1
    await s.finalize()


@pytest.mark.asyncio
async def test_filter_keys_returns_missing(global_config, embedding_func):
    s = await make_storage(global_config, embedding_func)
    await s.upsert({"existing": {"v": 1}})
    missing = await s.filter_keys({"existing", "new_key", "another_new"})
    assert "existing" not in missing
    assert "new_key" in missing
    assert "another_new" in missing
    await s.finalize()


@pytest.mark.asyncio
async def test_delete(global_config, embedding_func):
    s = await make_storage(global_config, embedding_func)
    await s.upsert({"k1": {"v": 1}, "k2": {"v": 2}})
    await s.delete(["k1"])
    assert await s.get_by_id("k1") is None
    assert await s.get_by_id("k2") is not None
    await s.finalize()


@pytest.mark.asyncio
async def test_delete_nonexistent_is_noop(global_config, embedding_func):
    s = await make_storage(global_config, embedding_func)
    # Should not raise
    await s.delete(["ghost_key"])
    await s.finalize()


@pytest.mark.asyncio
async def test_is_empty_true_initially(global_config, embedding_func):
    s = await make_storage(global_config, embedding_func)
    assert await s.is_empty() is True
    await s.finalize()


@pytest.mark.asyncio
async def test_is_empty_false_after_upsert(global_config, embedding_func):
    s = await make_storage(global_config, embedding_func)
    await s.upsert({"k": {"v": 1}})
    assert await s.is_empty() is False
    await s.finalize()


@pytest.mark.asyncio
async def test_upsert_overwrites_existing(global_config, embedding_func):
    s = await make_storage(global_config, embedding_func)
    await s.upsert({"k": {"v": 1}})
    await s.upsert({"k": {"v": 99}})
    result = await s.get_by_id("k")
    assert result["v"] == 99
    await s.finalize()


@pytest.mark.asyncio
async def test_drop_clears_all_data(global_config, embedding_func):
    s = await make_storage(global_config, embedding_func)
    await s.upsert({"a": {"v": 1}, "b": {"v": 2}})
    result = await s.drop()
    assert result["status"] == "success"
    assert await s.is_empty() is True
    await s.finalize()


@pytest.mark.asyncio
async def test_index_done_callback_does_not_raise(global_config, embedding_func):
    s = await make_storage(global_config, embedding_func)
    await s.upsert({"k": {"v": 1}})
    await s.index_done_callback()  # should not raise
    await s.finalize()


@pytest.mark.asyncio
async def test_namespace_isolation(tmp_dir, embedding_func):
    """Two KV storages in the same working_dir but different namespaces are isolated."""
    import os
    gc = {"working_dir": tmp_dir, "embedding_batch_num": 32,
          "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2}}
    s1 = SNKVKVStorage(namespace="ns_a", workspace="", global_config=gc, embedding_func=embedding_func)
    s2 = SNKVKVStorage(namespace="ns_b", workspace="", global_config=gc, embedding_func=embedding_func)
    await s1.initialize()
    await s2.initialize()

    await s1.upsert({"shared_key": {"owner": "ns_a"}})
    result_in_s2 = await s2.get_by_id("shared_key")
    assert result_in_s2 is None

    await s1.finalize()
    await s2.finalize()


@pytest.mark.asyncio
async def test_upsert_batch_atomicity(global_config, embedding_func):
    """All items in a batch upsert are committed together."""
    s = await make_storage(global_config, embedding_func)
    batch = {f"doc{i}": {"value": i} for i in range(50)}
    await s.upsert(batch)
    for i in range(50):
        r = await s.get_by_id(f"doc{i}")
        assert r is not None and r["value"] == i
    await s.finalize()
