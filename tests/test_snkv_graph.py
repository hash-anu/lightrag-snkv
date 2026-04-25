"""Unit tests for SNKVGraphStorage (14 test cases)."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from lightrag_snkv.snkv_graph_impl import SNKVGraphStorage


async def make_storage(global_config):
    from lightrag.namespace import NameSpace
    ef = AsyncMock()
    ef.embedding_dim = 128
    s = SNKVGraphStorage(
        namespace="graph_store",
        workspace="",
        global_config=global_config,
        embedding_func=ef,
    )
    await s.initialize()
    return s


@pytest.mark.asyncio
async def test_upsert_and_get_node(global_config):
    s = await make_storage(global_config)
    await s.upsert_node("Alice", {"entity_type": "PERSON", "description": "A person"})
    node = await s.get_node("Alice")
    assert node is not None
    assert node["entity_type"] == "PERSON"
    await s.finalize()


@pytest.mark.asyncio
async def test_has_node_true(global_config):
    s = await make_storage(global_config)
    await s.upsert_node("Alice", {})
    assert await s.has_node("Alice") is True
    await s.finalize()


@pytest.mark.asyncio
async def test_has_node_false(global_config):
    s = await make_storage(global_config)
    assert await s.has_node("Ghost") is False
    await s.finalize()


@pytest.mark.asyncio
async def test_upsert_and_get_edge(global_config):
    s = await make_storage(global_config)
    await s.upsert_node("Alice", {})
    await s.upsert_node("Bob", {})
    await s.upsert_edge("Alice", "Bob", {"weight": 1.0, "description": "friends"})
    edge = await s.get_edge("Alice", "Bob")
    assert edge is not None
    assert edge["weight"] == 1.0
    await s.finalize()


@pytest.mark.asyncio
async def test_has_edge_undirected(global_config):
    s = await make_storage(global_config)
    await s.upsert_node("Alice", {})
    await s.upsert_node("Bob", {})
    await s.upsert_edge("Alice", "Bob", {"weight": 1.0})
    # Both directions should return True
    assert await s.has_edge("Alice", "Bob") is True
    assert await s.has_edge("Bob", "Alice") is True
    await s.finalize()


@pytest.mark.asyncio
async def test_get_node_edges(global_config):
    s = await make_storage(global_config)
    await s.upsert_node("Alice", {})
    await s.upsert_node("Bob", {})
    await s.upsert_node("Carol", {})
    await s.upsert_edge("Alice", "Bob", {})
    await s.upsert_edge("Alice", "Carol", {})
    edges = await s.get_node_edges("Alice")
    assert edges is not None
    neighbours = {e[1] for e in edges}
    assert "Bob" in neighbours
    assert "Carol" in neighbours
    await s.finalize()


@pytest.mark.asyncio
async def test_get_node_edges_nonexistent_node(global_config):
    s = await make_storage(global_config)
    edges = await s.get_node_edges("Ghost")
    assert edges is None
    await s.finalize()


@pytest.mark.asyncio
async def test_node_degree(global_config):
    s = await make_storage(global_config)
    await s.upsert_node("A", {})
    await s.upsert_node("B", {})
    await s.upsert_node("C", {})
    await s.upsert_edge("A", "B", {})
    await s.upsert_edge("A", "C", {})
    assert await s.node_degree("A") == 2
    assert await s.node_degree("B") == 1
    await s.finalize()


@pytest.mark.asyncio
async def test_remove_node(global_config):
    s = await make_storage(global_config)
    await s.upsert_node("Alice", {})
    await s.remove_nodes(["Alice"])
    assert await s.has_node("Alice") is False
    await s.finalize()


@pytest.mark.asyncio
async def test_remove_edge(global_config):
    s = await make_storage(global_config)
    await s.upsert_node("A", {})
    await s.upsert_node("B", {})
    await s.upsert_edge("A", "B", {})
    await s.remove_edges([("A", "B")])
    assert await s.has_edge("A", "B") is False
    await s.finalize()


@pytest.mark.asyncio
async def test_get_all_labels(global_config):
    s = await make_storage(global_config)
    await s.upsert_node("Zebra", {})
    await s.upsert_node("Apple", {})
    labels = await s.get_all_labels()
    assert labels == ["Apple", "Zebra"]
    await s.finalize()


@pytest.mark.asyncio
async def test_get_popular_labels(global_config):
    s = await make_storage(global_config)
    await s.upsert_node("Hub", {})
    await s.upsert_node("Leaf1", {})
    await s.upsert_node("Leaf2", {})
    await s.upsert_node("Leaf3", {})
    await s.upsert_edge("Hub", "Leaf1", {})
    await s.upsert_edge("Hub", "Leaf2", {})
    await s.upsert_edge("Hub", "Leaf3", {})
    popular = await s.get_popular_labels(limit=1)
    assert popular[0] == "Hub"
    await s.finalize()


@pytest.mark.asyncio
async def test_search_labels(global_config):
    s = await make_storage(global_config)
    for name in ["AliceWonder", "AliceMirror", "BobSmith", "CarolDavis"]:
        await s.upsert_node(name, {})
    results = await s.search_labels("Alice", limit=10)
    assert any("Alice" in r for r in results)
    await s.finalize()


@pytest.mark.asyncio
async def test_get_knowledge_graph_star(global_config):
    s = await make_storage(global_config)
    await s.upsert_node("Center", {"entity_type": "HUB"})
    for i in range(3):
        await s.upsert_node(f"Spoke{i}", {})
        await s.upsert_edge("Center", f"Spoke{i}", {"weight": 1.0})
    kg = await s.get_knowledge_graph("Center", max_depth=1, max_nodes=100)
    assert len(kg.nodes) == 4  # center + 3 spokes
    assert len(kg.edges) == 3
    assert kg.is_truncated is False
    await s.finalize()
