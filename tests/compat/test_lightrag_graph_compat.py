"""LightRAG graph-storage compatibility tests for SNKVGraphStorage.

Ported from lightrag/tests/test_graph_storage.py — all 6 test scenarios run
against SNKVGraphStorage without env-var configuration, external services, or
the --run-integration flag.  Pass/fail semantics are identical to the upstream
tests; print() calls are preserved for diagnostic output on failure.
"""
from __future__ import annotations

import shutil
import tempfile
from unittest.mock import AsyncMock

import numpy as np
import pytest
import pytest_asyncio

from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.types import KnowledgeGraph

from lightrag_snkv.snkv_graph_impl import SNKVGraphStorage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="snkv_compat_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest_asyncio.fixture
async def storage(tmp_dir):
    """Fresh SNKVGraphStorage instance — dropped before and after each test."""
    ef = AsyncMock(side_effect=lambda texts, **kw: np.random.rand(len(texts), 10).astype(np.float32))
    ef.embedding_dim = 10
    ef.max_token_size = 512

    s = SNKVGraphStorage(
        namespace="test_graph",
        workspace="",
        global_config={
            "working_dir": tmp_dir,
            "embedding_batch_num": 10,
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.5},
        },
        embedding_func=ef,
    )
    await s.initialize()
    await s.drop()
    try:
        yield s
    finally:
        await s.drop()
        await s.finalize()


# ---------------------------------------------------------------------------
# 1. Basic: two nodes + one edge, read back, undirected check
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graph_basic(storage):
    node1_id = "Artificial Intelligence"
    node1_data = {
        "entity_id": node1_id,
        "description": "Artificial intelligence is a branch of computer science.",
        "keywords": "AI,Machine Learning,Deep Learning",
        "entity_type": "Technology Field",
    }
    await storage.upsert_node(node1_id, node1_data)

    node2_id = "Machine Learning"
    node2_data = {
        "entity_id": node2_id,
        "description": "Machine learning uses statistical methods to enable systems to learn.",
        "keywords": "Supervised Learning,Unsupervised Learning,Reinforcement Learning",
        "entity_type": "Technology Field",
    }
    await storage.upsert_node(node2_id, node2_data)

    edge_data = {
        "relationship": "includes",
        "weight": 1.0,
        "description": "The field of artificial intelligence includes the subfield of machine learning.",
    }
    await storage.upsert_edge(node1_id, node2_id, edge_data)

    node1_props = await storage.get_node(node1_id)
    assert node1_props is not None, f"Failed to read node: {node1_id}"
    assert node1_props.get("entity_id") == node1_id
    assert node1_props.get("description") == node1_data["description"]
    assert node1_props.get("entity_type") == node1_data["entity_type"]

    edge_props = await storage.get_edge(node1_id, node2_id)
    assert edge_props is not None, f"Failed to read edge: {node1_id} -> {node2_id}"
    assert edge_props.get("relationship") == edge_data["relationship"]
    assert edge_props.get("description") == edge_data["description"]
    assert edge_props.get("weight") == edge_data["weight"]

    reverse_edge_props = await storage.get_edge(node2_id, node1_id)
    assert reverse_edge_props is not None, \
        f"Reverse edge {node2_id} -> {node1_id} not found (not undirected?)"
    assert edge_props == reverse_edge_props, \
        "Forward and reverse edge properties are inconsistent"


# ---------------------------------------------------------------------------
# 2. Advanced: degrees, labels, knowledge graph, delete/remove
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graph_advanced(storage):
    node1_id = "Artificial Intelligence"
    node1_data = {
        "entity_id": node1_id,
        "description": "AI description.",
        "keywords": "AI,ML,DL",
        "entity_type": "Technology Field",
    }
    node2_id = "Machine Learning"
    node2_data = {
        "entity_id": node2_id,
        "description": "ML description.",
        "keywords": "SL,UL,RL",
        "entity_type": "Technology Field",
    }
    node3_id = "Deep Learning"
    node3_data = {
        "entity_id": node3_id,
        "description": "DL description.",
        "keywords": "NN,CNN,RNN",
        "entity_type": "Technology Field",
    }

    await storage.upsert_node(node1_id, node1_data)
    await storage.upsert_node(node2_id, node2_data)
    await storage.upsert_node(node3_id, node3_data)

    edge1_data = {"relationship": "includes", "weight": 1.0, "description": "AI includes ML."}
    edge2_data = {"relationship": "includes", "weight": 1.0, "description": "ML includes DL."}

    await storage.upsert_edge(node1_id, node2_id, edge1_data)
    await storage.upsert_edge(node2_id, node3_id, edge2_data)

    # node_degree
    node1_degree = await storage.node_degree(node1_id)
    assert node1_degree == 1, f"Expected degree 1 for {node1_id}, got {node1_degree}"
    node2_degree = await storage.node_degree(node2_id)
    assert node2_degree == 2, f"Expected degree 2 for {node2_id}, got {node2_degree}"
    node3_degree = await storage.node_degree(node3_id)
    assert node3_degree == 1, f"Expected degree 1 for {node3_id}, got {node3_degree}"

    # edge_degree
    edge_degree = await storage.edge_degree(node1_id, node2_id)
    assert edge_degree == 3, f"Expected edge degree 3, got {edge_degree}"
    reverse_edge_degree = await storage.edge_degree(node2_id, node1_id)
    assert edge_degree == reverse_edge_degree, \
        "Forward/reverse edge degrees are inconsistent"

    # get_node_edges
    node2_edges = await storage.get_node_edges(node2_id)
    assert node2_edges is not None
    assert len(node2_edges) == 2, f"Expected 2 edges for {node2_id}, got {len(node2_edges)}"
    has_node1 = any(
        (e[0] == node1_id and e[1] == node2_id) or (e[0] == node2_id and e[1] == node1_id)
        for e in node2_edges
    )
    has_node3 = any(
        (e[0] == node2_id and e[1] == node3_id) or (e[0] == node3_id and e[1] == node2_id)
        for e in node2_edges
    )
    assert has_node1, f"{node2_id} edges missing connection to {node1_id}"
    assert has_node3, f"{node2_id} edges missing connection to {node3_id}"

    # get_all_labels
    all_labels = await storage.get_all_labels()
    assert len(all_labels) == 3, f"Expected 3 labels, got {len(all_labels)}"
    assert node1_id in all_labels
    assert node2_id in all_labels
    assert node3_id in all_labels

    # get_knowledge_graph
    kg = await storage.get_knowledge_graph("*", max_depth=2, max_nodes=10)
    assert isinstance(kg, KnowledgeGraph)
    assert len(kg.nodes) == 3, f"Expected 3 nodes in KG, got {len(kg.nodes)}"
    assert len(kg.edges) == 2, f"Expected 2 edges in KG, got {len(kg.edges)}"

    # delete_node
    await storage.delete_node(node3_id)
    assert await storage.get_node(node3_id) is None

    # re-insert for subsequent ops
    await storage.upsert_node(node3_id, node3_data)
    await storage.upsert_edge(node2_id, node3_id, edge2_data)

    # remove_edges
    await storage.remove_edges([(node2_id, node3_id)])
    assert await storage.get_edge(node2_id, node3_id) is None, \
        f"Edge {node2_id} -> {node3_id} should be deleted"
    assert await storage.get_edge(node3_id, node2_id) is None, \
        f"Reverse edge {node3_id} -> {node2_id} should also be deleted"

    # remove_nodes
    await storage.remove_nodes([node2_id, node3_id])
    assert await storage.get_node(node2_id) is None
    assert await storage.get_node(node3_id) is None


# ---------------------------------------------------------------------------
# 3. Batch operations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graph_batch_operations(storage):
    chunk1_id, chunk2_id, chunk3_id = "1", "2", "3"

    node1_id = "Artificial Intelligence"
    node1_data = {
        "entity_id": node1_id,
        "description": "AI is a branch of computer science.",
        "keywords": "AI,ML,DL",
        "entity_type": "Technology Field",
        "source_id": GRAPH_FIELD_SEP.join([chunk1_id, chunk2_id]),
    }
    node2_id = "Machine Learning"
    node2_data = {
        "entity_id": node2_id,
        "description": "ML uses statistical methods.",
        "keywords": "SL,UL,RL",
        "entity_type": "Technology Field",
        "source_id": GRAPH_FIELD_SEP.join([chunk2_id, chunk3_id]),
    }
    node3_id = "Deep Learning"
    node3_data = {
        "entity_id": node3_id,
        "description": "DL uses multi-layer neural networks.",
        "keywords": "NN,CNN,RNN",
        "entity_type": "Technology Field",
        "source_id": GRAPH_FIELD_SEP.join([chunk3_id]),
    }
    node4_id = "Natural Language Processing"
    node4_data = {
        "entity_id": node4_id,
        "description": "NLP focuses on human language.",
        "keywords": "NLP,Text,Language",
        "entity_type": "Technology Field",
    }
    node5_id = "Computer Vision"
    node5_data = {
        "entity_id": node5_id,
        "description": "CV focuses on images/videos.",
        "keywords": "CV,Image,Object",
        "entity_type": "Technology Field",
    }

    for nid, nd in [(node1_id, node1_data), (node2_id, node2_data), (node3_id, node3_data),
                    (node4_id, node4_data), (node5_id, node5_data)]:
        await storage.upsert_node(nid, nd)

    edge1_data = {"relationship": "includes", "weight": 1.0, "description": "AI includes ML.",
                  "source_id": GRAPH_FIELD_SEP.join([chunk1_id, chunk2_id])}
    edge2_data = {"relationship": "includes", "weight": 1.0, "description": "ML includes DL.",
                  "source_id": GRAPH_FIELD_SEP.join([chunk2_id, chunk3_id])}
    edge3_data = {"relationship": "includes", "weight": 1.0, "description": "AI includes NLP.",
                  "source_id": GRAPH_FIELD_SEP.join([chunk3_id])}
    edge4_data = {"relationship": "includes", "weight": 1.0, "description": "AI includes CV."}
    edge5_data = {"relationship": "applied to", "weight": 0.8,
                  "description": "DL is applied to NLP."}
    edge6_data = {"relationship": "applied to", "weight": 0.8,
                  "description": "DL is applied to CV."}

    await storage.upsert_edge(node1_id, node2_id, edge1_data)
    await storage.upsert_edge(node2_id, node3_id, edge2_data)
    await storage.upsert_edge(node1_id, node4_id, edge3_data)
    await storage.upsert_edge(node1_id, node5_id, edge4_data)
    await storage.upsert_edge(node3_id, node4_id, edge5_data)
    await storage.upsert_edge(node3_id, node5_id, edge6_data)

    # get_nodes_batch
    node_ids = [node1_id, node2_id, node3_id]
    nodes_dict = await storage.get_nodes_batch(node_ids)
    assert len(nodes_dict) == 3
    assert nodes_dict[node1_id]["description"] == node1_data["description"]
    assert nodes_dict[node2_id]["description"] == node2_data["description"]
    assert nodes_dict[node3_id]["description"] == node3_data["description"]

    # node_degrees_batch
    node_degrees = await storage.node_degrees_batch(node_ids)
    assert node_degrees[node1_id] == 3, f"Expected 3 for {node1_id}, got {node_degrees[node1_id]}"
    assert node_degrees[node2_id] == 2, f"Expected 2 for {node2_id}, got {node_degrees[node2_id]}"
    assert node_degrees[node3_id] == 3, f"Expected 3 for {node3_id}, got {node_degrees[node3_id]}"

    # edge_degrees_batch
    edges = [(node1_id, node2_id), (node2_id, node3_id), (node3_id, node4_id)]
    edge_degrees = await storage.edge_degrees_batch(edges)
    assert edge_degrees[(node1_id, node2_id)] == 5
    assert edge_degrees[(node2_id, node3_id)] == 5
    assert edge_degrees[(node3_id, node4_id)] == 5

    # get_edges_batch
    edge_dicts = [{"src": src, "tgt": tgt} for src, tgt in edges]
    edges_dict = await storage.get_edges_batch(edge_dicts)
    assert len(edges_dict) == 3
    assert edges_dict[(node1_id, node2_id)]["relationship"] == edge1_data["relationship"]
    assert edges_dict[(node2_id, node3_id)]["relationship"] == edge2_data["relationship"]
    assert edges_dict[(node3_id, node4_id)]["relationship"] == edge5_data["relationship"]

    # get_edges_batch — reverse edges (undirected property)
    reverse_edge_dicts = [{"src": tgt, "tgt": src} for src, tgt in edges]
    reverse_edges_dict = await storage.get_edges_batch(reverse_edge_dicts)
    assert len(reverse_edges_dict) == 3
    for (src, tgt), props in edges_dict.items():
        assert (tgt, src) in reverse_edges_dict, \
            f"Reverse edge ({tgt}, {src}) missing from get_edges_batch"
        assert props == reverse_edges_dict[(tgt, src)], \
            f"Edge ({src},{tgt}) and reverse ({tgt},{src}) properties differ"

    # get_nodes_edges_batch
    nodes_edges = await storage.get_nodes_edges_batch([node1_id, node3_id])
    assert len(nodes_edges) == 2
    assert len(nodes_edges[node1_id]) == 3, \
        f"{node1_id} should have 3 edges, got {len(nodes_edges[node1_id])}"
    assert len(nodes_edges[node3_id]) == 3, \
        f"{node3_id} should have 3 edges, got {len(nodes_edges[node3_id])}"

    # undirected: node1 edges go to node2, node4, node5
    n1_out = {tgt for src, tgt in nodes_edges[node1_id] if src == node1_id}
    assert node2_id in n1_out
    assert node4_id in n1_out
    assert node5_id in n1_out

    # undirected: node3 connects to node2, node4, node5 (any direction)
    def connects(edge_list, a, b):
        return any((s == a and t == b) or (s == b and t == a) for s, t in edge_list)

    assert connects(nodes_edges[node3_id], node3_id, node2_id)
    assert connects(nodes_edges[node3_id], node3_id, node4_id)
    assert connects(nodes_edges[node3_id], node3_id, node5_id)


# ---------------------------------------------------------------------------
# 4. Special characters in node IDs, descriptions, and edge payloads
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graph_special_characters(storage):
    node1_id = "Node with 'single quotes'"
    node1_data = {
        "entity_id": node1_id,
        "description": "This description contains 'single quotes', \"double quotes\", and \\backslashes",
        "keywords": "special characters,quotes,escaping",
        "entity_type": "Test Node",
    }
    node2_id = 'Node with "double quotes"'
    node2_data = {
        "entity_id": node2_id,
        "description": "This description contains both 'single quotes' and \"double quotes\" and \\a\\path",
        "keywords": "special characters,quotes,JSON",
        "entity_type": "Test Node",
    }
    node3_id = "Node with \\backslashes\\"
    node3_data = {
        "entity_id": node3_id,
        "description": "This description contains a Windows path C:\\Program Files\\ and escape characters \\n\\t",
        "keywords": "backslashes,paths,escaping",
        "entity_type": "Test Node",
    }

    for nid, nd in [(node1_id, node1_data), (node2_id, node2_data), (node3_id, node3_data)]:
        await storage.upsert_node(nid, nd)

    edge1_data = {
        "relationship": "special 'relationship'",
        "weight": 1.0,
        "description": "This edge description contains 'single quotes', \"double quotes\", and \\backslashes",
    }
    edge2_data = {
        "relationship": 'complex "relationship"\\type',
        "weight": 0.8,
        "description": "Contains SQL injection attempt: SELECT * FROM users WHERE name='admin'--",
    }
    await storage.upsert_edge(node1_id, node2_id, edge1_data)
    await storage.upsert_edge(node2_id, node3_id, edge2_data)

    for node_id, original_data in [(node1_id, node1_data), (node2_id, node2_data), (node3_id, node3_data)]:
        node_props = await storage.get_node(node_id)
        assert node_props is not None, f"Failed to read node: {node_id}"
        assert node_props.get("entity_id") == node_id, \
            f"entity_id mismatch for {node_id}"
        assert node_props.get("description") == original_data["description"], \
            f"description mismatch for {node_id}"

    edge1_props = await storage.get_edge(node1_id, node2_id)
    assert edge1_props is not None
    assert edge1_props.get("relationship") == edge1_data["relationship"]
    assert edge1_props.get("description") == edge1_data["description"]

    edge2_props = await storage.get_edge(node2_id, node3_id)
    assert edge2_props is not None
    assert edge2_props.get("relationship") == edge2_data["relationship"]
    assert edge2_props.get("description") == edge2_data["description"]


# ---------------------------------------------------------------------------
# 5. String-escaping regressions (quoted/backslash node IDs, batch + delete)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graph_string_escaping_regressions(storage):
    center_id = 'Danh mục "bài toán lớn"'
    backslash_id = r"C:\Program Files\LightRAG"
    mixed_id = 'Path "C:\\RAG\\docs"'
    single_quote_id = "Node with 'single quotes'"

    node_payloads = {
        center_id: {
            "entity_id": center_id,
            "description": 'Quoted entity with JSON-ish payload {"path": "C:\\\\temp"}',
            "keywords": 'quotes,"double quotes",unicode',
            "entity_type": "Regression Node",
        },
        backslash_id: {
            "entity_id": backslash_id,
            "description": r"Windows path C:\Program Files\LightRAG\bin",
            "keywords": r"paths,C:\temp,backslashes",
            "entity_type": "Regression Node",
        },
        mixed_id: {
            "entity_id": mixed_id,
            "description": 'Mixed quotes "and" slashes \\ in one entity id',
            "keywords": r'mixed,"quoted",C:\RAG\docs',
            "entity_type": "Regression Node",
        },
        single_quote_id: {
            "entity_id": single_quote_id,
            "description": "Single quotes stay literal in entity identifiers",
            "keywords": "single quotes,escaping",
            "entity_type": "Regression Node",
        },
    }

    for node_id, payload in node_payloads.items():
        await storage.upsert_node(node_id, payload)

    edge_payloads = {
        (center_id, backslash_id): {
            "relationship": r'contains "path"\edge',
            "weight": 1.0,
            "description": r'Links "quoted" title to C:\Program Files\LightRAG',
        },
        (center_id, mixed_id): {
            "relationship": 'references "docs"',
            "weight": 0.8,
            "description": r'Contains both "quotes" and \\backslashes\\',
        },
        (center_id, single_quote_id): {
            "relationship": "mentions 'alias'",
            "weight": 0.6,
            "description": 'Single quote entity linked to "quoted" center node',
        },
    }

    for (src_id, tgt_id), payload in edge_payloads.items():
        await storage.upsert_edge(src_id, tgt_id, payload)

    # single-node reads
    for node_id, payload in node_payloads.items():
        node = await storage.get_node(node_id)
        assert node is not None, f"Expected node {node_id!r} to round-trip"
        assert node["entity_id"] == node_id
        assert node["description"] == payload["description"]

    # batch node reads
    nodes_batch = await storage.get_nodes_batch(list(node_payloads))
    assert set(nodes_batch) == set(node_payloads)
    for node_id, payload in node_payloads.items():
        assert nodes_batch[node_id]["entity_id"] == node_id
        assert nodes_batch[node_id]["description"] == payload["description"]

    # batch degrees
    degrees = await storage.node_degrees_batch(list(node_payloads))
    assert degrees[center_id] == 3
    assert degrees[backslash_id] == 1
    assert degrees[mixed_id] == 1
    assert degrees[single_quote_id] == 1

    def connects(edge_list, a, b):
        return any(
            (src == a and tgt == b) or (src == b and tgt == a) for src, tgt in edge_list
        )

    center_edges = await storage.get_node_edges(center_id)
    assert center_edges is not None
    assert connects(center_edges, center_id, backslash_id)
    assert connects(center_edges, center_id, mixed_id)
    assert connects(center_edges, center_id, single_quote_id)

    batch_edges = await storage.get_nodes_edges_batch(
        [center_id, mixed_id, backslash_id, single_quote_id]
    )
    assert set(batch_edges) == {center_id, mixed_id, backslash_id, single_quote_id}
    assert connects(batch_edges[center_id], center_id, backslash_id)
    assert connects(batch_edges[center_id], center_id, mixed_id)
    assert connects(batch_edges[center_id], center_id, single_quote_id)
    assert connects(batch_edges[mixed_id], center_id, mixed_id)
    assert connects(batch_edges[backslash_id], center_id, backslash_id)
    assert connects(batch_edges[single_quote_id], center_id, single_quote_id)

    # get_edge undirected
    for (src_id, tgt_id), payload in edge_payloads.items():
        fwd = await storage.get_edge(src_id, tgt_id)
        rev = await storage.get_edge(tgt_id, src_id)
        assert fwd is not None, f"get_edge({src_id!r}, {tgt_id!r}) returned None"
        assert rev is not None, \
            f"get_edge({tgt_id!r}, {src_id!r}) returned None — not undirected?"
        assert fwd["relationship"] == payload["relationship"]
        assert fwd["description"] == payload["description"]
        assert rev["relationship"] == fwd["relationship"]
        assert rev["description"] == fwd["description"]

    # has_edge undirected
    for src_id, tgt_id in edge_payloads:
        assert await storage.has_edge(src_id, tgt_id), \
            f"has_edge({src_id!r}, {tgt_id!r}) False after insert"
        assert await storage.has_edge(tgt_id, src_id), \
            f"has_edge({tgt_id!r}, {src_id!r}) False — not undirected?"

    # get_edges_batch undirected
    forward_edges = await storage.get_edges_batch(
        [{"src": src_id, "tgt": tgt_id} for src_id, tgt_id in edge_payloads]
    )
    reverse_edges = await storage.get_edges_batch(
        [{"src": tgt_id, "tgt": src_id} for src_id, tgt_id in edge_payloads]
    )
    assert set(forward_edges) == set(edge_payloads)
    for pair, payload in edge_payloads.items():
        assert forward_edges[pair]["relationship"] == payload["relationship"]
        assert forward_edges[pair]["description"] == payload["description"]
        rev_pair = (pair[1], pair[0])
        assert rev_pair in reverse_edges, \
            f"get_edges_batch missing reverse pair {rev_pair!r}"
        assert reverse_edges[rev_pair]["relationship"] == payload["relationship"]
        assert reverse_edges[rev_pair]["description"] == payload["description"]

    # edge deletion removes both directions
    await storage.remove_edges([(center_id, mixed_id)])
    assert await storage.get_edge(center_id, mixed_id) is None
    assert await storage.get_edge(mixed_id, center_id) is None
    remaining = await storage.get_node_edges(center_id)
    assert remaining is not None
    assert not connects(remaining, center_id, mixed_id)

    # node deletion
    await storage.delete_node(single_quote_id)
    assert await storage.get_node(single_quote_id) is None

    await storage.remove_nodes([center_id, backslash_id])
    assert await storage.get_node(center_id) is None
    assert await storage.get_node(backslash_id) is None
    assert await storage.get_node(mixed_id) is not None


# ---------------------------------------------------------------------------
# 6. Undirected property — comprehensive forward/reverse consistency checks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graph_undirected_property(storage):
    node1_id = "Computer Science"
    node1_data = {
        "entity_id": node1_id,
        "description": "Computer science is the study of computers.",
        "keywords": "computer,science,technology",
        "entity_type": "Discipline",
    }
    node2_id = "Data Structures"
    node2_data = {
        "entity_id": node2_id,
        "description": "A data structure organises and stores data.",
        "keywords": "data,structure,organization",
        "entity_type": "Concept",
    }
    node3_id = "Algorithms"
    node3_data = {
        "entity_id": node3_id,
        "description": "An algorithm is a set of steps for solving problems.",
        "keywords": "algorithm,steps,methods",
        "entity_type": "Concept",
    }

    for nid, nd in [(node1_id, node1_data), (node2_id, node2_data), (node3_id, node3_data)]:
        await storage.upsert_node(nid, nd)

    edge1_data = {
        "relationship": "includes",
        "weight": 1.0,
        "description": "Computer science includes data structures.",
    }
    await storage.upsert_edge(node1_id, node2_id, edge1_data)

    # forward / reverse reads agree
    forward_edge = await storage.get_edge(node1_id, node2_id)
    assert forward_edge is not None
    reverse_edge = await storage.get_edge(node2_id, node1_id)
    assert reverse_edge is not None
    assert forward_edge == reverse_edge, "Forward/reverse edge properties inconsistent"

    edge2_data = {
        "relationship": "includes",
        "weight": 1.0,
        "description": "Computer science includes algorithms.",
    }
    await storage.upsert_edge(node1_id, node3_id, edge2_data)

    # edge degree consistency
    forward_degree = await storage.edge_degree(node1_id, node2_id)
    reverse_degree = await storage.edge_degree(node2_id, node1_id)
    assert forward_degree == reverse_degree, "Forward/reverse edge degrees inconsistent"

    # deletion removes both directions
    await storage.remove_edges([(node1_id, node2_id)])
    assert await storage.get_edge(node1_id, node2_id) is None
    assert await storage.get_edge(node2_id, node1_id) is None, \
        "Reverse edge should be deleted when forward is removed"

    # re-insert for batch test
    await storage.upsert_edge(node1_id, node2_id, edge1_data)

    edge_dicts = [
        {"src": node1_id, "tgt": node2_id},
        {"src": node1_id, "tgt": node3_id},
    ]
    reverse_edge_dicts = [
        {"src": node2_id, "tgt": node1_id},
        {"src": node3_id, "tgt": node1_id},
    ]
    edges_dict = await storage.get_edges_batch(edge_dicts)
    reverse_edges_dict = await storage.get_edges_batch(reverse_edge_dicts)

    for (src, tgt), props in edges_dict.items():
        assert (tgt, src) in reverse_edges_dict, \
            f"Reverse edge ({tgt},{src}) missing from get_edges_batch"
        assert props == reverse_edges_dict[(tgt, src)], \
            f"Batch edge ({src},{tgt}) vs reverse ({tgt},{src}) properties differ"

    # batch node edges
    nodes_edges = await storage.get_nodes_edges_batch([node1_id, node2_id])
    node1_edges = nodes_edges[node1_id]
    node2_edges = nodes_edges[node2_id]

    has_to_node2 = any(s == node1_id and t == node2_id for s, t in node1_edges)
    has_to_node3 = any(s == node1_id and t == node3_id for s, t in node1_edges)
    assert has_to_node2, f"{node1_id} edges missing edge to {node2_id}"
    assert has_to_node3, f"{node1_id} edges missing edge to {node3_id}"

    has_node1_in_node2_edges = any(
        (s == node2_id and t == node1_id) or (s == node1_id and t == node2_id)
        for s, t in node2_edges
    )
    assert has_node1_in_node2_edges, \
        f"{node2_id} edges missing connection to {node1_id}"
