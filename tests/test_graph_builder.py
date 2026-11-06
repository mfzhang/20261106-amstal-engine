"""
Unit tests for Dynamic Rolling GICS & Cross-Correlation GraphBuilder (QARE-MM v1.3)
Verifies:
1. 10-node subgraph invariant (target v0 + 8 peers + 1 sector ETF)
2. Star bidirectional edge topology (18 star + 10 self = 28 edges)
3. Zero-lookahead Point-in-Time peer selection via rolling correlation
4. PyG Batch tensor geometry and root target node slicing via batch.ptr[:-1]
"""

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.universe import UniverseSelector
from src.features.graph_builder import GraphBuilder


def test_graph_builder_10_node_subgraph_topology() -> None:
    selector = UniverseSelector(target_size=250)
    builder = GraphBuilder(universe_selector=selector)

    # Fake 15-channel features for universe assets
    target = "NVDA"
    node_features = {target: np.ones(15, dtype=np.float32)}

    subgraph = builder.build_subgraph(target_ticker=target, node_features=node_features)

    # 1. Node count invariant: exactly 10 nodes
    assert subgraph.num_nodes == 10
    assert subgraph.x.shape == (10, 15)

    # 2. Node identity: Node 0 is target, Node 9 is XLK (NVDA sector ETF)
    assert subgraph.node_tickers[0] == "NVDA"
    assert subgraph.node_tickers[9] == "XLK"
    assert len(subgraph.node_tickers) == 10

    # 3. Edge topology: 18 star edges (9 inward + 9 outward) + 10 self-loops = 28 edges
    assert subgraph.edge_index.shape == (2, 28)
    assert subgraph.edge_attr.shape == (28, 1)

    # Inward star edges (peer -> target 0)
    inward_targets = subgraph.edge_index[1, :18:2]
    assert torch.all(inward_targets == 0)

    # Outward star edges (target 0 -> peer)
    outward_sources = subgraph.edge_index[0, 1:18:2]
    assert torch.all(outward_sources == 0)

    # Self-loops on all 10 nodes
    self_loops_src = subgraph.edge_index[0, 18:]
    self_loops_dst = subgraph.edge_index[1, 18:]
    assert torch.equal(self_loops_src, torch.arange(10))
    assert torch.equal(self_loops_dst, torch.arange(10))


def test_peer_selection_point_in_time_no_lookahead() -> None:
    selector = UniverseSelector(target_size=250)
    builder = GraphBuilder(universe_selector=selector, lookback_corr=30)

    dates = pd.bdate_range("2023-01-01", periods=100)
    target = "NVDA"
    peers = selector.get_peers(target, min_count=8)
    assert len(peers) >= 8

    # Create synthetic residual returns where peer 0 has high correlation in first 50 days,
    # but low correlation in the last 50 days
    rng = np.random.default_rng(42)
    df_res = pd.DataFrame(index=dates)
    base_sig = rng.normal(0, 1, size=len(dates))
    df_res[target] = base_sig

    for p in peers:
        df_res[p] = rng.normal(0, 1, size=len(dates))

    # Give peers[0] high correlation strictly up to day 40
    df_res.iloc[:45, df_res.columns.get_loc(peers[0])] = base_sig[:45] + rng.normal(0, 0.1, size=45)

    # Query as of day 40 (Point-in-Time) -> peers[0] should be ranked first
    as_of_t1 = str(dates[40].date())
    selected_t1 = builder.select_peers(target, residual_returns=df_res, as_of_date=as_of_t1)
    assert len(selected_t1) == 8
    assert selected_t1[0] == peers[0]

    # Break correlation after day 50 for peers[0], but give peers[1] high correlation
    df_res.iloc[50:, df_res.columns.get_loc(peers[1])] = base_sig[50:] + rng.normal(0, 0.1, size=50)
    as_of_t2 = str(dates[90].date())
    selected_t2 = builder.select_peers(target, residual_returns=df_res, as_of_date=as_of_t2)
    assert selected_t2[0] == peers[1]


def test_graph_builder_batching_and_root_slicing() -> None:
    selector = UniverseSelector(target_size=250)
    builder = GraphBuilder(universe_selector=selector)

    tickers = ["AAPL", "NVDA", "MSFT", "AMZN"]
    batch_graphs = []

    for idx, t in enumerate(tickers):
        node_feats = {t: np.full(15, float(idx + 1), dtype=np.float32)}
        subgraph = builder.build_subgraph(target_ticker=t, node_features=node_feats)
        batch_graphs.append(subgraph)

    batch = builder.build_batch(batch_graphs)

    # 4 graphs of 10 nodes = 40 nodes total
    assert batch.num_nodes == 40
    assert batch.x.shape == (40, 15)
    # 4 graphs of 28 edges = 112 edges total
    assert batch.num_edges == 112

    # Root slicing via batch.ptr[:-1]
    root_indices = batch.ptr[:-1]
    assert len(root_indices) == 4
    assert torch.equal(root_indices, torch.tensor([0, 10, 20, 30]))

    # Target nodes must contain their distinct assigned feature values
    root_vals = batch.x[root_indices, 0]
    expected_vals = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    assert torch.allclose(root_vals, expected_vals)
