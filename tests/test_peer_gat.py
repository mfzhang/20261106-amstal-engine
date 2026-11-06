"""
Unit tests for CPU-Pinned 2-Layer PeerGAT (QARE-MM v1.3)
Verifies:
1. Target node embedding extraction: (Batch, 32) via batch.ptr[:-1]
2. Standalone classification logits and probability normalization: (Batch, 3)
3. End-to-end training loop and parameter gradient updates
4. Seamless integration with GraphBuilder subgraphs
"""

import numpy as np
import pytest
import torch

from src.features.graph_builder import GraphBuilder
from src.models.peer_gat import PeerGAT


def test_peer_gat_single_subgraph_forward() -> None:
    builder = GraphBuilder()
    gat = PeerGAT(in_channels=15, hidden_dim=32, num_heads=2, num_classes=3, num_threads=2)

    data = builder.build_subgraph(
        target_ticker="AAPL",
        node_features={"AAPL": np.ones(15, dtype=np.float32)},
    )

    h_target, logits = gat(data)

    # 1. Target node representation is 32-dimensional
    assert h_target.shape == (1, 32)
    # 2. Classification logits are 3-class ternary
    assert logits.shape == (1, 3)

    # 3. Probabilities sum to 1.0
    probs = gat.predict_proba(data)
    assert probs.shape == (1, 3)
    assert np.all(probs >= 0.0)
    assert np.allclose(probs.sum(), 1.0, atol=1e-5)


def test_peer_gat_batched_forward_and_latent_extraction() -> None:
    builder = GraphBuilder()
    gat = PeerGAT(in_channels=15, hidden_dim=32, num_heads=2, num_classes=3, num_threads=2)

    tickers = ["AAPL", "MSFT", "NVDA", "ORCL", "CRM", "AMD"]
    B = len(tickers)

    graphs = []
    for idx, t in enumerate(tickers):
        g = builder.build_subgraph(
            target_ticker=t,
            node_features={t: np.full(15, float(idx + 1), dtype=np.float32)},
        )
        graphs.append(g)

    batch = builder.build_batch(graphs)
    assert batch.num_graphs == B

    h_target, logits = gat(batch)

    # Output dimensions match batch size
    assert h_target.shape == (B, 32)
    assert logits.shape == (B, 3)

    # Latent extraction returns (B, 32) float32 NumPy array
    latent = gat.extract_latent(batch)
    assert isinstance(latent, np.ndarray)
    assert latent.shape == (B, 32)
    assert np.all(np.isfinite(latent))


def test_peer_gat_fit_gradient_step() -> None:
    builder = GraphBuilder()
    gat = PeerGAT(in_channels=15, hidden_dim=32, num_heads=2, num_classes=3, num_threads=2)

    tickers = ["AAPL", "MSFT", "NVDA", "AMZN"]
    graphs = [builder.build_subgraph(t, {t: np.random.randn(15).astype(np.float32)}) for t in tickers]
    batch = builder.build_batch(graphs)

    y_train = [0, 1, 2, 0]

    # Model parameters must change after fit
    init_param = next(gat.parameters()).clone().detach()

    gat.fit([batch], y_train, epochs=3, lr=0.01)

    assert gat.is_fitted_ is True
    updated_param = next(gat.parameters()).detach()
    assert not torch.equal(init_param, updated_param)
