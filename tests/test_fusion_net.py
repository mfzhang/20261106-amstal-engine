"""
Unit tests for Modality-Masked Fusion Network (QARE-MM v1.3 PRD Section 5.10)
Verifies:
1. Locked (Batch, 66) latent vector: [h_temp (32) || h_graph (32) || D_t (2)]
2. Phase 2 zero-masking for smart-money divergence channels
3. HMM Sigmoid Gate modulation by macro regime multiplier kappa in [0.25, 0.50]
4. Temperature scaling and probability calibration
5. End-to-end multimodal pipeline: AttentionGRU + PeerGAT + RegimeHMM -> FusionNet
"""

import numpy as np
import pytest
import torch

from src.features.graph_builder import GraphBuilder
from src.models.backbone_gru import AttentionGRUBackbone
from src.models.peer_gat import PeerGAT
from src.models.regime_hmm import RegimeHMM
from src.models.fusion_net import ModalityMaskedFusionNet


def test_fusion_net_latent_dimensions_and_phase2_masking() -> None:
    B = 8
    h_temp = torch.randn(B, 32)
    h_graph = torch.randn(B, 32)

    model = ModalityMaskedFusionNet(dropout=0.2)
    logits, h_gated, h_fused = model(h_temp, h_graph, kappa=0.50)

    # 1. Locked multimodal latent dimension invariant: (B, 66)
    assert h_fused.shape == (B, 66)

    # 2. Phase 2 Mode: Alt-data smart money divergence channels [64..65] are zero-masked
    assert torch.all(h_fused[:, 64:] == 0.0)

    # 3. Gated representation is 32-dimensional
    assert h_gated.shape == (B, 32)

    # 4. Classification logits are 3-class ternary
    assert logits.shape == (B, 3)

    # 5. Output probabilities sum to 1.0
    probs = model.predict_proba(h_temp.numpy(), h_graph.numpy(), kappa=0.50)
    assert probs.shape == (B, 3)
    assert np.all(probs >= 0.0)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_fusion_net_hmm_sigmoid_gate_modulation() -> None:
    B = 4
    h_temp = torch.ones(B, 32)
    h_graph = torch.ones(B, 32)

    model = ModalityMaskedFusionNet(dropout=0.0)

    # When kappa=0.50 (low vol) vs kappa=0.25 (turbulent regime)
    _, h_gated_calm, _ = model(h_temp, h_graph, kappa=0.50)
    _, h_gated_turbulent, _ = model(h_temp, h_graph, kappa=0.25)

    # Ratio of gated representations should strictly equal 0.50 / 0.25 = 2.0
    ratio = h_gated_calm / h_gated_turbulent
    assert torch.allclose(ratio, torch.full_like(ratio, 2.0), atol=1e-5)


def test_fusion_net_temperature_scaling() -> None:
    B = 100
    model = ModalityMaskedFusionNet()
    val_logits = np.random.randn(B, 3) * 2.0
    y_val = np.random.choice([0, 1, 2], size=B)

    temp = model.fit_temperature(val_logits, y_val)
    assert temp > 0.0
    assert 0.05 <= temp <= 5.0

    h_temp = np.random.randn(10, 32).astype(np.float32)
    h_graph = np.random.randn(10, 32).astype(np.float32)

    cal_probs = model.predict_calibrated_proba(h_temp, h_graph, kappa=0.35)
    assert cal_probs.shape == (10, 3)
    assert np.all(cal_probs >= 0.0)
    assert np.allclose(cal_probs.sum(axis=1), 1.0, atol=1e-5)


def test_end_to_end_multimodal_pipeline() -> None:
    B = 4
    T = 126
    in_dim = 15

    # 1. Temporal branch
    gru = AttentionGRUBackbone(input_dim=in_dim, hidden_dim=32, num_threads=2)
    seq_batch = np.random.randn(B, T, in_dim).astype(np.float32)
    h_temp = gru.extract_latent(seq_batch)
    assert h_temp.shape == (B, 32)

    # 2. Graph branch
    builder = GraphBuilder()
    gat = PeerGAT(in_channels=in_dim, hidden_dim=32, num_heads=2, num_threads=2)
    subgraphs = [
        builder.build_subgraph(t, {t: np.random.randn(in_dim).astype(np.float32)})
        for t in ["AAPL", "NVDA", "MSFT", "AMZN"]
    ]
    graph_batch = builder.build_batch(subgraphs)
    h_graph = gat.extract_latent(graph_batch)
    assert h_graph.shape == (B, 32)

    # 3. Macro regime branch
    hmm = RegimeHMM(random_state=42, n_fits=3)
    macro_data = np.random.randn(200, 2)
    hmm.fit(macro_data)
    curr_probs = hmm.filter_sequence(macro_data[-B:])
    kappa = np.array([hmm.get_regime_multiplier(curr_probs[i]) for i in range(B)])
    assert len(kappa) == B
    assert np.all(kappa >= 0.25)
    assert np.all(kappa <= 0.50)

    # 4. Multimodal Fusion
    fusion = ModalityMaskedFusionNet()
    calibrated_probs = fusion.predict_calibrated_proba(h_temp, h_graph, kappa=kappa)
    assert calibrated_probs.shape == (B, 3)
    assert np.all(calibrated_probs >= 0.0)
    assert np.allclose(calibrated_probs.sum(axis=1), 1.0, atol=1e-5)
