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


def test_fusion_net_phase3_cross_modal_attention() -> None:
    """Verifies Phase 3 Cross-Modal Attention with event features and temporal sequence."""
    B = 6
    T = 126
    h_temp = torch.randn(B, 32)
    h_graph = torch.randn(B, 32)
    temporal_seq = torch.randn(B, T, 32)

    # Active event features: masks at indices 9..11 are 0.0 (active news and insider)
    event_feats = torch.zeros(B, 12)
    event_feats[:, 0] = 0.75  # sentiment
    event_feats[:, 2] = 0.60  # insider score
    event_feats[:, 9] = 0.0   # news mask: present
    event_feats[:, 10] = 0.0  # insider mask: present
    event_feats[:, 11] = 1.0  # 10q mask: missing

    model = ModalityMaskedFusionNet(dropout=0.0)
    logits, h_gated, h_fused = model(
        h_temp,
        h_graph,
        kappa=0.50,
        event_features=event_feats,
        temporal_seq=temporal_seq,
    )

    # Multi-modal latent invariant: (B, 66)
    assert h_fused.shape == (B, 66)
    assert h_gated.shape == (B, 32)
    assert logits.shape == (B, 3)

    # When all event modalities are missing (masks = 1.0), cross-attention is bypassed
    missing_events = torch.zeros(B, 12)
    missing_events[:, 9:12] = 1.0
    _, _, h_fused_missing = model(
        h_temp,
        h_graph,
        kappa=0.50,
        event_features=missing_events,
        temporal_seq=temporal_seq,
    )
    # The effective temporal representation matches h_temp directly
    assert torch.allclose(h_fused_missing[:, :32], h_temp, atol=1e-5)


def test_fusion_net_phase3_smart_money_incorporation() -> None:
    """Verifies that signed smart-money divergence [D_t^+, D_t^-] enters channels 64..65."""
    B = 4
    h_temp = torch.randn(B, 32)
    h_graph = torch.randn(B, 32)
    d_t = torch.tensor([[0.45, 0.0], [0.0, 0.82], [0.12, 0.0], [0.0, 0.0]], dtype=torch.float32)

    model = ModalityMaskedFusionNet()
    _, _, h_fused = model(h_temp, h_graph, kappa=0.40, d_t=d_t)

    assert h_fused.shape == (B, 66)
    # Channels 64..65 must strictly match d_t
    assert torch.allclose(h_fused[:, 64:66], d_t)


def test_fusion_net_fit_with_event_features() -> None:
    """Verifies training ModalityMaskedFusionNet end-to-end with event features and smart money."""
    B = 64
    T = 20
    h_temp = np.random.randn(B, 32).astype(np.float32)
    h_graph = np.random.randn(B, 32).astype(np.float32)
    temporal_seq = np.random.randn(B, T, 32).astype(np.float32)
    event_feats = np.zeros((B, 12), dtype=np.float32)
    event_feats[:, 0] = np.random.uniform(-1, 1, size=B)
    event_feats[:, 9] = 0.0  # active
    d_t = np.maximum(0.0, np.random.randn(B, 2).astype(np.float32))
    y = np.random.choice([0, 1, 2], size=B)

    model = ModalityMaskedFusionNet(dropout=0.1)
    model.fit(
        h_temp,
        h_graph,
        y,
        kappa=0.40,
        d_t=d_t,
        event_features=event_feats,
        temporal_seq=temporal_seq,
        epochs=3,
        batch_size=16,
    )
    assert model.is_fitted_

    cal_probs = model.predict_calibrated_proba(
        h_temp[:5],
        h_graph[:5],
        kappa=0.40,
        d_t=d_t[:5],
        event_features=event_feats[:5],
        temporal_seq=temporal_seq[:5],
    )
    assert cal_probs.shape == (5, 3)
    assert np.allclose(cal_probs.sum(axis=1), 1.0, atol=1e-5)
