"""
Unit Tests for Machine Learning Backbones & Volatility Regime HMM (QARE-MM v1.3)
Verifies:
1. Sample-Weighted LightGBM training on (Batch, 105) and Temperature Calibration
2. CPU-Pinned Attention-GRU (Batch, 126, 15) latent extraction (Batch, 32) and forward prediction
3. 3-State Volatility Regime HMM with causal log-space forward filtering and variance ordering
"""

import numpy as np
import pytest

from src.models.backbone_lgbm import LightGBMBackbone
from src.models.backbone_gru import AttentionGRUBackbone
from src.models.regime_hmm import RegimeHMM


def test_lightgbm_backbone_fit_and_predict(rng) -> None:
    N = 200
    n_features = 105
    X = rng.normal(0, 1, size=(N, n_features))
    # Labels 0: Buy, 1: Hold, 2: Sell
    y = rng.choice([0, 1, 2], size=N, p=[0.3, 0.4, 0.3])
    weights = rng.uniform(0.5, 1.5, size=N)
    weights /= np.mean(weights)

    model = LightGBMBackbone(n_estimators=30, learning_rate=0.05, random_state=42)
    model.fit(X, y, sample_weights=weights)

    probs = model.predict_proba(X)
    assert probs.shape == (N, 3)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    preds = model.predict(X)
    assert len(preds) == N
    assert set(preds).issubset({0, 1, 2})


def test_lightgbm_temperature_scaling(rng) -> None:
    N = 300
    n_features = 105
    X = rng.normal(0, 1, size=(N, n_features))
    y = rng.choice([0, 1, 2], size=N, p=[0.25, 0.5, 0.25])

    X_train, y_train = X[:200], y[:200]
    X_val, y_val = X[200:], y[200:]

    model = LightGBMBackbone(n_estimators=30, random_state=42)
    model.fit(X_train, y_train)

    # Fit temperature on validation set
    temp = model.fit_temperature(X_val, y_val)
    assert temp > 0.0
    assert 0.05 <= temp <= 5.0

    calibrated_probs = model.predict_calibrated_proba(X_val)
    assert calibrated_probs.shape == (100, 3)
    assert np.all(calibrated_probs >= 0.0)
    assert np.all(calibrated_probs <= 1.0)
    assert np.allclose(calibrated_probs.sum(axis=1), 1.0, atol=1e-5)


def test_attention_gru_latent_extraction_and_predict(rng) -> None:
    Batch = 8
    T = 126
    in_dim = 15

    X = rng.normal(0, 1, size=(Batch, T, in_dim)).astype(np.float32)
    gru_backbone = AttentionGRUBackbone(
        input_dim=in_dim, hidden_dim=32, num_heads=2, num_classes=3, num_threads=2
    )

    # 1. Latent feature vector extraction for multimodal fusion (Batch, 32)
    latent = gru_backbone.extract_latent(X)
    assert latent.shape == (Batch, 32)
    assert np.all(np.isfinite(latent))

    # 2. Probability prediction (Batch, 3)
    probs = gru_backbone.predict_proba(X)
    assert probs.shape == (Batch, 3)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    preds = gru_backbone.predict(X)
    assert len(preds) == Batch
    assert set(preds).issubset({0, 1, 2})


def test_attention_gru_fit(rng) -> None:
    N = 64
    T = 126
    in_dim = 15

    X = rng.normal(0, 1, size=(N, T, in_dim)).astype(np.float32)
    y = rng.choice([0, 1, 2], size=N).astype(np.int64)
    weights = rng.uniform(0.8, 1.2, size=N).astype(np.float32)

    gru = AttentionGRUBackbone(input_dim=in_dim, hidden_dim=32, num_threads=2)
    gru.fit(X, y, sample_weights=weights, epochs=2, batch_size=32)

    probs = gru.predict_proba(X[:10])
    assert probs.shape == (10, 3)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_regime_hmm_causal_forward_filtering(rng) -> None:
    T = 250
    # SPY returns and standardized VIX
    spy_ret = rng.normal(0.0004, 0.01, size=T)
    vix_std = rng.normal(0.0, 1.0, size=T)
    X = np.column_stack([spy_ret, vix_std])

    hmm = RegimeHMM(random_state=42)
    hmm.fit(X)

    # States must be ordered by SPY return variance (feature 0): Sigma_0 <= Sigma_1 <= Sigma_2
    covars = hmm.covars_
    variances = covars[:, 0] if covars.ndim == 2 else np.array([covars[k, 0, 0] for k in range(3)])
    assert variances[0] <= variances[1] <= variances[2]

    # Causal forward filtering
    filtered_probs = hmm.filter_sequence(X)
    assert filtered_probs.shape == (T, 3)
    assert np.all(filtered_probs >= 0.0)
    assert np.all(filtered_probs <= 1.0)
    assert np.allclose(filtered_probs.sum(axis=1), 1.0, atol=1e-5)
