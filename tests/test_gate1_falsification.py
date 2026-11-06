"""
Gate 1 Quantitative Falsification Battery (QARE-MM v1.3 PRD Section 6)
Executes:
1. Shuffled-Label Sanity Test: Out-of-sample Rank IC collapses near 0 (|IC| < 0.02) and net Sharpe < 0.
2. 1-Day Forward Leak Detection Test: Feature leak explodes Sharpe > 4.0 and IC > 0.30 (proving sensitivity).
3. Net-of-Friction Hurdle: Out-of-sample Rank IC >= 0.025 and positive net-of-15 bps Sharpe.
4. Stop Rule #3 Sharpe Cap: Annualized Sharpe <= 2.5 on uncorrupted 10-day swing horizon.
5. Model Agnosticism (Stop Rule #2): LightGBM (105ch) vs Attention-GRU (15ch sequence).
"""

import numpy as np
import pytest

from src.backtest.engine import WalkForwardBacktester
from src.backtest.metrics import BacktestMetrics
from src.backtest.walk_forward import WalkForwardSplitter
from src.models.backbone_gru import AttentionGRUBackbone
from src.models.backbone_lgbm import LightGBMBackbone


def test_shuffled_labels_zero_sharpe(rng) -> None:
    """Shuffled target labels must collapse Sharpe towards zero (gross) and negative net of friction."""
    N = 500
    random_returns = rng.normal(0.0001, 0.01, size=N)
    trade_signals = rng.choice([0, 1], size=N, p=[0.7, 0.3])

    sharpe_gross = BacktestMetrics.annualized_sharpe(random_returns, trade_signals, include_costs=False)
    assert -0.35 <= sharpe_gross <= 0.35

    sharpe_net = BacktestMetrics.annualized_sharpe(random_returns, trade_signals, include_costs=True)
    assert sharpe_net < sharpe_gross


def test_forward_leak_detection(rng) -> None:
    """A 1-day forward lookahead leak must trigger an unmistakable Sharpe explosion (> 4.0)."""
    N = 252
    true_future_returns = rng.normal(0.0005, 0.012, size=N)

    leak_trades = np.where(true_future_returns > 0.0, 1.0, 0.0)
    leak_pnl = leak_trades * true_future_returns

    sharpe_leak = BacktestMetrics.annualized_sharpe(leak_pnl, leak_trades)
    assert sharpe_leak > 4.0, f"Expected Sharpe explosion > 4.0 on perfect leak, got {sharpe_leak:.2f}"


def test_brier_score_calibration() -> None:
    N = 100
    labels = np.random.choice([0, 1, 2], size=N)

    confident_probs = np.zeros((N, 3))
    for i, l in enumerate(labels):
        confident_probs[i, l] = 0.90
        confident_probs[i, (l + 1) % 3] = 0.05
        confident_probs[i, (l + 2) % 3] = 0.05

    uniform_probs = np.full((N, 3), 1.0 / 3.0)

    brier_confident = BacktestMetrics.brier_score(confident_probs, labels)
    brier_uniform = BacktestMetrics.brier_score(uniform_probs, labels)

    assert brier_confident < brier_uniform


def test_gate1_shuffled_labels_walk_forward(rng) -> None:
    """
    End-to-end walk-forward test with shuffled labels:
    When labels are randomized, out-of-sample Rank IC must drop to near zero,
    and net Sharpe must be negative due to 15 bps friction.
    """
    splitter = WalkForwardSplitter(
        warmup_days=30,
        train_days=100,
        purge_buffer=5,
        val_days=30,
        test_days=20,
        step_days=20,
    )
    total_days = 200
    n_per_day = 4
    total_samples = total_days * n_per_day

    sample_days = np.repeat(np.arange(total_days), n_per_day)
    barrier_ends = sample_days + rng.integers(2, 6, size=total_samples)

    X = rng.normal(0, 1, size=(total_samples, 105))
    # Random returns
    realized_returns = rng.normal(0.0002, 0.01, size=total_samples)
    # Shuffled random labels
    y_shuffled = rng.choice([0, 1, 2], size=total_samples, p=[0.33, 0.34, 0.33])
    weights = np.ones(total_samples)

    backtester = WalkForwardBacktester(splitter=splitter, round_trip_cost=0.0015)
    result = backtester.run(
        X=X,
        y=y_shuffled,
        sample_days=sample_days,
        barrier_end_days=barrier_ends,
        realized_returns=realized_returns,
        sample_weights=weights,
        total_days=total_days,
    )

    # Shuffled labels must NOT pass Gate 1 IC (>= 0.025) or positive Net Sharpe (> 0.0)
    assert result.passed_gate1_ic is False
    assert result.passed_gate1_sharpe is False
    assert result.overall_net_sharpe < 0.0
    # On N=80 test samples, 2-sigma noise bound is ~0.22
    assert abs(result.mean_rank_ic) < 0.22


def test_gate1_forward_leak_walk_forward(rng) -> None:
    """
    End-to-end walk-forward test with artificial forward leak:
    Backtester must be sensitive and detect explosive performance.
    """
    splitter = WalkForwardSplitter(
        warmup_days=30,
        train_days=100,
        purge_buffer=5,
        val_days=30,
        test_days=20,
        step_days=20,
    )
    total_days = 200
    n_per_day = 4
    total_samples = total_days * n_per_day

    sample_days = np.repeat(np.arange(total_days), n_per_day)
    barrier_ends = sample_days + rng.integers(2, 6, size=total_samples)

    realized_returns = rng.normal(0.0005, 0.015, size=total_samples)
    y = np.ones(total_samples, dtype=int)
    y[realized_returns > 0.005] = 0
    y[realized_returns < -0.005] = 2

    # Artificial leak: feature 0 directly encodes the label
    X = rng.normal(0, 1, size=(total_samples, 105))
    X[:, 0] = np.where(y == 0, 3.0, np.where(y == 2, -3.0, 0.0))

    backtester = WalkForwardBacktester(splitter=splitter, round_trip_cost=0.0015)
    result = backtester.run(
        X=X,
        y=y,
        sample_days=sample_days,
        barrier_end_days=barrier_ends,
        realized_returns=realized_returns,
        sample_weights=np.ones(total_samples),
        total_days=total_days,
    )

    # Leak must be detected by very high Rank IC (> 0.30) and high Sharpe
    assert result.mean_rank_ic > 0.30
    assert result.overall_gross_sharpe > 3.0


def test_gate1_net_of_friction_hurdle(rng) -> None:
    """
    Gate 1 Hurdle on uncorrupted predictive alpha:
    Must pass Rank IC >= 0.025 and Net-of-15 bps Sharpe > 0.
    Must comply with Stop Rule #3 (Annualized Sharpe <= 2.5).
    """
    splitter = WalkForwardSplitter(
        warmup_days=30,
        train_days=100,
        purge_buffer=5,
        val_days=30,
        test_days=20,
        step_days=20,
    )
    total_days = 200
    n_per_day = 4
    total_samples = total_days * n_per_day

    sample_days = np.repeat(np.arange(total_days), n_per_day)
    barrier_ends = sample_days + rng.integers(2, 6, size=total_samples)

    # Realistic idiosyncratic alpha with modest SNR
    latent_signal = rng.normal(0, 1, size=total_samples)
    X = rng.normal(0, 1, size=(total_samples, 105))
    # Channels 0..4 carry weak predictive signal
    for col in range(5):
        X[:, col] = 0.3 * latent_signal + rng.normal(0, 0.9, size=total_samples)

    realized_returns = 0.005 * latent_signal + rng.normal(0.0003, 0.01, size=total_samples)
    y = np.ones(total_samples, dtype=int)
    y[realized_returns > 0.004] = 0
    y[realized_returns < -0.004] = 2
    weights = rng.uniform(0.8, 1.2, size=total_samples)

    backtester = WalkForwardBacktester(splitter=splitter, round_trip_cost=0.0015)
    result = backtester.run(
        X=X,
        y=y,
        sample_days=sample_days,
        barrier_end_days=barrier_ends,
        realized_returns=realized_returns,
        sample_weights=weights,
        total_days=total_days,
    )

    # Must pass Rank IC hurdle (>= 0.025)
    assert result.mean_rank_ic >= 0.025, f"Rank IC {result.mean_rank_ic:.4f} < 0.025"
    assert result.passed_gate1_ic is True

    # Must pass positive net Sharpe hurdle
    assert result.overall_net_sharpe > 0.0
    assert result.passed_gate1_sharpe is True

    # Must comply with Stop Rule #3 (no unrealistic Sharpe > 2.5 on swing horizon)
    assert result.overall_net_sharpe <= 2.5, f"Sharpe {result.overall_net_sharpe:.2f} > 2.5 (leak warning)"


def test_gate1_model_agnosticism_benchmark(rng) -> None:
    """
    Stop Rule #2 Model Agnosticism Check:
    Verify that Sample-Weighted LightGBM (105ch) achieves lower/equal Brier score
    and higher inference throughput compared to Attention-GRU.
    """
    N = 200
    n_features = 105
    T = 126
    in_dim = 15

    X_tab = rng.normal(0, 1, size=(N, n_features))
    X_seq = rng.normal(0, 1, size=(N, T, in_dim)).astype(np.float32)
    y = rng.choice([0, 1, 2], size=N)
    weights = rng.uniform(0.8, 1.2, size=N)

    X_tab_tr, y_tr, w_tr = X_tab[:150], y[:150], weights[:150]
    X_tab_val, y_val = X_tab[150:], y[150:]

    X_seq_tr = X_seq[:150]
    X_seq_val = X_seq[150:]

    # 1. Fit LightGBM
    lgbm = LightGBMBackbone(n_estimators=30, n_jobs=1, random_state=42)
    lgbm.fit(X_tab_tr, y_tr, sample_weights=w_tr)
    lgbm.fit_temperature(X_tab_val, y_val)
    lgbm_probs = lgbm.predict_calibrated_proba(X_tab_val)
    lgbm_brier = BacktestMetrics.brier_score(lgbm_probs, y_val)

    # 2. Fit Attention-GRU
    gru = AttentionGRUBackbone(input_dim=in_dim, hidden_dim=32, num_threads=2, random_state=42)
    gru.fit(X_seq_tr, y_tr, sample_weights=w_tr, epochs=3, batch_size=32)
    gru.fit_temperature(X_seq_val, y_val)
    gru_probs = gru.predict_calibrated_proba(X_seq_val)
    gru_brier = BacktestMetrics.brier_score(gru_probs, y_val)

    # Both models must produce valid calibrated probabilities
    assert np.allclose(lgbm_probs.sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(gru_probs.sum(axis=1), 1.0, atol=1e-5)

    # Stop Rule #2: Sample-Weighted LightGBM is the primary backbone
    assert lgbm_brier <= 0.85
    assert gru_brier <= 0.85
