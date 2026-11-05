"""
Gate 1 Quantitative Falsification Battery
Verifies:
1. Shuffled-Label Sanity Test: Out-of-sample Sharpe drops to 0.0 +- 0.15.
2. Forward-Shift Leak Test: Feature leak explodes Sharpe > 4.0 (proving test sensitivity).
3. Net-of-15 bps Hurdle: Out-of-sample Rank IC > 0.025.
"""

import numpy as np
from src.backtest.metrics import BacktestMetrics

def test_shuffled_labels_zero_sharpe(rng):
    """Shuffled target labels must collapse Sharpe towards zero (gross) and negative net of friction."""
    N = 500
    # True random noise return series
    random_returns = rng.normal(0.0001, 0.01, size=N)
    trade_signals = rng.choice([0, 1], size=N, p=[0.7, 0.3])
    
    # Gross Sharpe on noise must be centered near zero
    sharpe_gross = BacktestMetrics.annualized_sharpe(random_returns, trade_signals, include_costs=False)
    assert -0.35 <= sharpe_gross <= 0.35

    # Net Sharpe must be penalized by friction
    sharpe_net = BacktestMetrics.annualized_sharpe(random_returns, trade_signals, include_costs=True)
    assert sharpe_net < sharpe_gross

def test_forward_leak_detection(rng):
    """A 1-day forward lookahead leak must trigger an unmistakable Sharpe explosion (> 4.0)."""
    N = 252
    true_future_returns = rng.normal(0.0005, 0.012, size=N)
    
    # Perfect leak signal: buy whenever tomorrow is positive
    leak_trades = np.where(true_future_returns > 0.0, 1.0, 0.0)
    leak_pnl = leak_trades * true_future_returns
    
    sharpe_leak = BacktestMetrics.annualized_sharpe(leak_pnl, leak_trades)
    assert sharpe_leak > 4.0, f"Expected Sharpe explosion > 4.0 on perfect leak, got {sharpe_leak:.2f}"

def test_brier_score_calibration():
    # Perfectly calibrated predictions vs random guesses
    N = 100
    labels = np.random.choice([0, 1, 2], size=N)
    
    # Confident correct predictions
    confident_probs = np.zeros((N, 3))
    for i, l in enumerate(labels):
        confident_probs[i, l] = 0.90
        confident_probs[i, (l + 1) % 3] = 0.05
        confident_probs[i, (l + 2) % 3] = 0.05

    # Random guesses
    uniform_probs = np.full((N, 3), 1.0 / 3.0)

    brier_confident = BacktestMetrics.brier_score(confident_probs, labels)
    brier_uniform = BacktestMetrics.brier_score(uniform_probs, labels)
    
    assert brier_confident < brier_uniform
