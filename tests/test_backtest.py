"""
Unit & Integration Tests for Purged Walk-Forward Cross-Validation & Metric Engine
Verifies:
1. Locked fold geometry (246d warmup, 756d train, 10d purge, 126d val, 10d purge, 63d test)
2. Strict barrier-end purging preventing cross-boundary leakage
3. Comprehensive quantitative metric evaluations (Rank IC, Brier score, Net Sharpe, Max Drawdown)
4. End-to-end WalkForwardBacktester execution with LightGBM and temperature calibration
"""

import numpy as np
import pytest

from src.backtest.engine import BacktestResult, FoldMetrics, WalkForwardBacktester
from src.backtest.metrics import BacktestMetrics
from src.backtest.walk_forward import FoldIndices, WalkForwardSplitter


def test_walk_forward_splitter_geometry() -> None:
    total_days = 1300
    splitter = WalkForwardSplitter(
        warmup_days=246,
        train_days=756,
        purge_buffer=10,
        val_days=126,
        test_days=63,
        step_days=63,
    )

    barrier_ends = np.arange(total_days) + 5
    folds = list(splitter.split(total_days=total_days, barrier_end_indices=barrier_ends))

    assert len(folds) >= 2

    first_fold = folds[0]
    assert first_fold.fold_id == 0
    # Boundary checks
    assert first_fold.train_end_idx == 246 + 756
    assert first_fold.val_end_idx == (246 + 756 + 10) + 126
    assert first_fold.test_end_idx == ((246 + 756 + 10) + 126 + 10) + 63

    # Fold 1 should advance by exactly step_days = 63
    second_fold = folds[1]
    assert second_fold.fold_id == 1
    assert second_fold.train_end_idx == first_fold.train_end_idx + 63
    assert second_fold.val_end_idx == first_fold.val_end_idx + 63
    assert second_fold.test_end_idx == first_fold.test_end_idx + 63


def test_walk_forward_purging_no_lookahead() -> None:
    total_days = 1250
    splitter = WalkForwardSplitter(
        warmup_days=100,
        train_days=300,
        purge_buffer=10,
        val_days=60,
        test_days=30,
        step_days=30,
    )

    # Construct sample dates and resolutions
    sample_days = np.repeat(np.arange(total_days), 2)  # 2 samples per day
    # Most resolve in 5 days, but sample near cutoff resolves in 15 days (violating purge)
    barrier_ends = sample_days + 5

    # Force a leak violation at day 398 (train ends at 400, resolves at 413)
    violator_idx = np.where(sample_days == 398)[0][0]
    barrier_ends[violator_idx] = 413

    folds = list(splitter.split_panel(sample_days, barrier_ends, total_days=total_days))
    f0 = folds[0]

    # The violating index must be purged from training set
    assert violator_idx not in f0.train_indices

    # All train samples must resolve at or before train_end_idx
    for idx in f0.train_indices:
        assert sample_days[idx] >= 100
        assert sample_days[idx] < f0.train_end_idx
        assert barrier_ends[idx] <= f0.train_end_idx

    # All val samples must resolve at or before val_end_idx
    for idx in f0.val_indices:
        assert sample_days[idx] >= f0.train_end_idx + 10
        assert sample_days[idx] < f0.val_end_idx
        assert barrier_ends[idx] <= f0.val_end_idx

    # All test samples must enter inside test window
    for idx in f0.test_indices:
        assert sample_days[idx] >= f0.val_end_idx + 10
        assert sample_days[idx] < f0.test_end_idx


def test_backtest_metrics_comprehensive(rng) -> None:
    N = 300
    # 1. Rank IC
    true_signal = rng.normal(0, 1, size=N)
    forward_returns = 0.4 * true_signal + rng.normal(0, 0.5, size=N)
    ic_pos = BacktestMetrics.rank_information_coefficient(true_signal, forward_returns)
    assert ic_pos > 0.30

    noise = rng.normal(0, 1, size=N)
    ic_noise = BacktestMetrics.rank_information_coefficient(noise, forward_returns)
    assert abs(ic_noise) < 0.20

    # 2. Brier score
    probs_perfect = np.zeros((N, 3))
    labels = rng.choice([0, 1, 2], size=N)
    for i, l in enumerate(labels):
        probs_perfect[i, l] = 1.0
    assert BacktestMetrics.brier_score(probs_perfect, labels) == 0.0

    probs_uniform = np.full((N, 3), 1.0 / 3.0)
    brier_uniform = BacktestMetrics.brier_score(probs_uniform, labels)
    assert np.isclose(brier_uniform, 2.0 / 3.0, atol=1e-3)

    # 3. Annualized Sharpe
    profitable_returns = rng.normal(0.002, 0.01, size=252)
    trades = np.ones(252)
    gross_sharpe = BacktestMetrics.annualized_sharpe(profitable_returns, trades, include_costs=False)
    net_sharpe = BacktestMetrics.annualized_sharpe(profitable_returns, trades, include_costs=True)
    assert gross_sharpe > 0.0
    assert net_sharpe < gross_sharpe

    # 4. Max Drawdown
    equity_drops = np.array([0.05, -0.10, -0.15, 0.08, -0.05])
    mdd = BacktestMetrics.max_drawdown(equity_drops)
    assert mdd > 0.20

    # 5. Win Rate and Profit Factor
    rets = np.array([0.02, 0.03, -0.01, 0.04, -0.02])
    assert BacktestMetrics.win_rate(rets) == 0.60
    assert BacktestMetrics.profit_factor(rets) == pytest.approx(0.09 / 0.03)

    # 6. Evaluate all
    summary = BacktestMetrics.evaluate_all(
        predicted_scores=true_signal,
        predicted_probs=probs_uniform,
        realized_returns=forward_returns,
        true_labels=labels,
    )
    assert "rank_ic" in summary
    assert "brier_score" in summary
    assert "sharpe_gross" in summary
    assert "sharpe_net" in summary
    assert "max_drawdown" in summary


def test_walk_forward_backtester_end_to_end(rng) -> None:
    # Use compact fold geometry for fast test execution
    splitter = WalkForwardSplitter(
        warmup_days=50,
        train_days=150,
        purge_buffer=5,
        val_days=40,
        test_days=25,
        step_days=25,
    )

    total_days = 300
    n_samples_per_day = 4
    total_samples = total_days * n_samples_per_day
    n_features = 105

    sample_days = np.repeat(np.arange(total_days), n_samples_per_day)
    barrier_ends = sample_days + rng.integers(2, 6, size=total_samples)

    # Synthetic features and predictive signal
    latent_alpha = rng.normal(0, 1, size=total_samples)
    X = rng.normal(0, 1, size=(total_samples, n_features))
    X[:, 0] = latent_alpha  # Channel 0 carries predictive signal

    realized_returns = 0.02 * latent_alpha + rng.normal(0, 0.01, size=total_samples)
    # Ternary labels: 0 if positive return, 2 if negative return, 1 if small
    y = np.ones(total_samples, dtype=int)
    y[realized_returns > 0.008] = 0  # Buy
    y[realized_returns < -0.008] = 2  # Sell

    weights = rng.uniform(0.8, 1.2, size=total_samples)

    backtester = WalkForwardBacktester(
        splitter=splitter,
        round_trip_cost=0.0015,
        execution_mode="LONG_ONLY",
    )

    result = backtester.run(
        X=X,
        y=y,
        sample_days=sample_days,
        barrier_end_days=barrier_ends,
        realized_returns=realized_returns,
        sample_weights=weights,
        total_days=total_days,
    )

    assert isinstance(result, BacktestResult)
    assert len(result.fold_metrics) >= 2
    assert len(result.oos_predictions) > 0
    assert len(result.oos_scores) == len(result.oos_predictions)
    assert len(result.oos_net_returns) == len(result.oos_predictions)
    assert 0.0 <= result.overall_brier_score <= 1.0

    summary_dict = result.summary()
    assert summary_dict["num_folds"] == len(result.fold_metrics)
    assert "mean_rank_ic" in summary_dict
    assert "overall_net_sharpe" in summary_dict
