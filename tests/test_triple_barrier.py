"""
Tests for Residualized Triple-Barrier Labeling & Sample Uniqueness (QARE-MM v1.3)
Verifies:
1. 1D ATR horizon volatility scaling: sigma_ATR(eps) = sqrt(10) * RMS_20(eps)
2. Asymmetric barrier triggers: Upper (+2 sigma -> Label 0), Lower (-1 sigma -> Label 2), Vertical (10d -> Label 1)
3. Concurrency calculation and fold mean-normalized uniqueness weights (mean == 1.0)
4. Full time series labeling engine and panel labeling
5. Calendar-date purging and 10-day embargo fold splitting
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.label.triple_barrier import TripleBarrierLabeler, BarrierResult
from src.label.weights import UniquenessWeighter


def test_volatility_scaling() -> None:
    labeler = TripleBarrierLabeler(upper_multiplier=2.0, lower_multiplier=1.0, vertical_horizon=10)
    eps_history = np.full(20, 0.01)  # Constant 1% residuals
    vol_scale = labeler.compute_volatility_scale(eps_history)
    # sqrt(10) * sqrt(1/20 * 20 * 0.01^2) = sqrt(10) * 0.01 ~= 0.03162
    expected = np.sqrt(10) * 0.01
    assert np.isclose(vol_scale, expected, atol=1e-5)


def test_triple_barrier_hits() -> None:
    labeler = TripleBarrierLabeler(upper_multiplier=2.0, lower_multiplier=1.0, vertical_horizon=10)
    vol_scale = 0.02  # Upper = +0.04, Lower = -0.02

    # Path hitting upper barrier at step 3
    path_up = np.array([0.01, 0.015, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    res_up = labeler.label_path(current_idx=50, future_eps_path_10=path_up, vol_scale=vol_scale)
    assert res_up.label == 0  # Buy / Outperform
    assert res_up.exit_horizon == 3

    # Path hitting lower barrier at step 2
    path_down = np.array([-0.01, -0.015, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    res_down = labeler.label_path(current_idx=50, future_eps_path_10=path_down, vol_scale=vol_scale)
    assert res_down.label == 2  # Sell / Underperform
    assert res_down.exit_horizon == 2

    # Path expiring at vertical barrier
    path_hold = np.zeros(10)
    res_hold = labeler.label_path(current_idx=50, future_eps_path_10=path_hold, vol_scale=vol_scale)
    assert res_hold.label == 1  # Hold / Neutral
    assert res_hold.exit_horizon == 10


def test_uniqueness_weight_normalization() -> None:
    total_len = 100
    # Overlapping intervals
    intervals = [
        (10, 20),
        (15, 25),
        (18, 28),
        (50, 60),
    ]
    weights = UniquenessWeighter.compute_uniqueness_weights(total_len, intervals)
    assert len(weights) == len(intervals)
    assert np.isclose(np.mean(weights), 1.0, atol=1e-6)

    # Isolated interval must have higher unnormalized weight than heavily overlapped intervals
    weights_raw = UniquenessWeighter.compute_uniqueness_weights(total_len, intervals, normalize=False)
    # intervals[3] is isolated (50-60), while 0, 1, 2 overlap
    assert weights_raw[3] > weights_raw[0]
    assert weights_raw[3] == 1.0  # Concurrency is strictly 1 everywhere in 50..60


def test_label_series_full_lifecycle(rng) -> None:
    T = 120
    dates = pd.bdate_range("2023-01-01", periods=T)
    # Generate residuals with random drift
    residuals = pd.Series(rng.normal(0.0002, 0.012, size=T), index=dates)

    labeler = TripleBarrierLabeler(vertical_horizon=10, vol_lookback=20)
    df_labels = labeler.label_series(residuals)

    assert not df_labels.empty
    expected_rows = T - 20 - 10
    assert len(df_labels) == expected_rows

    expected_cols = [
        "label",
        "exit_horizon",
        "realized_return",
        "vol_scale",
        "upper_barrier",
        "lower_barrier",
        "t_start",
        "t_end",
        "start_idx",
        "end_idx",
    ]
    for col in expected_cols:
        assert col in df_labels.columns

    # Invariants
    assert set(df_labels["label"].unique()).issubset({0, 1, 2})
    assert df_labels["exit_horizon"].min() >= 1
    assert df_labels["exit_horizon"].max() <= 10
    assert np.all(df_labels["upper_barrier"] > 0)
    assert np.all(df_labels["lower_barrier"] < 0)
    # Verification of zero same-day lookahead: t_end must be strictly after t_start
    assert np.all(pd.to_datetime(df_labels["t_end"]) > pd.to_datetime(df_labels["t_start"]))


def test_weight_labeled_df(rng) -> None:
    T = 100
    dates = pd.bdate_range("2023-01-01", periods=T)
    residuals = pd.Series(rng.normal(0.0001, 0.01, size=T), index=dates)

    labeler = TripleBarrierLabeler()
    df_labels = labeler.label_series(residuals)

    weighted_df = UniquenessWeighter.weight_labeled_df(df_labels, normalize=True)
    assert "uniqueness_weight" in weighted_df.columns
    assert np.isclose(weighted_df["uniqueness_weight"].mean(), 1.0, atol=1e-5)
    assert np.all(weighted_df["uniqueness_weight"] > 0.0)


def test_purge_and_embargo(rng) -> None:
    T = 120
    dates = pd.bdate_range("2023-01-01", periods=T)
    residuals = pd.Series(rng.normal(0.0002, 0.01, size=T), index=dates)

    labeler = TripleBarrierLabeler()
    df_labels = labeler.label_series(residuals)

    split_date = "2023-03-15"
    split_dt = pd.to_datetime(split_date)
    embargo_days = 10

    train_df, test_df = UniquenessWeighter.purge_and_embargo(
        df_labels, split_date=split_date, embargo_days=embargo_days
    )

    # 1. Purge verification: NO train sample can resolve at or after split_date
    assert train_df["t_end"].max() < split_dt

    # 2. Embargo verification: NO test sample can start before split_date + embargo_days
    embargo_dt = split_dt + pd.Timedelta(days=embargo_days)
    assert test_df["t_start"].min() >= embargo_dt

    # 3. Disjoint check: Train and test sets must have zero overlap
    assert len(set(train_df.index).intersection(set(test_df.index))) == 0


def test_label_panel(rng) -> None:
    T = 80
    dates = pd.bdate_range("2023-01-01", periods=T)
    res_aapl = pd.Series(rng.normal(0.0005, 0.01, size=T), index=dates)
    res_msft = pd.Series(rng.normal(0.0003, 0.01, size=T), index=dates)

    labeler = TripleBarrierLabeler()
    panel = {"AAPL": res_aapl, "MSFT": res_msft}
    labeled_panel = labeler.label_panel(panel)

    assert set(labeled_panel.keys()) == {"AAPL", "MSFT"}
    assert len(labeled_panel["AAPL"]) == T - 20 - 10
    assert len(labeled_panel["MSFT"]) == T - 20 - 10
