"""
Tests for Microstructure Feature Extractor (QARE-MM v1.3)
Verifies:
1. Cross-sectional rank normalization bounds [-1.0, +1.0] and NaN imputation to 0.0
2. 15-channel raw feature computation with Hybrid Dual-Mode fallback
3. Rogers-Satchell, CLV volume skew, Parkinson, Amihud Z-score, and ATR ratio
4. Multi-scale LightGBM summary statistics (105 features) for 2D and 3D tensors
5. Panel rank normalization across multiple universe assets
6. Sliding window sequence tensor generation (Batch, 126, 15)
"""

import numpy as np
import pandas as pd
import pytest

from src.features.microstructure import MicrostructureFeatureExtractor, CHANNEL_NAMES


def test_rank_normalization_bounds() -> None:
    raw_vals = np.array([10.5, -2.1, 0.0, 45.2, -100.0, np.nan, 8.8])
    normalized = MicrostructureFeatureExtractor.rank_normalize(raw_vals)

    # Must be in [-1.0, +1.0]
    assert np.all(normalized >= -1.0)
    assert np.all(normalized <= 1.0)
    # NaN must be mapped to 0.0 (neutral cross-sectional median)
    assert normalized[5] == 0.0
    # Min non-nan value (-100.0) must map to -1.0, max (45.2) to +1.0
    assert np.isclose(normalized[4], -1.0)
    assert np.isclose(normalized[3], 1.0)

    # All-NaN edge case
    all_nan = np.array([np.nan, np.nan])
    assert np.all(MicrostructureFeatureExtractor.rank_normalize(all_nan) == 0.0)

    # Single-element edge case
    single = np.array([42.0])
    assert np.all(MicrostructureFeatureExtractor.rank_normalize(single) == 0.0)


def test_extract_summary_105_2d_and_3d() -> None:
    T = 126
    C = 15
    extractor = MicrostructureFeatureExtractor(lookback_window=T)

    # 1. 2D Sequence (126, 15)
    dummy_sequence = np.random.normal(0, 1, size=(T, C))
    summaries = extractor.extract_summary_105(dummy_sequence)
    assert summaries.shape == (105,)
    assert np.all(np.isfinite(summaries))

    # Verify that slope matches exact linear trend
    # If channel 0 is a pure linear ramp y = 2 * k + 5
    linear_seq = np.zeros((T, C))
    k = np.arange(T)
    linear_seq[:, 0] = 0.5 * k + 10.0
    sum_linear = extractor.extract_summary_105(linear_seq)
    # Slope of channel 0 is feature index 6
    assert np.isclose(sum_linear[6], 0.5, atol=1e-5)

    # 2. 3D Batch (Batch, 126, 15)
    batch_seq = np.random.normal(0, 1, size=(8, T, C))
    batch_summaries = extractor.extract_summary_105(batch_seq)
    assert batch_summaries.shape == (8, 105)
    assert np.all(np.isfinite(batch_summaries))


def test_rogers_satchell_positivity(synthetic_ohlcv_df: pd.DataFrame) -> None:
    df = synthetic_ohlcv_df
    rs_vol = MicrostructureFeatureExtractor.rogers_satchell_volatility(
        df["high"].values,
        df["low"].values,
        df["open"].values,
        df["close"].values,
    )
    assert np.all(rs_vol >= 0.0)
    assert np.all(np.isfinite(rs_vol))


def test_clv_volume_skew_and_typical_price(synthetic_ohlcv_df: pd.DataFrame) -> None:
    df = synthetic_ohlcv_df
    adv20 = df["volume"].rolling(20, min_periods=1).mean().values

    skew = MicrostructureFeatureExtractor.clv_volume_skew(
        df["high"].values,
        df["low"].values,
        df["close"].values,
        df["volume"].values,
        adv20,
    )
    assert len(skew) == len(df)
    assert np.all(np.isfinite(skew))

    tp_dev = MicrostructureFeatureExtractor.typical_price_deviation(
        df["high"].values,
        df["low"].values,
        df["close"].values,
    )
    assert len(tp_dev) == len(df)
    assert np.all(np.isfinite(tp_dev))


def test_parkinson_and_amihud_and_atr(synthetic_ohlcv_df: pd.DataFrame) -> None:
    df = synthetic_ohlcv_df
    returns = np.log(df["close"] / df["close"].shift(1)).fillna(0.0).values

    # Parkinson Volatility >= 0
    pv = MicrostructureFeatureExtractor.parkinson_volatility(df["high"].values, df["low"].values)
    assert np.all(pv >= 0.0)
    assert np.all(np.isfinite(pv))

    # Amihud Illiquidity Z-score bounded in [-5, 5]
    amihud = MicrostructureFeatureExtractor.amihud_illiquidity_z(returns, df["close"].values, df["volume"].values)
    assert len(amihud) == len(df)
    assert np.all(amihud >= -5.0)
    assert np.all(amihud <= 5.0)

    # ATR Volatility Ratio > 0
    atr_ratio = MicrostructureFeatureExtractor.atr_volatility_ratio(
        df["high"].values, df["low"].values, df["close"].values
    )
    assert np.all(atr_ratio > 0.0)
    assert np.all(np.isfinite(atr_ratio))


def test_compute_raw_channels_df(synthetic_ohlcv_df: pd.DataFrame) -> None:
    df = synthetic_ohlcv_df
    residuals = pd.Series(np.random.normal(0, 0.01, size=len(df)), index=df.index)

    extractor = MicrostructureFeatureExtractor(lookback_window=126)
    raw_df = extractor.compute_raw_channels_df(df, residuals)

    assert len(raw_df) == len(df)
    assert len(raw_df.columns) == 15
    for ch in CHANNEL_NAMES:
        assert ch in raw_df.columns
    assert np.all(np.isfinite(raw_df.values))


def test_rank_normalize_panel(synthetic_ohlcv_df: pd.DataFrame) -> None:
    df1 = synthetic_ohlcv_df
    df2 = df1.copy() * 1.05
    df3 = df1.copy() * 0.95

    extractor = MicrostructureFeatureExtractor(lookback_window=126)
    res1 = pd.Series(np.random.normal(0, 0.01, size=len(df1)), index=df1.index)
    res2 = pd.Series(np.random.normal(0, 0.01, size=len(df2)), index=df2.index)
    res3 = pd.Series(np.random.normal(0, 0.01, size=len(df3)), index=df3.index)

    raw1 = extractor.compute_raw_channels_df(df1, res1)
    raw2 = extractor.compute_raw_channels_df(df2, res2)
    raw3 = extractor.compute_raw_channels_df(df3, res3)

    panel_raw = {"AAPL": raw1, "MSFT": raw2, "NVDA": raw3}
    normalized_panel = extractor.rank_normalize_panel(panel_raw)

    assert set(normalized_panel.keys()) == {"AAPL", "MSFT", "NVDA"}

    # Invariants: On every date and channel, rank values must be within [-1.0, +1.0]
    for ticker, norm_df in normalized_panel.items():
        assert len(norm_df) == len(df1)
        assert np.all(norm_df.values >= -1.0)
        assert np.all(norm_df.values <= 1.0)


def test_build_sequence_tensor_and_lgbm_features(synthetic_ohlcv_df: pd.DataFrame) -> None:
    df = synthetic_ohlcv_df  # 150 bars
    residuals = pd.Series(np.random.normal(0, 0.01, size=len(df)), index=df.index)

    extractor = MicrostructureFeatureExtractor(lookback_window=126)
    raw_df = extractor.compute_raw_channels_df(df, residuals)
    norm_df = extractor.rank_normalize_panel({"AAPL": raw_df})["AAPL"]

    # 1. Attention-GRU Sequence Tensor: (N, 126, 15)
    seq_tensor = extractor.build_sequence_tensor(norm_df)
    expected_samples = 150 - 126 + 1
    assert seq_tensor.shape == (expected_samples, 126, 15)

    # 2. LightGBM Tabular Features: (N, 105)
    lgbm_features = extractor.build_lgbm_features(norm_df)
    assert lgbm_features.shape == (expected_samples, 105)
    assert np.all(np.isfinite(lgbm_features))
