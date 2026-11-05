"""
Tests for Microstructure Feature Extractor
Verifies 15ch sequence, 105ch LightGBM multi-scale summary, and rank normalization bounds.
"""

import numpy as np
from src.features.microstructure import MicrostructureFeatureExtractor

def test_rank_normalization_bounds():
    raw_vals = np.array([10.5, -2.1, 0.0, 45.2, -100.0, np.nan, 8.8])
    normalized = MicrostructureFeatureExtractor.rank_normalize(raw_vals)
    
    # Must be in [-1.0, +1.0]
    assert np.all(normalized >= -1.0)
    assert np.all(normalized <= 1.0)
    # NaN must be mapped to 0.0
    assert normalized[5] == 0.0
    # Min non-nan value must map to -1.0, max to +1.0
    assert np.isclose(normalized[4], -1.0)
    assert np.isclose(normalized[3], 1.0)

def test_extract_summary_105():
    T = 126
    C = 15
    dummy_sequence = np.random.normal(0, 1, size=(T, C))
    extractor = MicrostructureFeatureExtractor(lookback_window=T)
    
    summaries = extractor.extract_summary_105(dummy_sequence)
    assert len(summaries) == 105
    assert np.all(np.isfinite(summaries))

def test_rogers_satchell_positivity(synthetic_ohlcv_df):
    df = synthetic_ohlcv_df
    rs_vol = MicrostructureFeatureExtractor.rogers_satchell_volatility(
        df["high"].values,
        df["low"].values,
        df["open"].values,
        df["close"].values,
    )
    assert np.all(rs_vol >= 0.0)
    assert np.all(np.isfinite(rs_vol))
