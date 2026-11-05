"""
Shared Test Fixtures for Amstal Quant Engine
Provides deterministic synthetic price, volume, and factor return generators.
"""

import pytest
import numpy as np
import pandas as pd

@pytest.fixture
def rng():
    return np.random.default_rng(seed=42)

@pytest.fixture
def synthetic_factor_returns(rng):
    """Generates 200 trading days of synthetic SPY and XLK returns with 0.82 correlation."""
    T = 200
    r_spy = rng.normal(0.0004, 0.01, size=T)
    # Sector has heavy beta to SPY plus idiosyncratic noise
    r_sector = 1.15 * r_spy + rng.normal(0.0, 0.005, size=T)
    # Asset has beta to both plus alpha drift
    alpha_true = 0.0008
    r_asset = alpha_true + 1.2 * r_spy + 0.4 * r_sector + rng.normal(0.0, 0.008, size=T)
    return {
        "r_spy": r_spy,
        "r_sector": r_sector,
        "r_asset": r_asset,
        "alpha_true": alpha_true,
    }

@pytest.fixture
def synthetic_ohlcv_df(rng):
    """Generates 150 daily OHLCV bars."""
    T = 150
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, size=T)))
    high = close * (1.0 + np.abs(rng.normal(0.005, 0.004, size=T)))
    low = close * (1.0 - np.abs(rng.normal(0.005, 0.004, size=T)))
    open_ = low + (high - low) * rng.uniform(0.2, 0.8, size=T)
    volume = rng.lognormal(mean=14.0, sigma=0.5, size=T)
    
    dates = pd.date_range("2024-01-01", periods=T, freq="B")
    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)
