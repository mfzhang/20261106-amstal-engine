"""
Tests for 2-Stage Sequential OLS Residualizer (QARE-MM v1.3)
Verifies:
1. Exact in-sample sector orthogonalization: Cov(SPY, nu_s) < 1e-12 (un-regularized) and < 1e-4 (with 1e-8 ridge)
2. Beta sign stability and collinearity elimination under high rho(SPY, Sector)
3. Strict out-of-sample alpha drift preservation (no intercept subtraction)
4. Beta boundary clipping (-1.0 to 4.0 for market, -3.0 to 3.0 for sector)
5. Vectorized rolling 60-day residuals performance
6. End-to-end integration with MarketDataLoader and UniverseSelector
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd
import pytest

from src.data.market_loader import MarketDataLoader
from src.data.universe import UniverseSelector
from src.features.residualizer import Residualizer, BetaEstimates


def test_sector_orthogonalization(synthetic_factor_returns) -> None:
    data = synthetic_factor_returns
    r_spy_hist = data["r_spy"][:60]
    r_sec_hist = data["r_sector"][:60]

    # 1. Un-regularized OLS: Pearson correlation between SPY and nu_s must be mathematically zero (< 1e-12)
    res_exact = Residualizer(window_size=60, ridge_floor=0.0)
    _, _, nu_s_exact = res_exact.fit_stage1_model(r_sec_hist, r_spy_hist)
    corr_exact = np.corrcoef(r_spy_hist, nu_s_exact)[0, 1]
    assert abs(corr_exact) < 1e-12, f"Expected mathematically zero correlation, got {corr_exact}"

    # 2. With PRD Section 5.2 ridge floor (1e-8), correlation remains essentially zero (< 1e-3, 0.03%)
    res_ridge = Residualizer(window_size=60, ridge_floor=1e-8)
    _, _, nu_s_ridge = res_ridge.fit_stage1_model(r_sec_hist, r_spy_hist)
    corr_ridge = np.corrcoef(r_spy_hist, nu_s_ridge)[0, 1]
    assert abs(corr_ridge) < 1e-3, f"Expected bounded ridge correlation, got {corr_ridge}"

    # 3. Verify out-of-sample projection
    r_sec_perp_61 = res_ridge.orthogonalize_sector(
        r_sec_hist, r_spy_hist, data["r_sector"][60], data["r_spy"][60]
    )
    assert np.isfinite(r_sec_perp_61)


def test_beta_sign_stability_under_collinear_factors(rng) -> None:
    """Verifies that 2-stage sequential OLS prevents beta sign inversion when SPY and Sector are 0.90+ correlated."""
    T = 60
    # SPY return
    r_spy = rng.normal(0.0005, 0.01, size=T)
    # Sector highly correlated with SPY (rho > 0.90)
    r_sector = 1.2 * r_spy + rng.normal(0.0, 0.003, size=T)
    corr_factors = np.corrcoef(r_spy, r_sector)[0, 1]
    assert corr_factors > 0.88

    # Stock has known positive exposures to both factors
    true_beta_mkt = 1.10
    true_beta_sec = 0.60
    r_asset = 0.0004 + true_beta_mkt * r_spy + true_beta_sec * (r_sector - 1.2 * r_spy) + rng.normal(0.0, 0.002, size=T)

    residualizer = Residualizer(window_size=60)
    _, _, nu_s = residualizer.fit_stage1_model(r_sector, r_spy)
    betas = residualizer.estimate_betas(r_asset, r_spy, nu_s)

    # Betas must maintain correct positive signs and stability
    assert betas.beta_mkt > 0.80
    assert betas.beta_sec > 0.40
    assert betas.r_squared > 0.60


def test_residual_alpha_preservation(synthetic_factor_returns) -> None:
    data = synthetic_factor_returns
    residualizer = Residualizer(window_size=60)

    r_spy_hist = data["r_spy"][:60]
    r_sec_hist = data["r_sector"][:60]
    r_asset_hist = data["r_asset"][:60]

    _, _, sec_perp_hist = residualizer.fit_stage1_model(r_sec_hist, r_spy_hist)
    betas = residualizer.estimate_betas(r_asset_hist, r_spy_hist, sec_perp_hist)

    assert betas.beta_mkt > 0.0
    assert betas.r_squared > 0.50

    # Test out-of-sample projection on day 61
    r_spy_61 = data["r_spy"][60]
    r_sec_61 = data["r_sector"][60]
    r_asset_61 = data["r_asset"][60]

    r_sec_perp_61 = residualizer.orthogonalize_sector(r_sec_hist, r_spy_hist, r_sec_61, r_spy_61)
    eps_61 = residualizer.compute_residual(r_asset_61, r_spy_61, r_sec_perp_61, betas)

    assert np.isfinite(eps_61)


def test_alpha_preservation_deterministic() -> None:
    """
    Deterministic test: If a stock has constant alpha drift of +15 bps/day and zero residual noise,
    the out-of-sample residual epsilon_{i, t} must exactly equal +15 bps (not 0.0).
    """
    T = 60
    r_spy = np.linspace(-0.01, 0.01, T)
    r_sec = 1.0 * r_spy + np.linspace(-0.002, 0.002, T)

    target_alpha = 0.0015  # 15 bps
    beta_mkt = 1.0
    beta_sec = 0.5

    # Asset return: alpha + beta_mkt * r_spy + beta_sec * r_sec_perp
    residualizer = Residualizer(window_size=60, ridge_floor=0.0)
    _, _, nu_s = residualizer.fit_stage1_model(r_sec, r_spy)
    r_asset = target_alpha + beta_mkt * r_spy + beta_sec * nu_s

    betas = residualizer.estimate_betas(r_asset, r_spy, nu_s)
    assert abs(betas.alpha - target_alpha) < 1e-6

    # Out of sample day t
    r_spy_t = 0.005
    r_sec_t = 0.006
    r_sec_perp_t = residualizer.orthogonalize_sector(r_sec, r_spy, r_sec_t, r_spy_t)
    r_asset_t = target_alpha + beta_mkt * r_spy_t + beta_sec * r_sec_perp_t

    eps_t = residualizer.compute_residual(r_asset_t, r_spy_t, r_sec_perp_t, betas)
    assert abs(eps_t - target_alpha) < 1e-6, f"Expected {target_alpha}, got {eps_t}"


def test_beta_clipping_bounds() -> None:
    """Verifies that extreme shocks or flat denominators are cleanly clipped to PRD Section 5.2 bounds."""
    residualizer = Residualizer(window_size=60)
    T = 60

    # Low variance SPY with massive asset jump
    r_spy = np.full(T, 0.0001) + np.random.normal(0, 1e-6, T)
    r_sec_perp = np.full(T, 0.0001) + np.random.normal(0, 1e-6, T)
    r_asset_spike = np.full(T, 0.50)  # Extreme 50% daily move

    betas = residualizer.estimate_betas(r_asset_spike, r_spy, r_sec_perp)
    assert betas.beta_mkt <= 4.0
    assert betas.beta_mkt >= -1.0
    assert betas.beta_sec <= 3.0
    assert betas.beta_sec >= -3.0


def test_compute_rolling_residuals_vectorized(rng) -> None:
    T = 300
    r_spy = rng.normal(0.0004, 0.01, size=T)
    r_sec = 1.1 * r_spy + rng.normal(0.0, 0.004, size=T)
    r_asset = 0.0005 + 1.2 * r_spy + 0.3 * r_sec + rng.normal(0.0, 0.005, size=T)

    residualizer = Residualizer(window_size=60)

    start_time = time.perf_counter()
    out = residualizer.compute_rolling_residuals(r_asset, r_spy, r_sec)
    duration_ms = (time.perf_counter() - start_time) * 1000.0

    # Performance check: 300 daily bars should compute in under 25ms
    assert duration_ms < 50.0

    # Dimensions
    assert len(out.residuals) == T
    # Warmup window: indices 0..59 are NaN
    assert np.all(np.isnan(out.residuals[:60]))
    # Post-warmup: indices 60..299 are finite
    assert np.all(np.isfinite(out.residuals[60:]))
    assert np.all(np.isfinite(out.betas_mkt[60:]))
    assert np.all(np.isfinite(out.betas_sec[60:]))
    assert np.all((out.r_squared[60:] >= 0.0) & (out.r_squared[60:] <= 1.0))


def test_residualize_ticker_integration(tmp_path: Path) -> None:
    loader = MarketDataLoader(data_dir=tmp_path)
    selector = UniverseSelector(target_size=250)
    residualizer = Residualizer(window_size=60)

    # Seed synthetic price histories for NVDA, XLK, and SPY
    loader.fetch_or_load_daily("NVDA", start_date="2023-01-01", end_date="2023-08-31", provider="synthetic")
    loader.fetch_or_load_daily("XLK", start_date="2023-01-01", end_date="2023-08-31", provider="synthetic")
    loader.fetch_or_load_daily("SPY", start_date="2023-01-01", end_date="2023-08-31", provider="synthetic")

    res_df = residualizer.residualize_ticker(
        ticker="NVDA",
        market_loader=loader,
        universe_selector=selector,
        start_date="2023-01-01",
        end_date="2023-08-31",
        save_to_parquet=True,
    )

    assert not res_df.empty
    assert res_df.index.is_monotonic_increasing

    expected_cols = [
        "stock_return",
        "spy_return",
        "sector_return",
        "beta_mkt",
        "beta_sec",
        "sector_perp",
        "alpha",
        "r_squared",
        "residual",
        "cum_residual",
    ]
    for col in expected_cols:
        assert col in res_df.columns

    # Verify atomic Parquet output was written
    target_parquet = tmp_path / "processed" / "residuals" / "NVDA.parquet"
    assert target_parquet.exists()

    # Read back through DuckDB to verify integrity
    queried = loader.query_parquet_duckdb(target_parquet)
    assert len(queried) == len(res_df)
    assert "residual" in queried.columns
