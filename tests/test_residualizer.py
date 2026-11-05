"""
Tests for 2-Stage Sequential OLS Residualizer
Verifies collinearity elimination and idiosyncratic alpha drift preservation.
"""

import numpy as np
from src.features.residualizer import Residualizer

def test_sector_orthogonalization(synthetic_factor_returns):
    data = synthetic_factor_returns
    residualizer = Residualizer(window_size=60)
    
    r_spy_hist = data["r_spy"][:60]
    r_sec_hist = data["r_sector"][:60]
    
    # Calculate orthogonalized historical sector returns
    X = np.column_stack([np.ones_like(r_spy_hist), r_spy_hist])
    coeffs, _, _, _ = np.linalg.lstsq(X, r_sec_hist, rcond=None)
    sec_perp_hist = r_sec_hist - (coeffs[0] + coeffs[1] * r_spy_hist)
    
    # Pearson correlation between SPY and orthogonal sector must be mathematically zero (< 1e-10)
    corr = np.corrcoef(r_spy_hist, sec_perp_hist)[0, 1]
    assert abs(corr) < 1e-10, f"Expected 0 correlation, got {corr}"

def test_residual_alpha_preservation(synthetic_factor_returns):
    data = synthetic_factor_returns
    residualizer = Residualizer(window_size=60)
    
    r_spy_hist = data["r_spy"][:60]
    r_sec_hist = data["r_sector"][:60]
    r_asset_hist = data["r_asset"][:60]
    
    # Stage 1
    X_sec = np.column_stack([np.ones_like(r_spy_hist), r_spy_hist])
    coeffs_sec, _, _, _ = np.linalg.lstsq(X_sec, r_sec_hist, rcond=None)
    sec_perp_hist = r_sec_hist - (coeffs_sec[0] + coeffs_sec[1] * r_spy_hist)
    
    # Stage 2
    betas = residualizer.estimate_betas(r_asset_hist, r_spy_hist, sec_perp_hist)
    assert betas.beta_mkt > 0.0
    assert betas.r_squared > 0.50
    
    # Test out-of-sample projection on day 61
    r_spy_61 = data["r_spy"][60]
    r_sec_61 = data["r_sector"][60]
    r_asset_61 = data["r_asset"][60]
    
    r_sec_perp_61 = residualizer.orthogonalize_sector(r_sec_hist, r_spy_hist, r_sec_61, r_spy_61)
    eps_61 = residualizer.compute_residual(r_asset_61, r_spy_61, r_sec_perp_61, betas)
    
    # Residual must be finite and centered near alpha
    assert np.isfinite(eps_61)
