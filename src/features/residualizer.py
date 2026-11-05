"""
2-Stage Sequential OLS Residualizer
Isolates idiosyncratic asset return by stripping market (SPY) and orthogonalized sector factor risk.
"""

from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class BetaEstimates:
    beta_mkt: float
    beta_sec: float
    alpha: float
    r_squared: float

class Residualizer:
    """Computes rolling 60-day 2-stage sequential OLS betas and idiosyncratic residuals."""

    def __init__(self, window_size: int = 60) -> None:
        self.window_size = window_size

    def orthogonalize_sector(
        self,
        r_sec_hist: np.ndarray,
        r_spy_hist: np.ndarray,
        r_sec_current: float,
        r_spy_current: float,
    ) -> float:
        """
        Stage 1: Regresses Sector ETF return on SPY over historical window [t-60, t-1].
        Projects out-of-sample orthogonal sector return for day t.
        """
        # Stack design matrix with intercept
        X = np.column_stack([np.ones_like(r_spy_hist), r_spy_hist])
        coeffs, _, _, _ = np.linalg.lstsq(X, r_sec_hist, rcond=None)
        alpha_sec, gamma = coeffs[0], coeffs[1]
        
        # Out-of-sample projection
        r_sec_perp = r_sec_current - (alpha_sec + gamma * r_spy_current)
        return float(r_sec_perp)

    def estimate_betas(
        self,
        r_asset_hist: np.ndarray,
        r_spy_hist: np.ndarray,
        r_sec_perp_hist: np.ndarray,
    ) -> BetaEstimates:
        """
        Stage 2: Regresses asset return on SPY and orthogonalized sector return.
        """
        X = np.column_stack([np.ones_like(r_spy_hist), r_spy_hist, r_sec_perp_hist])
        coeffs, residuals, _, _ = np.linalg.lstsq(X, r_asset_hist, rcond=None)
        alpha, beta_mkt, beta_sec = coeffs[0], coeffs[1], coeffs[2]
        
        ss_tot = np.sum((r_asset_hist - np.mean(r_asset_hist)) ** 2)
        r_squared = 1.0 - (residuals[0] / (ss_tot + 1e-12)) if len(residuals) > 0 else 0.0
        
        return BetaEstimates(
            beta_mkt=float(beta_mkt),
            beta_sec=float(beta_sec),
            alpha=float(alpha),
            r_squared=float(r_squared),
        )

    def compute_residual(
        self,
        r_asset: float,
        r_spy: float,
        r_sec_perp: float,
        betas: BetaEstimates,
    ) -> float:
        """
        Computes day t out-of-sample idiosyncratic return epsilon_{i, t}.
        Strictly preserves alpha drift by only subtracting factor beta exposures.
        """
        return float(r_asset - (betas.beta_mkt * r_spy + betas.beta_sec * r_sec_perp))
