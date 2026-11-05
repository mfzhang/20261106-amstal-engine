"""
Residualized Dynamic Triple-Barrier Labeling
Evaluates cumulative idiosyncratic return paths against asymmetric (+2.0 sigma / -1.0 sigma) barriers.
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np

@dataclass(frozen=True)
class BarrierResult:
    label: int  # 0: Buy/Outperform, 1: Hold/Neutral, 2: Sell/Underperform
    exit_horizon: int  # 1 to 10 trading days
    realized_residual_return: float
    volatility_scale: float
    end_date_idx: int

class TripleBarrierLabeler:
    """Computes dynamic triple-barrier events on idiosyncratic cumulative return paths."""

    def __init__(
        self,
        upper_multiplier: float = 2.0,
        lower_multiplier: float = 1.0,
        vertical_horizon: int = 10,
        vol_lookback: int = 20,
    ) -> None:
        self.upper_mult = upper_multiplier
        self.lower_mult = lower_multiplier
        self.vertical_horizon = vertical_horizon
        self.vol_lookback = vol_lookback

    def compute_volatility_scale(self, eps_history_20: np.ndarray) -> float:
        """
        Computes 1D horizon-scaled residual volatility unit:
        sigma_ATR(eps) = sqrt(10) * sqrt(1/20 * sum(eps_t^2))
        """
        if len(eps_history_20) < self.vol_lookback:
            raise ValueError(f"Need at least {self.vol_lookback} historical residual points.")
        rms = np.sqrt(np.mean(eps_history_20[-self.vol_lookback:] ** 2))
        return float(np.sqrt(self.vertical_horizon) * rms)

    def label_path(
        self,
        current_idx: int,
        future_eps_path_10: np.ndarray,
        vol_scale: float,
    ) -> BarrierResult:
        """
        Labels cumulative residual path C_eps(t, k) = sum(eps_{t+1:t+k}).
        """
        cum_returns = np.cumsum(future_eps_path_10[:self.vertical_horizon])
        upper_barrier = self.upper_mult * vol_scale
        lower_barrier = -self.lower_mult * vol_scale

        for k, ret in enumerate(cum_returns, start=1):
            if ret >= upper_barrier:
                return BarrierResult(
                    label=0,  # Buy
                    exit_horizon=k,
                    realized_residual_return=float(ret),
                    volatility_scale=vol_scale,
                    end_date_idx=current_idx + k,
                )
            elif ret <= lower_barrier:
                return BarrierResult(
                    label=2,  # Sell
                    exit_horizon=k,
                    realized_residual_return=float(ret),
                    volatility_scale=vol_scale,
                    end_date_idx=current_idx + k,
                )

        # Vertical barrier expiration
        final_ret = float(cum_returns[-1]) if len(cum_returns) > 0 else 0.0
        return BarrierResult(
            label=1,  # Hold
            exit_horizon=len(cum_returns),
            realized_residual_return=final_ret,
            volatility_scale=vol_scale,
            end_date_idx=current_idx + len(cum_returns),
        )
