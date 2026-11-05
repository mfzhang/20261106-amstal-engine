"""
Residualized Dynamic Triple-Barrier Labeling (QARE-MM v1.3)
Evaluates cumulative idiosyncratic return paths against dynamic asymmetric (+2.0 sigma / -1.0 sigma) barriers
with a 10-day vertical horizon barrier to produce ternary classification labels:
- Label 0: Buy / Outperform (Upper barrier hit first: C_eps >= +2.0 sigma)
- Label 1: Hold / Neutral (Vertical barrier reached at k=10 without breach)
- Label 2: Sell / Underperform (Lower barrier hit first: C_eps <= -1.0 sigma)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BarrierResult:
    label: int  # 0: Buy/Outperform, 1: Hold/Neutral, 2: Sell/Underperform
    exit_horizon: int  # 1 to 10 trading days
    realized_residual_return: float
    volatility_scale: float
    end_date_idx: int
    t_start: Optional[Any] = None
    t_end: Optional[Any] = None


class TripleBarrierLabeler:
    """Computes dynamic triple-barrier events on idiosyncratic cumulative return paths."""

    def __init__(
        self,
        upper_multiplier: float = 2.0,
        lower_multiplier: float = 1.0,
        vertical_horizon: int = 10,
        vol_lookback: int = 20,
        min_vol_scale: float = 1e-4,
    ) -> None:
        self.upper_mult = upper_multiplier
        self.lower_mult = lower_multiplier
        self.vertical_horizon = vertical_horizon
        self.vol_lookback = vol_lookback
        self.min_vol_scale = min_vol_scale

    def compute_volatility_scale(self, eps_history_20: np.ndarray) -> float:
        """
        Computes 1D horizon-scaled residual volatility unit:
        sigma_ATR(eps) = sqrt(10) * sqrt(1/20 * sum(eps_t^2))
        """
        arr = np.asarray(eps_history_20, dtype=np.float64)
        if len(arr) < self.vol_lookback:
            raise ValueError(f"Need at least {self.vol_lookback} historical residual points.")
        rms = np.sqrt(np.mean(arr[-self.vol_lookback:] ** 2))
        scale = float(np.sqrt(self.vertical_horizon) * rms)
        return max(scale, self.min_vol_scale)

    def label_path(
        self,
        current_idx: int,
        future_eps_path_10: np.ndarray,
        vol_scale: float,
        t_start: Optional[Any] = None,
        future_dates: Optional[List[Any]] = None,
    ) -> BarrierResult:
        """
        Labels cumulative residual path C_eps(t, k) = sum(eps_{t+1:t+k}) for k in 1..10.
        First barrier hit determines label and exit horizon.
        """
        path = np.asarray(future_eps_path_10, dtype=np.float64)[: self.vertical_horizon]
        cum_returns = np.cumsum(path)
        upper_barrier = self.upper_mult * vol_scale
        lower_barrier = -self.lower_mult * vol_scale

        for k, ret in enumerate(cum_returns, start=1):
            t_end_val = future_dates[k - 1] if future_dates is not None and len(future_dates) >= k else None
            if ret >= upper_barrier:
                return BarrierResult(
                    label=0,  # Buy / Outperform
                    exit_horizon=k,
                    realized_residual_return=float(ret),
                    volatility_scale=vol_scale,
                    end_date_idx=current_idx + k,
                    t_start=t_start,
                    t_end=t_end_val,
                )
            elif ret <= lower_barrier:
                return BarrierResult(
                    label=2,  # Sell / Underperform
                    exit_horizon=k,
                    realized_residual_return=float(ret),
                    volatility_scale=vol_scale,
                    end_date_idx=current_idx + k,
                    t_start=t_start,
                    t_end=t_end_val,
                )

        # Vertical barrier expiration at k = len(cum_returns) <= 10
        final_ret = float(cum_returns[-1]) if len(cum_returns) > 0 else 0.0
        final_k = len(cum_returns)
        final_end_val = (
            future_dates[final_k - 1] if future_dates is not None and len(future_dates) >= final_k else None
        )
        return BarrierResult(
            label=1,  # Hold / Neutral
            exit_horizon=final_k,
            realized_residual_return=final_ret,
            volatility_scale=vol_scale,
            end_date_idx=current_idx + final_k,
            t_start=t_start,
            t_end=final_end_val,
        )

    def label_series(self, residuals: pd.Series) -> pd.DataFrame:
        """
        Computes triple barrier labels and exit horizons for a full residual time series.
        Requires at least vol_lookback (20) points for warmup and vertical_horizon (10) points for resolution.
        Returns DataFrame indexed by evaluation timestamp with columns:
        ['label', 'exit_horizon', 'realized_return', 'vol_scale', 'upper_barrier', 'lower_barrier', 't_start', 't_end']
        """
        s = residuals.dropna()
        n = len(s)
        w = self.vol_lookback
        h = self.vertical_horizon

        if n < w + h:
            raise ValueError(f"Series length ({n}) is too short for lookback ({w}) + horizon ({h}).")

        eps_vals = s.values
        dates = s.index

        rows = []
        valid_indices = []

        for i in range(w, n - h):
            hist_eps = eps_vals[i - w : i]
            future_eps = eps_vals[i : i + h]
            vol_scale = self.compute_volatility_scale(hist_eps)
            future_dates = dates[i : i + h].tolist()

            res = self.label_path(
                current_idx=i,
                future_eps_path_10=future_eps,
                vol_scale=vol_scale,
                t_start=dates[i],
                future_dates=future_dates,
            )

            rows.append({
                "label": res.label,
                "exit_horizon": res.exit_horizon,
                "realized_return": res.realized_residual_return,
                "vol_scale": res.volatility_scale,
                "upper_barrier": self.upper_mult * res.volatility_scale,
                "lower_barrier": -self.lower_mult * res.volatility_scale,
                "t_start": dates[i],
                "t_end": res.t_end,
                "start_idx": i,
                "end_idx": res.end_date_idx,
            })
            valid_indices.append(dates[i])

        df_labels = pd.DataFrame(rows, index=valid_indices)
        return df_labels

    def label_panel(self, residuals_dict: Dict[str, pd.Series]) -> Dict[str, pd.DataFrame]:
        """Labels all tickers in a residual return panel."""
        labeled_dict = {}
        for ticker, series in residuals_dict.items():
            if len(series.dropna()) >= self.vol_lookback + self.vertical_horizon:
                labeled_dict[ticker] = self.label_series(series)
        return labeled_dict
