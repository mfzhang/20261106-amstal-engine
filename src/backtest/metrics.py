"""
Quantitative Validation Metrics & Performance Analytics
Computes Rank Information Coefficient (IC), Brier Calibration Score, and Net-of-15bps Sharpe.
"""

from typing import Any, Dict, Optional
import numpy as np
from scipy.stats import spearmanr


class BacktestMetrics:
    """Evaluates quantitative falsification metrics net of 15 bps round-trip transaction costs."""

    ROUND_TRIP_COST: float = 0.0015  # 15 bps (0.15%)

    @staticmethod
    def rank_information_coefficient(
        predicted_scores: np.ndarray,
        realized_returns: np.ndarray,
    ) -> float:
        """Computes Spearman Rank Correlation between model scores and forward returns."""
        if len(predicted_scores) < 2:
            return 0.0
        p = np.asarray(predicted_scores, dtype=np.float64)
        r = np.asarray(realized_returns, dtype=np.float64)
        valid = np.isfinite(p) & np.isfinite(r)
        if np.sum(valid) < 2:
            return 0.0
        corr, _ = spearmanr(p[valid], r[valid])
        return float(corr) if not np.isnan(corr) else 0.0

    @staticmethod
    def brier_score(
        predicted_probs: np.ndarray,  # (N, 3)
        true_labels: np.ndarray,       # (N,) with values in {0, 1, 2}
    ) -> float:
        """Computes multiclass Brier calibration score (lower is better calibrated)."""
        probs = np.asarray(predicted_probs, dtype=np.float64)
        labels = np.asarray(true_labels, dtype=int)
        if len(probs) == 0:
            return 0.0
        N, K = probs.shape
        one_hot = np.zeros((N, K), dtype=np.float64)
        for i, label in enumerate(labels):
            if 0 <= label < K:
                one_hot[i, label] = 1.0
        return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

    @classmethod
    def annualized_sharpe(
        cls,
        gross_returns: np.ndarray,
        trade_indicators: np.ndarray,
        periods_per_year: int = 252,
        include_costs: bool = True,
    ) -> float:
        """Computes annualized Sharpe ratio (net of 15 bps friction by default)."""
        g_ret = np.asarray(gross_returns, dtype=np.float64)
        trades = np.asarray(trade_indicators, dtype=np.float64)
        friction = (np.abs(trades) * cls.ROUND_TRIP_COST) if include_costs else 0.0
        net_returns = g_ret - friction
        if len(net_returns) < 2 or np.std(net_returns) == 0:
            return 0.0
        return float(np.mean(net_returns) / (np.std(net_returns) + 1e-12) * np.sqrt(periods_per_year))

    @staticmethod
    def max_drawdown(returns: np.ndarray) -> float:
        """Computes maximum peak-to-trough drawdown from a return series."""
        r = np.asarray(returns, dtype=np.float64)
        if len(r) == 0:
            return 0.0
        cum_equity = np.cumprod(1.0 + r)
        peak = np.maximum.accumulate(cum_equity)
        drawdown = (cum_equity - peak) / (peak + 1e-12)
        return float(np.abs(np.min(drawdown))) if len(drawdown) > 0 else 0.0

    @staticmethod
    def win_rate(net_returns: np.ndarray, trade_mask: Optional[np.ndarray] = None) -> float:
        """Computes proportion of winning trades."""
        r = np.asarray(net_returns, dtype=np.float64)
        if trade_mask is not None:
            mask = np.asarray(trade_mask, dtype=bool)
            r = r[mask]
        if len(r) == 0:
            return 0.0
        return float(np.mean(r > 0.0))

    @staticmethod
    def profit_factor(net_returns: np.ndarray) -> float:
        """Computes ratio of gross gains to gross losses."""
        r = np.asarray(net_returns, dtype=np.float64)
        gains = np.sum(r[r > 0])
        losses = np.abs(np.sum(r[r < 0]))
        if losses == 0:
            return float(gains) if gains > 0 else 1.0
        return float(gains / losses)

    @classmethod
    def evaluate_all(
        cls,
        predicted_scores: np.ndarray,
        predicted_probs: np.ndarray,
        realized_returns: np.ndarray,
        true_labels: np.ndarray,
        trade_indicators: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Calculates comprehensive quantitative metric suite."""
        if trade_indicators is None:
            # Long-only default: enter trade if predicted score > 0
            trade_indicators = np.where(predicted_scores > 0.0, 1.0, 0.0)

        rank_ic = cls.rank_information_coefficient(predicted_scores, realized_returns)
        brier = cls.brier_score(predicted_probs, true_labels)
        sharpe_gross = cls.annualized_sharpe(
            realized_returns * trade_indicators, trade_indicators, include_costs=False
        )
        sharpe_net = cls.annualized_sharpe(
            realized_returns * trade_indicators, trade_indicators, include_costs=True
        )
        net_ret = realized_returns * trade_indicators - np.abs(trade_indicators) * cls.ROUND_TRIP_COST
        mdd = cls.max_drawdown(net_ret)
        wr = cls.win_rate(net_ret, trade_mask=(trade_indicators != 0.0))
        pf = cls.profit_factor(net_ret)

        return {
            "rank_ic": rank_ic,
            "brier_score": brier,
            "sharpe_gross": sharpe_gross,
            "sharpe_net": sharpe_net,
            "max_drawdown": mdd,
            "win_rate": wr,
            "profit_factor": pf,
            "total_trades": int(np.sum(trade_indicators != 0.0)),
        }
