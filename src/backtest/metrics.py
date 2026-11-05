"""
Quantitative Validation Metrics & Performance Analytics
Computes Rank Information Coefficient (IC), Brier Calibration Score, and Net-of-15bps Sharpe.
"""

from typing import Dict, Any
import numpy as np
from scipy.stats import spearmanr

class BacktestMetrics:
    """Evaluates quantitative falsification metrics net of 15 bps round-trip transaction costs."""

    ROUND_TRIP_COST: float = 0.0015  # 15 bps

    @staticmethod
    def rank_information_coefficient(
        predicted_scores: np.ndarray,
        realized_returns: np.ndarray,
    ) -> float:
        """Computes Spearman Rank Correlation between model scores and forward returns."""
        if len(predicted_scores) < 2:
            return 0.0
        corr, _ = spearmanr(predicted_scores, realized_returns)
        return float(corr) if not np.isnan(corr) else 0.0

    @staticmethod
    def brier_score(
        predicted_probs: np.ndarray,  # (N, 3)
        true_labels: np.ndarray,       # (N,) with values in {0, 1, 2}
    ) -> float:
        """Computes multiclass Brier calibration score (lower is better calibrated)."""
        N, K = predicted_probs.shape
        one_hot = np.zeros((N, K))
        for i, label in enumerate(true_labels):
            one_hot[i, int(label)] = 1.0
        return float(np.mean(np.sum((predicted_probs - one_hot) ** 2, axis=1)))

    @classmethod
    def annualized_sharpe(
        cls,
        gross_returns: np.ndarray,
        trade_indicators: np.ndarray,
        periods_per_year: int = 252,
        include_costs: bool = True,
    ) -> float:
        """Computes annualized Sharpe ratio (net of 15 bps friction by default)."""
        friction = (trade_indicators * cls.ROUND_TRIP_COST) if include_costs else 0.0
        net_returns = gross_returns - friction
        if len(net_returns) < 2 or np.std(net_returns) == 0:
            return 0.0
        return float(np.mean(net_returns) / (np.std(net_returns) + 1e-12) * np.sqrt(periods_per_year))
