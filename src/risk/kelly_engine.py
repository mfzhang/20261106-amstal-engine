"""
Multi-Outcome Fractional Kelly & Portfolio Exposure Normalization (QARE-MM v1.3)
Solves concave expected log-wealth objective with 15 bps round-trip friction,
dynamic volatility barriers, macro regime scaling kappa in [0.25, 0.50],
single-stock ceilings (10% soft, 15% hard cap), and cross-sectional L1 portfolio normalization.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import minimize_scalar


@dataclass
class ExecutionRecommendation:
    action: str
    conviction_tier: str
    unconstrained_kelly_pct: float
    regime_scaled_kelly_pct: float
    l1_normalized_portfolio_allocation_pct: float
    entry_price: float
    dynamic_take_profit: float
    dynamic_stop_loss: float
    reward_to_risk_ratio: float
    estimated_round_trip_cost_bps: float = 15.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "conviction_tier": self.conviction_tier,
            "unconstrained_kelly_pct": round(self.unconstrained_kelly_pct, 2),
            "l1_normalized_portfolio_allocation_pct": round(
                self.l1_normalized_portfolio_allocation_pct, 2
            ),
            "entry_price": round(self.entry_price, 2),
            "dynamic_take_profit": round(self.dynamic_take_profit, 2),
            "dynamic_stop_loss": round(self.dynamic_stop_loss, 2),
            "reward_to_risk_ratio": round(self.reward_to_risk_ratio, 2),
            "estimated_round_trip_cost_bps": self.estimated_round_trip_cost_bps,
        }


class MultiOutcomeKellySizer:
    """
    Computes optimal position sizing via Multi-Outcome Fractional Kelly Criterion:
    max_{f in [0, f_max]} E[ln(1 + f * R_net)] =
        p_buy * ln(1 + f * (b_upper - c)) +
        p_sell * ln(1 - f * (b_lower + c)) +
        p_hold * ln(1 + f * (b_vert - c))
    Scaled by macro regime multiplier kappa in [0.25, 0.50] (Quarter/Half Kelly)
    subject to single-stock soft ceiling f_max = 0.10 and absolute hard cap = 0.15.
    """

    DEFAULT_FRICTION: float = 0.0015  # 15 bps round-trip
    SOFT_CEILING: float = 0.10        # 10% soft single-stock ceiling
    HARD_CAP: float = 0.15            # 15% absolute hard cap

    def __init__(
        self,
        round_trip_cost: float = DEFAULT_FRICTION,
        soft_ceiling: float = SOFT_CEILING,
        hard_cap: float = HARD_CAP,
    ) -> None:
        self.round_trip_cost = round_trip_cost
        self.soft_ceiling = soft_ceiling
        self.hard_cap = hard_cap

    def solve_unconstrained_kelly(
        self,
        p_buy: float,
        p_hold: float,
        p_sell: float,
        b_upper: float,
        b_lower: float,
        b_vert: float = 0.0,
    ) -> float:
        """
        Solves for the optimal unconstrained fraction f* in [0, soft_ceiling].
        Returns 0.0 if expected net edge <= 0.
        """
        c = self.round_trip_cost
        r_buy_net = b_upper - c
        r_sell_net = -(b_lower + c)  # Loss magnitude is (b_lower + c)
        r_hold_net = b_vert - c

        # First derivative at f = 0: E[R_net]
        dG_0 = p_buy * r_buy_net + p_sell * r_sell_net + p_hold * r_hold_net
        if dG_0 <= 0.0:
            return 0.0

        # Maximum feasible f before total loss: f < 1 / (b_lower + c)
        loss_barrier = b_lower + c
        upper_limit = (
            min(self.soft_ceiling, 0.99 / loss_barrier)
            if loss_barrier > 0
            else self.soft_ceiling
        )

        def neg_expected_log_wealth(f: float) -> float:
            if f <= 0.0:
                return 0.0
            term_buy = 1.0 + f * r_buy_net
            term_sell = 1.0 + f * r_sell_net
            term_hold = 1.0 + f * r_hold_net

            if term_buy <= 1e-12 or term_sell <= 1e-12 or term_hold <= 1e-12:
                return 1e9

            g = (
                p_buy * np.log(term_buy)
                + p_sell * np.log(term_sell)
                + p_hold * np.log(term_hold)
            )
            return float(-g)

        res = minimize_scalar(
            neg_expected_log_wealth,
            bounds=(0.0, upper_limit),
            method="bounded",
        )

        f_opt = float(np.clip(res.x, 0.0, self.soft_ceiling))
        return f_opt

    def compute_recommendation(
        self,
        p_buy: float,
        p_hold: float,
        p_sell: float,
        b_upper: float,
        b_lower: float,
        entry_price: float,
        kappa: float = 0.50,
        b_vert: float = 0.0,
        l1_normalized_pct: Optional[float] = None,
    ) -> ExecutionRecommendation:
        """
        Generates full execution recommendation matching QARE-MM v1.3 Target Output Contract.
        """
        # Ensure probabilities sum to 1
        total_p = p_buy + p_hold + p_sell
        if total_p > 0:
            p_buy /= total_p
            p_hold /= total_p
            p_sell /= total_p

        f_star = self.solve_unconstrained_kelly(
            p_buy=p_buy,
            p_hold=p_hold,
            p_sell=p_sell,
            b_upper=b_upper,
            b_lower=b_lower,
            b_vert=b_vert,
        )

        # Macro regime scaling: f_regime = min(f* * kappa, HARD_CAP)
        kappa_clamped = float(np.clip(kappa, 0.25, 0.50))
        f_regime = float(np.clip(f_star * kappa_clamped, 0.0, self.hard_cap))

        # Conviction tier
        if p_buy >= 0.60 or f_star >= 0.06:
            conviction = "HIGH"
        elif p_buy >= 0.45 or f_star >= 0.02:
            conviction = "MEDIUM"
        else:
            conviction = "LOW"

        # Action classification
        if f_star > 0.005 and p_buy > p_sell:
            action = "BUY"
        elif p_sell > p_buy and p_sell > 0.45:
            action = "SELL"
        else:
            action = "HOLD"

        p_0 = float(entry_price)
        dynamic_tp = p_0 * (1.0 + b_upper)
        dynamic_sl = p_0 * (1.0 - b_lower)
        r_to_r = float(b_upper / (b_lower + 1e-12))

        norm_pct = (
            l1_normalized_pct
            if l1_normalized_pct is not None
            else f_regime * 100.0
        )

        return ExecutionRecommendation(
            action=action,
            conviction_tier=conviction,
            unconstrained_kelly_pct=f_star * 100.0,
            regime_scaled_kelly_pct=f_regime * 100.0,
            l1_normalized_portfolio_allocation_pct=norm_pct,
            entry_price=p_0,
            dynamic_take_profit=dynamic_tp,
            dynamic_stop_loss=dynamic_sl,
            reward_to_risk_ratio=r_to_r,
            estimated_round_trip_cost_bps=self.round_trip_cost * 10000.0,
        )


class PortfolioExposureNormalizer:
    """
    Portfolio-Level L1 Gross Exposure Normalization:
    w_i = (f_i* * kappa_i) / max(1.0, sum_{j=1}^N |f_j* * kappa_j|)
    Guarantees sum_{i=1}^N |w_i| <= 1.0 (100% gross equity exposure),
    eliminating leverage blowups while preserving relative conviction weights.
    """

    @staticmethod
    def normalize_weights(
        raw_allocations: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Normalizes single-stock fractional Kelly allocations across the portfolio.
        Inputs: Dict of {ticker: f_i * kappa_i}
        Returns: Dict of {ticker: w_i} where sum |w_i| <= 1.0
        """
        if not raw_allocations:
            return {}

        total_gross_exposure = float(sum(abs(w) for w in raw_allocations.values()))
        normalizer = max(1.0, total_gross_exposure)

        return {
            ticker: float(w / normalizer)
            for ticker, w in raw_allocations.items()
        }

    @staticmethod
    def normalize_array(raw_weights: np.ndarray) -> np.ndarray:
        """
        Vectorized normalization for an array of asset weights.
        """
        w = np.asarray(raw_weights, dtype=np.float64)
        total_gross = np.sum(np.abs(w))
        normalizer = max(1.0, total_gross)
        return w / normalizer
