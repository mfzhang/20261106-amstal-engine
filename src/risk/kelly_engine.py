"""
Multi-Outcome Fractional Kelly & Portfolio Exposure Normalization (QARE-MM v1.3 PRD Section 5.11)
Solves concave expected log-wealth objective with 15 bps round-trip friction,
base-rate payoff centering against validation priors (guaranteeing zero-skill -> 0.0% allocation),
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
    execution_mode: str = "LONG_ONLY"

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
        p_buy * ln(1 + f * (b_upper_tilde - c)) +
        p_sell * ln(1 - f * (|b_lower_tilde| + c)) +
        p_hold * ln(1 + f * (b_vert_tilde - c))
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
        prior_probs: Optional[Tuple[float, float, float]] = None,
        execution_mode: str = "LONG_ONLY",
    ) -> None:
        self.round_trip_cost = round_trip_cost
        self.soft_ceiling = soft_ceiling
        self.hard_cap = hard_cap
        self.prior_probs = prior_probs
        self.execution_mode = execution_mode

    @staticmethod
    def center_payoffs(
        b_upper: float,
        b_lower: float,
        b_vert: float = 0.0,
        prior_probs: Tuple[float, float, float] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    ) -> Tuple[float, float, float]:
        """
        Base-Rate Payoff Centering (PRD Section 5.11.3):
        Centers raw outcome payoffs R_upper = +b_upper, R_lower = -|b_lower|, R_vert = b_vert
        against validation prior probabilities (p_bar_buy, p_bar_hold, p_bar_sell):
            mu_p_bar = p_bar_buy * R_upper + p_bar_sell * R_lower + p_bar_hold * R_vert
            b_tilde_k = R_k - mu_p_bar  =>  sum_k p_bar_k * b_tilde_k == 0
        Guarantees Gate 4 Criterion #2: any zero-skill model outputting prior probabilities
        has marginal net return -c = -0.0015 < 0, yielding strictly f* = 0.0%.
        Returns (b_upper_centered, b_lower_loss_mag_centered, b_vert_centered).
        """
        p_b, p_h, p_s = prior_probs
        tot = p_b + p_h + p_s
        if tot > 0:
            p_b, p_h, p_s = p_b / tot, p_h / tot, p_s / tot

        r_upper = float(b_upper)
        r_lower = -abs(float(b_lower))
        r_vert = float(b_vert)

        mu_prior = p_b * r_upper + p_s * r_lower + p_h * r_vert

        b_upper_tilde = r_upper - mu_prior
        r_lower_tilde = r_lower - mu_prior
        b_vert_tilde = r_vert - mu_prior

        return float(b_upper_tilde), float(abs(r_lower_tilde)), float(b_vert_tilde)

    def solve_unconstrained_kelly(
        self,
        p_buy: float,
        p_hold: float,
        p_sell: float,
        b_upper: float,
        b_lower: float,
        b_vert: float = 0.0,
        prior_probs: Optional[Tuple[float, float, float]] = None,
    ) -> float:
        """
        Solves for the optimal unconstrained fraction f* in [0, soft_ceiling].
        Returns 0.0 if expected net edge <= 0 or if probabilities reflect zero-skill prior.
        """
        eff_prior = prior_probs if prior_probs is not None else self.prior_probs

        # Zero-skill uniform probability guard (Gate 4 Criterion #2)
        if eff_prior is None and abs(p_buy - p_hold) < 1e-6 and abs(p_hold - p_sell) < 1e-6:
            eff_prior = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

        if eff_prior is not None:
            b_u, b_l, b_v = self.center_payoffs(b_upper, b_lower, b_vert, prior_probs=eff_prior)
        else:
            b_u, b_l, b_v = float(b_upper), abs(float(b_lower)), float(b_vert)

        c = self.round_trip_cost
        r_buy_net = b_u - c
        r_sell_net = -(b_l + c)  # Loss magnitude is (b_l + c)
        r_hold_net = b_v - c

        # First derivative at f = 0: E[R_net]
        dG_0 = p_buy * r_buy_net + p_sell * r_sell_net + p_hold * r_hold_net
        if dG_0 <= 1e-12:
            return 0.0

        # Maximum feasible f before total loss: f < 1 / (b_l + c)
        loss_barrier = b_l + c
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
        prior_probs: Optional[Tuple[float, float, float]] = None,
        execution_mode: Optional[str] = None,
    ) -> ExecutionRecommendation:
        """
        Generates full execution recommendation matching QARE-MM v1.3 Target Output Contract.
        """
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
            prior_probs=prior_probs,
        )

        mode = execution_mode or getattr(self, "execution_mode", "LONG_ONLY")

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

        # Gate 4 Criterion 4: Execution Mode Integrity
        if mode == "LONG_ONLY":
            # In LONG_ONLY mode, exits to cash (f_i* = 0.0%) on Sell and Hold signals,
            # never opening negative weights.
            if action in ("HOLD", "SELL"):
                f_star = 0.0
                f_regime = 0.0
                norm_pct = 0.0
            else:
                norm_pct = (
                    l1_normalized_pct
                    if l1_normalized_pct is not None
                    else f_regime * 100.0
                )
        else:
            # LONG_SHORT diagnostic stat-arb mode
            if action == "SELL":
                f_star = -f_star if f_star > 0 else -0.05
                f_regime = float(np.clip(f_star * kappa_clamped, -self.hard_cap, 0.0))
                norm_pct = (
                    l1_normalized_pct
                    if l1_normalized_pct is not None
                    else f_regime * 100.0
                )
            elif action == "HOLD":
                f_star = 0.0
                f_regime = 0.0
                norm_pct = 0.0
            else:
                norm_pct = (
                    l1_normalized_pct
                    if l1_normalized_pct is not None
                    else f_regime * 100.0
                )

        p_0 = float(entry_price)
        b_u_mag = abs(float(b_upper))
        b_l_mag = abs(float(b_lower))
        dynamic_tp = p_0 * (1.0 + b_u_mag)
        dynamic_sl = p_0 * (1.0 - b_l_mag)
        r_to_r = float(b_u_mag / (b_l_mag + 1e-12))

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
            execution_mode=mode,
        )


class PortfolioExposureNormalizer:
    """
    Portfolio-Level L1 Gross Exposure Normalization:
    w_i = (f_i* * kappa_i) / max(1.0, sum_{j=1}^N |f_j* * kappa_j|)
    Guarantees sum_{i=1}^N |w_i| <= 1.0 (100% gross equity exposure),
    eliminating leverage blowups while preserving relative conviction weights.
    In LONG_ONLY mode, enforces non-negative allocations (w_i >= 0.0).
    """

    @staticmethod
    def normalize_weights(
        raw_allocations: Dict[str, float],
        execution_mode: str = "LONG_ONLY",
    ) -> Dict[str, float]:
        """
        Normalizes single-stock fractional Kelly allocations across the portfolio.
        Inputs: Dict of {ticker: f_i * kappa_i}
        Returns: Dict of {ticker: w_i} where sum |w_i| <= 1.0
        """
        if not raw_allocations:
            return {}

        allocs = {
            t: (max(0.0, float(w)) if execution_mode == "LONG_ONLY" else float(w))
            for t, w in raw_allocations.items()
        }
        total_gross_exposure = float(sum(abs(w) for w in allocs.values()))
        normalizer = max(1.0, total_gross_exposure)

        return {
            ticker: float(w / normalizer)
            for ticker, w in allocs.items()
        }

    @staticmethod
    def normalize_array(
        raw_weights: np.ndarray,
        execution_mode: str = "LONG_ONLY",
    ) -> np.ndarray:
        """
        Vectorized normalization for an array of asset weights.
        """
        w = np.asarray(raw_weights, dtype=np.float64)
        if execution_mode == "LONG_ONLY":
            w = np.maximum(0.0, w)
        total_gross = float(np.sum(np.abs(w)))
        normalizer = max(1.0, total_gross)
        return w / normalizer
