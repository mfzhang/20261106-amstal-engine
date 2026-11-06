"""
Unit Tests for Risk Management, Calibration & Portfolio Exposure Engine (QARE-MM v1.3)
Verifies:
1. Post-hoc Temperature Scaling calibration and NLL optimization
2. Multi-Outcome Fractional Kelly concavity and edge detection
3. Impact of 15 bps transaction friction on optimal position sizing
4. Macro regime scaling multiplier kappa in [0.25, 0.50] and ceilings
5. Portfolio-level L1 gross exposure normalization (sum |w_i| <= 1.0)
"""

import numpy as np
import pytest

from src.risk.calibration import TemperatureScaler
from src.risk.kelly_engine import (
    ExecutionRecommendation,
    MultiOutcomeKellySizer,
    PortfolioExposureNormalizer,
)


def test_temperature_scaler(rng) -> None:
    N = 300
    labels = rng.choice([0, 1, 2], size=N)

    # Overconfident uncalibrated logits
    logits = rng.normal(0, 3.0, size=(N, 3))
    raw_probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

    scaler = TemperatureScaler()
    temp = scaler.fit(raw_probs, labels)

    assert 0.05 <= temp <= 5.0
    assert scaler.is_fitted_

    cal_probs = scaler.transform(raw_probs)
    assert cal_probs.shape == (N, 3)
    assert np.all(cal_probs >= 0.0)
    assert np.all(cal_probs <= 1.0)
    assert np.allclose(cal_probs.sum(axis=1), 1.0, atol=1e-5)


def test_multi_outcome_kelly_positive_edge() -> None:
    sizer = MultiOutcomeKellySizer(round_trip_cost=0.0015, soft_ceiling=0.10, hard_cap=0.15)

    # Strong buy conviction: 65% buy, 25% hold, 10% sell
    p_buy, p_hold, p_sell = 0.65, 0.25, 0.10
    b_upper = 0.07   # +7.0% ATR take-profit
    b_lower = 0.035  # -3.5% ATR stop-loss
    b_vert = 0.005   # +0.5% expected drift at t+10

    f_star = sizer.solve_unconstrained_kelly(
        p_buy=p_buy, p_hold=p_hold, p_sell=p_sell,
        b_upper=b_upper, b_lower=b_lower, b_vert=b_vert
    )

    assert f_star > 0.0
    assert f_star <= 0.10  # Must obey soft single-stock ceiling

    # Check macro regime scaling kappa = 0.40
    rec = sizer.compute_recommendation(
        p_buy=p_buy, p_hold=p_hold, p_sell=p_sell,
        b_upper=b_upper, b_lower=b_lower,
        entry_price=312.40,
        kappa=0.40,
        b_vert=b_vert,
    )

    assert rec.action == "BUY"
    assert rec.conviction_tier == "HIGH"
    assert rec.unconstrained_kelly_pct == pytest.approx(f_star * 100.0, rel=1e-3)
    assert rec.regime_scaled_kelly_pct == pytest.approx(f_star * 0.40 * 100.0, rel=1e-3)
    assert rec.dynamic_take_profit == pytest.approx(312.40 * 1.07, rel=1e-3)
    assert rec.dynamic_stop_loss == pytest.approx(312.40 * (1.0 - 0.035), rel=1e-3)
    assert rec.reward_to_risk_ratio == pytest.approx(2.0, rel=1e-3)
    assert rec.estimated_round_trip_cost_bps == 15.0


def test_multi_outcome_kelly_negative_or_zero_edge() -> None:
    sizer = MultiOutcomeKellySizer(round_trip_cost=0.0015)

    # Favorable to sell: 15% buy, 25% hold, 60% sell
    p_buy, p_hold, p_sell = 0.15, 0.25, 0.60
    b_upper, b_lower = 0.06, 0.04

    f_star = sizer.solve_unconstrained_kelly(
        p_buy=p_buy, p_hold=p_hold, p_sell=p_sell,
        b_upper=b_upper, b_lower=b_lower
    )

    # For long allocation, negative edge must yield exactly 0.0
    assert f_star == 0.0

    rec = sizer.compute_recommendation(
        p_buy=p_buy, p_hold=p_hold, p_sell=p_sell,
        b_upper=b_upper, b_lower=b_lower,
        entry_price=100.0,
        kappa=0.50,
    )
    assert rec.action == "SELL"
    assert rec.unconstrained_kelly_pct == 0.0
    assert rec.regime_scaled_kelly_pct == 0.0


def test_multi_outcome_kelly_friction_penalty() -> None:
    p_buy, p_hold, p_sell = 0.40, 0.20, 0.40
    b_upper, b_lower = 0.08, 0.075

    # Zero friction vs 10 bps vs 20 bps
    s_zero = MultiOutcomeKellySizer(round_trip_cost=0.0, soft_ceiling=1.0)
    s_10bps = MultiOutcomeKellySizer(round_trip_cost=0.0010, soft_ceiling=1.0)
    s_20bps = MultiOutcomeKellySizer(round_trip_cost=0.0020, soft_ceiling=1.0)

    f_zero = s_zero.solve_unconstrained_kelly(p_buy, p_hold, p_sell, b_upper, b_lower)
    f_10 = s_10bps.solve_unconstrained_kelly(p_buy, p_hold, p_sell, b_upper, b_lower)
    f_20 = s_20bps.solve_unconstrained_kelly(p_buy, p_hold, p_sell, b_upper, b_lower)

    # Transaction friction must monotonically reduce optimal position sizing
    assert f_zero > f_10 > f_20


def test_portfolio_l1_normalization_under_and_over_leveraged() -> None:
    normalizer = PortfolioExposureNormalizer()

    # Case 1: Total allocations <= 100% -> No scaling needed
    allocations_under = {
        "AAPL": 0.06,
        "NVDA": 0.08,
        "MSFT": 0.05,
        "CRWD": 0.07,
    }
    w_under = normalizer.normalize_weights(allocations_under)
    assert sum(w_under.values()) == pytest.approx(0.26)
    assert sum(abs(v) for v in w_under.values()) <= 1.0

    # Case 2: Concurrent signals across multiple stocks exceed 100% (e.g. 20 stocks at 8% each = 160%)
    allocations_over = {f"STOCK_{i}": 0.08 for i in range(20)}
    total_raw = sum(allocations_over.values())
    assert total_raw == pytest.approx(1.60)

    w_over = normalizer.normalize_weights(allocations_over)
    total_normalized = sum(w_over.values())
    assert total_normalized == pytest.approx(1.0, rel=1e-5)
    assert sum(abs(v) for v in w_over.values()) <= 1.0

    # Relative weights must be strictly preserved
    for ticker, weight in w_over.items():
        assert weight == pytest.approx(0.08 / 1.60, rel=1e-5)


def test_portfolio_l1_normalization_vectorized(rng) -> None:
    # 250-stock universe
    N = 250
    raw_weights = rng.uniform(0.0, 0.05, size=N)  # Mean ~0.025 * 250 = ~6.25 (625% leverage)
    norm_weights = PortfolioExposureNormalizer.normalize_array(raw_weights)

    assert len(norm_weights) == N
    assert np.all(norm_weights >= 0.0)
    assert np.sum(np.abs(norm_weights)) == pytest.approx(1.0, rel=1e-5)


def test_kelly_base_rate_payoff_centering_zero_skill_invariant() -> None:
    """
    Verifies PRD Section 5.11.3 (Gate 4 Criterion #2):
    Base-rate payoff centering ensures that any zero-skill model outputting validation
    prior probabilities (or uniform 1/3, 1/3, 1/3) produces strictly 0.0% allocation,
    even under asymmetric +2 sigma / -1 sigma barriers.
    """
    sizer = MultiOutcomeKellySizer(round_trip_cost=0.0015)
    b_upper = 0.08  # +2 sigma
    b_lower = 0.04  # -1 sigma
    b_vert = 0.002

    # 1. Uniform (1/3, 1/3, 1/3) zero-skill probabilities -> strictly 0.0%
    f_uniform = sizer.solve_unconstrained_kelly(
        p_buy=1.0 / 3.0,
        p_hold=1.0 / 3.0,
        p_sell=1.0 / 3.0,
        b_upper=b_upper,
        b_lower=b_lower,
        b_vert=b_vert,
    )
    assert f_uniform == 0.0

    # 2. Non-uniform validation prior (e.g. 0.30, 0.35, 0.35):
    prior = (0.30, 0.35, 0.35)
    b_u_c, b_l_c, b_v_c = MultiOutcomeKellySizer.center_payoffs(b_upper, b_lower, b_vert, prior_probs=prior)
    # Expected centered return under prior must equal 0.0
    expected_prior_return = prior[0] * b_u_c - prior[2] * b_l_c + prior[1] * b_v_c
    assert expected_prior_return == pytest.approx(0.0, abs=1e-12)

    # Outputting prior probabilities yields strictly 0.0% allocation
    f_prior = sizer.solve_unconstrained_kelly(
        p_buy=prior[0],
        p_hold=prior[1],
        p_sell=prior[2],
        b_upper=b_upper,
        b_lower=b_lower,
        b_vert=b_vert,
        prior_probs=prior,
    )
    assert f_prior == 0.0
