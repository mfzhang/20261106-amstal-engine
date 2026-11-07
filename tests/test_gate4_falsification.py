"""
Gate 4 Quantitative Falsification Battery (QARE-MM v1.3 PRD Section 3.4.4 & Section 5.11)
Verifies:
1. Criterion 1: Post-hoc temperature calibration drops validation Brier score by >10% over uncalibrated raw softmax:
   (Brier_uncal - Brier_cal) / Brier_uncal > 0.10
2. Criterion 2: Zero-Skill Allocation Guard: Any uninformative model predicting prior probabilities p_hat = p_bar
   produces strictly f_i* = 0.0% allocation due to base-rate payoff centering b_{i,k} = b_tilde_{i,k} / p_bar_k
3. Criterion 3: Portfolio L1 Gross Exposure Bound: Concurrent buy signals across all universe assets
   never exceed 100% gross portfolio leverage: sum |w_i| <= 1.0 under L1 normalization
4. Criterion 4: Execution Mode Integrity: In LONG_ONLY mode, exits to cash (f_i* = 0.0%) on Sell and Hold signals,
   never opening negative weights (w_i >= 0.0 for all assets)
"""

import numpy as np
import pytest

from src.backtest.metrics import BacktestMetrics
from src.risk.calibration import TemperatureScaler
from src.risk.kelly_engine import (
    ExecutionRecommendation,
    MultiOutcomeKellySizer,
    PortfolioExposureNormalizer,
)


def test_gate4_criterion1_temperature_calibration_brier_drop() -> None:
    """
    GATE 4 CRITERION 1:
    Post-hoc temperature calibration drops validation Brier score by >10%
    over uncalibrated raw softmax:
        (Brier_uncal - Brier_cal) / Brier_uncal > 0.10
    """
    rng = np.random.default_rng(42)
    N = 600  # Validation sample size (spanning purged validation fold)

    # Ground truth class labels with realistic empirical frequencies:
    # Class 0: Outperform/Buy (30%), Class 1: Neutral/Hold (45%), Class 2: Underperform/Sell (25%)
    p_true = np.array([0.30, 0.45, 0.25])
    y_val = rng.choice([0, 1, 2], size=N, p=p_true)

    # Modern deep networks and gradient boosted models trained on cross-entropy produce
    # overconfident raw logits with inflated variance (logits ~ N(0, 3.2^2)).
    logits = rng.normal(0, 3.2, size=(N, 3))

    # Inject modest predictive signal into the true class
    for i in range(N):
        logits[i, y_val[i]] += rng.normal(0.8, 0.4)

    # Raw uncalibrated softmax probabilities
    raw_softmax = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

    # Learn post-hoc temperature scaler on the validation fold
    scaler = TemperatureScaler(bounds=(0.05, 5.0))
    fitted_temp = scaler.fit(raw_softmax, y_val)

    # Temperature parameter must indicate overconfidence (T > 1.0)
    assert fitted_temp > 1.0, f"Expected T > 1.0 for overconfident logits, got {fitted_temp:.4f}"

    # Generate calibrated probabilities
    calibrated_probs = scaler.transform(raw_softmax)

    # Compute Brier scores
    brier_uncal = BacktestMetrics.brier_score(raw_softmax, y_val)
    brier_cal = BacktestMetrics.brier_score(calibrated_probs, y_val)

    # Relative Brier improvement
    brier_drop = (brier_uncal - brier_cal) / brier_uncal

    # Verification of Gate 4 Criterion 1 Hurdle (> 10% reduction)
    assert brier_uncal > brier_cal, "Calibration failed to reduce Brier score"
    assert brier_drop > 0.10, (
        f"Gate 4 Criterion 1 violated: Brier drop {brier_drop * 100:.2f}% "
        f"did not exceed the required 10.00% hurdle (Uncal: {brier_uncal:.4f}, Cal: {brier_cal:.4f})"
    )


def test_gate4_criterion2_zero_skill_allocation_guard() -> None:
    """
    GATE 4 CRITERION 2:
    Zero-Skill Allocation Guard: Any uninformative model predicting prior probabilities
    p_hat = p_bar produces strictly f_i* = 0.0% allocation due to base-rate payoff centering:
        mu_p_bar = sum_k p_bar_k * R_k
        b_tilde_k = R_k - mu_p_bar  ==>  sum_k p_bar_k * b_tilde_k == 0
        dE[ln(1 + f R_net)] / df |_{f=0} = -c = -0.0015 < 0  ==>  f_i* = 0.0%
    """
    sizer = MultiOutcomeKellySizer(round_trip_cost=0.0015, soft_ceiling=0.10, hard_cap=0.15)

    # Test across multiple prior distributions and barrier geometries
    priors_to_test = [
        (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),  # Uniform prior
        (0.28, 0.48, 0.24),                  # Empirical triple-barrier validation prior
        (0.35, 0.45, 0.20),                  # Bullish regime validation prior
        (0.20, 0.35, 0.45),                  # Bearish regime validation prior
    ]

    barrier_configs = [
        # (b_upper, b_lower, b_vert): asymmetric +2 sigma / -1 sigma barriers
        (0.08, 0.04, 0.002),
        (0.12, 0.04, -0.001),
        (0.06, 0.03, 0.000),
        (0.15, 0.05, 0.005),
    ]

    for prior in priors_to_test:
        p_b, p_h, p_s = prior

        for b_upper, b_lower, b_vert in barrier_configs:
            # 1. Base-Rate Centering Identity: sum_k p_bar_k * b_tilde_k must equal 0.0
            b_u_c, b_l_c, b_v_c = MultiOutcomeKellySizer.center_payoffs(
                b_upper=b_upper,
                b_lower=b_lower,
                b_vert=b_vert,
                prior_probs=prior,
            )
            centered_expected_return = p_b * b_u_c - p_s * b_l_c + p_h * b_v_c
            assert centered_expected_return == pytest.approx(0.0, abs=1e-12), (
                f"Base-rate centering failure: expected centered return is {centered_expected_return} != 0.0"
            )

            # 2. Optimal Kelly Allocation under prior probabilities p_hat = p_bar must be strictly 0.0%
            f_star = sizer.solve_unconstrained_kelly(
                p_buy=p_b,
                p_hold=p_h,
                p_sell=p_s,
                b_upper=b_upper,
                b_lower=b_lower,
                b_vert=b_vert,
                prior_probs=prior,
            )
            assert f_star == 0.0, (
                f"Gate 4 Criterion 2 violated: uninformative model predicting prior {prior} "
                f"allocated f* = {f_star * 100:.2f}% > 0.0% under barriers (+{b_upper}, -{b_lower})"
            )

            # 3. Execution Recommendation check
            rec = sizer.compute_recommendation(
                p_buy=p_b,
                p_hold=p_h,
                p_sell=p_s,
                b_upper=b_upper,
                b_lower=b_lower,
                entry_price=150.0,
                kappa=0.50,
                b_vert=b_vert,
                prior_probs=prior,
            )
            assert rec.unconstrained_kelly_pct == 0.0, "Unconstrained Kelly must be 0.0% on prior"
            assert rec.regime_scaled_kelly_pct == 0.0, "Regime scaled Kelly must be 0.0% on prior"
            assert rec.l1_normalized_portfolio_allocation_pct == 0.0, "Portfolio allocation must be 0.0% on prior"

    # Contrast with an informative model that holds genuine predictive edge:
    # High conviction Buy: p_buy = 0.65, p_hold = 0.25, p_sell = 0.10
    prior_emp = (0.28, 0.48, 0.24)
    f_informative = sizer.solve_unconstrained_kelly(
        p_buy=0.65,
        p_hold=0.25,
        p_sell=0.10,
        b_upper=0.08,
        b_lower=0.04,
        b_vert=0.002,
        prior_probs=prior_emp,
    )
    assert f_informative > 0.0, "Informative model must produce positive allocation"
    assert f_informative <= 0.10, "Informative model must obey soft single-stock ceiling (10%)"


def test_gate4_criterion3_portfolio_l1_exposure_cap() -> None:
    """
    GATE 4 CRITERION 3:
    Portfolio L1 Gross Exposure Bound:
    Concurrent buy signals across all universe assets never exceed 100% gross portfolio leverage:
        sum_{i=1}^N |w_i| <= 1.0
    under L1 normalization:
        w_i = (f_i* * kappa_i) / max(1.0, sum_{j=1}^N |f_j* * kappa_j|)
    """
    normalizer = PortfolioExposureNormalizer()

    # Scenario 1: Extreme Bull Market Cluster (250 assets in S&P universe all triggering BUY)
    # Each asset has optimal single-stock allocation f_i* * kappa_i ~ 0.03 to 0.05
    N = 250
    rng = np.random.default_rng(42)
    raw_allocations = {
        f"TICKER_{i:03d}": float(rng.uniform(0.03, 0.05))
        for i in range(N)
    }

    raw_gross_exposure = sum(raw_allocations.values())
    assert raw_gross_exposure > 5.0, f"Expected extreme raw leverage > 500%, got {raw_gross_exposure * 100:.1f}%"

    normalized_weights = normalizer.normalize_weights(raw_allocations, execution_mode="LONG_ONLY")

    # The sum of absolute weights must never exceed 100% (1.0)
    total_norm_gross = sum(abs(w) for w in normalized_weights.values())
    assert total_norm_gross <= 1.0 + 1e-12, (
        f"Gate 4 Criterion 3 violated: normalized gross leverage {total_norm_gross * 100:.4f}% > 100%"
    )
    assert total_norm_gross == pytest.approx(1.0, abs=1e-7), (
        f"Over-leveraged portfolio must normalize exactly to 100.0%, got {total_norm_gross * 100:.4f}%"
    )

    # Relative conviction weights must be strictly preserved
    keys = list(raw_allocations.keys())
    for i in range(0, len(keys) - 1, 15):
        t1, t2 = keys[i], keys[i + 1]
        ratio_raw = raw_allocations[t1] / raw_allocations[t2]
        ratio_norm = normalized_weights[t1] / normalized_weights[t2]
        assert ratio_norm == pytest.approx(ratio_raw, rel=1e-5), (
            f"Conviction ratio not preserved: raw={ratio_raw}, norm={ratio_norm}"
        )

    # Scenario 2: Sparse / Under-Leveraged Portfolio (Only 4 assets trigger BUY)
    # Total raw exposure = 0.04 + 0.05 + 0.03 + 0.06 = 0.18 <= 1.0
    sparse_allocations = {
        "AAPL": 0.04,
        "NVDA": 0.05,
        "MSFT": 0.03,
        "CRWD": 0.06,
    }
    sparse_norm = normalizer.normalize_weights(sparse_allocations, execution_mode="LONG_ONLY")
    sparse_gross = sum(abs(w) for w in sparse_norm.values())

    # When total raw exposure is under 1.0, normalization factor max(1.0, gross) = 1.0
    # Weights must NOT be artificially pumped up to 100%
    assert sparse_gross == pytest.approx(0.18, rel=1e-5)
    assert sparse_gross <= 1.0
    for ticker, raw_w in sparse_allocations.items():
        assert sparse_norm[ticker] == pytest.approx(raw_w, rel=1e-5)

    # Scenario 3: Vectorized Array Normalization across 250 assets
    raw_array = rng.uniform(0.01, 0.06, size=N)
    norm_array = normalizer.normalize_array(raw_array, execution_mode="LONG_ONLY")
    assert np.all(norm_array >= 0.0), "All weights must be non-negative in LONG_ONLY"
    assert np.sum(np.abs(norm_array)) <= 1.0 + 1e-12
    assert np.sum(np.abs(norm_array)) == pytest.approx(1.0, abs=1e-7)


def test_gate4_criterion4_execution_mode_integrity() -> None:
    """
    GATE 4 CRITERION 4:
    Execution Mode Integrity:
    In LONG_ONLY mode, exits to cash (f_i* = 0.0%) on Sell and Hold signals,
    never opening negative weights (w_i >= 0.0 for all assets).
    """
    sizer = MultiOutcomeKellySizer(
        round_trip_cost=0.0015,
        soft_ceiling=0.10,
        hard_cap=0.15,
        execution_mode="LONG_ONLY",
    )

    # 1. Clear SELL Signal (Underperform hit dominant: p_sell = 0.65, p_hold = 0.25, p_buy = 0.10)
    rec_sell = sizer.compute_recommendation(
        p_buy=0.10,
        p_hold=0.25,
        p_sell=0.65,
        b_upper=0.08,
        b_lower=0.04,
        entry_price=200.0,
        kappa=0.50,
        execution_mode="LONG_ONLY",
    )
    assert rec_sell.action == "SELL"
    assert rec_sell.unconstrained_kelly_pct == 0.0, "SELL in LONG_ONLY must exit to cash (0.0%)"
    assert rec_sell.regime_scaled_kelly_pct == 0.0, "SELL in LONG_ONLY must have 0.0% regime allocation"
    assert rec_sell.l1_normalized_portfolio_allocation_pct == 0.0, "SELL in LONG_ONLY must have 0.0% weight"

    # 2. Clear HOLD Signal (Neutral dominant: p_hold = 0.60, p_buy = 0.20, p_sell = 0.20)
    rec_hold = sizer.compute_recommendation(
        p_buy=0.20,
        p_hold=0.60,
        p_sell=0.20,
        b_upper=0.08,
        b_lower=0.04,
        entry_price=100.0,
        kappa=0.50,
        execution_mode="LONG_ONLY",
    )
    assert rec_hold.action == "HOLD"
    assert rec_hold.unconstrained_kelly_pct == 0.0, "HOLD in LONG_ONLY must exit to cash (0.0%)"
    assert rec_hold.regime_scaled_kelly_pct == 0.0, "HOLD in LONG_ONLY must have 0.0% regime allocation"
    assert rec_hold.l1_normalized_portfolio_allocation_pct == 0.0, "HOLD in LONG_ONLY must have 0.0% weight"

    # 3. Clear BUY Signal (Outperform dominant: p_buy = 0.65, p_hold = 0.25, p_sell = 0.10)
    rec_buy = sizer.compute_recommendation(
        p_buy=0.65,
        p_hold=0.25,
        p_sell=0.10,
        b_upper=0.08,
        b_lower=0.04,
        entry_price=150.0,
        kappa=0.50,
        execution_mode="LONG_ONLY",
    )
    assert rec_buy.action == "BUY"
    assert rec_buy.unconstrained_kelly_pct > 0.0, "BUY must produce positive unconstrained Kelly fraction"
    assert rec_buy.regime_scaled_kelly_pct > 0.0, "BUY must produce positive regime scaled allocation"
    assert rec_buy.l1_normalized_portfolio_allocation_pct > 0.0, "BUY must produce positive portfolio weight"

    # 4. Panel Test across 250 simulated universe assets in LONG_ONLY mode
    rng = np.random.default_rng(42)
    N = 250
    panel_recommendations = []

    for i in range(N):
        p_raw = rng.dirichlet([1.0, 1.5, 1.0])
        rec = sizer.compute_recommendation(
            p_buy=float(p_raw[0]),
            p_hold=float(p_raw[1]),
            p_sell=float(p_raw[2]),
            b_upper=float(rng.uniform(0.05, 0.10)),
            b_lower=float(rng.uniform(0.03, 0.05)),
            entry_price=100.0,
            kappa=float(rng.uniform(0.25, 0.50)),
            execution_mode="LONG_ONLY",
        )
        panel_recommendations.append(rec)

    for rec in panel_recommendations:
        # Non-negative weight guarantee
        assert rec.unconstrained_kelly_pct >= 0.0, "Negative unconstrained Kelly prohibited in LONG_ONLY"
        assert rec.regime_scaled_kelly_pct >= 0.0, "Negative regime Kelly prohibited in LONG_ONLY"
        assert rec.l1_normalized_portfolio_allocation_pct >= 0.0, "Negative portfolio weight prohibited in LONG_ONLY"

        # Action integrity: Sell and Hold must strictly exit to cash
        if rec.action in ("SELL", "HOLD"):
            assert rec.unconstrained_kelly_pct == 0.0, f"Non-zero allocation on {rec.action} action"
            assert rec.regime_scaled_kelly_pct == 0.0, f"Non-zero regime allocation on {rec.action} action"
            assert rec.l1_normalized_portfolio_allocation_pct == 0.0, f"Non-zero portfolio weight on {rec.action} action"

    # 5. Diagnostic Stat-Arb Mode Separation (LONG_SHORT mode)
    # In LONG_SHORT mode, SELL signals open short positions (negative weights)
    rec_short = sizer.compute_recommendation(
        p_buy=0.10,
        p_hold=0.20,
        p_sell=0.70,
        b_upper=0.08,
        b_lower=0.04,
        entry_price=200.0,
        kappa=0.50,
        execution_mode="LONG_SHORT",
    )
    assert rec_short.action == "SELL"
    assert rec_short.regime_scaled_kelly_pct < 0.0, "LONG_SHORT mode must allow negative allocation on SELL"
    assert rec_short.l1_normalized_portfolio_allocation_pct < 0.0, "LONG_SHORT mode must allow negative weight on SELL"
