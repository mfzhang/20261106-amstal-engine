"""
Gate 4 Quantitative Falsification Battery CLI Runner (QARE-MM v1.3 PRD Section 3.4.4 & 5.11)
Executes Phase 4 Gate 4 falsification battery:
1. Criterion 1 (Brier Improvement): Post-hoc temperature calibration drops validation Brier score by >10%
2. Criterion 2 (Zero-Skill Allocation Guard): Base-rate payoff centering guarantees f* = 0.0% under prior
3. Criterion 3 (Portfolio L1 Exposure Cap): Concurrent buy signals across universe assets never exceed 100% gross leverage
4. Criterion 4 (Execution Mode Integrity): In LONG_ONLY mode, exits to cash (f* = 0.0%) on Sell and Hold, no negative weights
"""

import math
import sys
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import numpy as np

from src.backtest.metrics import BacktestMetrics
from src.risk.calibration import TemperatureScaler
from src.risk.kelly_engine import (
    MultiOutcomeKellySizer,
    PortfolioExposureNormalizer,
)


def run_battery() -> int:
    print("=" * 80)
    print("AMSTAL QUANT ENGINE: GATE 4 FALSIFICATION BATTERY (QARE-MM v1.3)")
    print("Target: Temperature Calibration, Zero-Skill Kelly Guard, L1 Exposure & Execution Mode")
    print("=" * 80)

    failures = 0
    t_start = time.perf_counter()

    # ---------------------------------------------------------
    # TEST 1: Post-Hoc Temperature Calibration (Brier Drop > 10%)
    # ---------------------------------------------------------
    print("\n[TEST 1/4] Checking Post-Hoc Temperature Scaling Calibration (Brier Drop > 10%)...")
    rng = np.random.default_rng(42)
    N = 600

    p_true = np.array([0.30, 0.45, 0.25])
    y_val = rng.choice([0, 1, 2], size=N, p=p_true)

    logits = rng.normal(0, 3.2, size=(N, 3))
    for i in range(N):
        logits[i, y_val[i]] += rng.normal(0.8, 0.4)

    raw_softmax = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

    scaler = TemperatureScaler(bounds=(0.05, 5.0))
    fitted_temp = scaler.fit(raw_softmax, y_val)
    calibrated_probs = scaler.transform(raw_softmax)

    brier_uncal = BacktestMetrics.brier_score(raw_softmax, y_val)
    brier_cal = BacktestMetrics.brier_score(calibrated_probs, y_val)
    brier_drop = (brier_uncal - brier_cal) / brier_uncal

    t1_pass = (fitted_temp > 1.0) and (brier_drop > 0.10)
    print(f"  > Uncalibrated Brier Score: {brier_uncal:.4f}")
    print(f"  > Calibrated Brier Score:   {brier_cal:.4f}")
    print(f"  > Fitted Temperature (T):   {fitted_temp:.4f}")
    print(f"  > Brier Score Reduction:    {brier_drop * 100:.2f}% (Hurdle: > 10.00%)")
    print(f"  > Status:                   {'[PASS]' if t1_pass else '[FAIL]'}")
    if not t1_pass:
        failures += 1

    # ---------------------------------------------------------
    # TEST 2: Zero-Skill Allocation Guard (Base-Rate Centered f* = 0.0%)
    # ---------------------------------------------------------
    print("\n[TEST 2/4] Checking Zero-Skill Allocation Guard (Base-Rate Centered f* = 0.0%)...")
    sizer = MultiOutcomeKellySizer(round_trip_cost=0.0015, soft_ceiling=0.10, hard_cap=0.15)

    test_priors = [
        ("Uniform (1/3, 1/3, 1/3)", (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)),
        ("Empirical Triple-Barrier", (0.28, 0.48, 0.24)),
        ("Bullish Regime Prior", (0.35, 0.45, 0.20)),
        ("Bearish Regime Prior", (0.20, 0.35, 0.45)),
    ]

    t2_pass = True
    b_upper, b_lower, b_vert = 0.08, 0.04, 0.002  # Asymmetric +2 sigma / -1 sigma barriers

    for label, prior in test_priors:
        p_b, p_h, p_s = prior
        b_u_c, b_l_c, b_v_c = MultiOutcomeKellySizer.center_payoffs(
            b_upper=b_upper, b_lower=b_lower, b_vert=b_vert, prior_probs=prior
        )
        centered_expected = p_b * b_u_c - p_s * b_l_c + p_h * b_v_c
        f_star = sizer.solve_unconstrained_kelly(
            p_buy=p_b, p_hold=p_h, p_sell=p_s,
            b_upper=b_upper, b_lower=b_lower, b_vert=b_vert, prior_probs=prior,
        )
        rec = sizer.compute_recommendation(
            p_buy=p_b, p_hold=p_h, p_sell=p_s,
            b_upper=b_upper, b_lower=b_lower, entry_price=100.0,
            kappa=0.50, b_vert=b_vert, prior_probs=prior,
        )

        sub_ok = (
            abs(centered_expected) < 1e-12
            and f_star == 0.0
            and rec.unconstrained_kelly_pct == 0.0
            and rec.regime_scaled_kelly_pct == 0.0
            and rec.l1_normalized_portfolio_allocation_pct == 0.0
        )
        if not sub_ok:
            t2_pass = False
        print(f"  > Prior: {label:<24} Centered E[R]: {centered_expected:+.1e} | f*: {f_star * 100:.2f}% | Alloc: {rec.l1_normalized_portfolio_allocation_pct:.2f}%")

    # Informative model check
    f_info = sizer.solve_unconstrained_kelly(
        p_buy=0.65, p_hold=0.25, p_sell=0.10,
        b_upper=b_upper, b_lower=b_lower, b_vert=b_vert, prior_probs=(0.28, 0.48, 0.24),
    )
    rec_info = sizer.compute_recommendation(
        p_buy=0.65, p_hold=0.25, p_sell=0.10,
        b_upper=b_upper, b_lower=b_lower, entry_price=100.0,
        kappa=0.50, b_vert=b_vert, prior_probs=(0.28, 0.48, 0.24),
    )
    info_ok = (f_info > 0.0) and (rec_info.action == "BUY") and (rec_info.l1_normalized_portfolio_allocation_pct > 0.0)
    if not info_ok:
        t2_pass = False

    print(f"  > Informative Conviction Check: f*: {f_info * 100:.2f}% (Action: {rec_info.action}, Soft Cap: 10.0%)")
    print(f"  > Status:                   {'[PASS]' if t2_pass else '[FAIL]'}")
    if not t2_pass:
        failures += 1

    # ---------------------------------------------------------
    # TEST 3: Portfolio L1 Gross Exposure Bound (sum |w_i| <= 1.0)
    # ---------------------------------------------------------
    print("\n[TEST 3/4] Checking Portfolio L1 Gross Exposure Bound (sum |w_i| <= 1.0)...")
    normalizer = PortfolioExposureNormalizer()
    N_universe = 250
    rng3 = np.random.default_rng(123)

    # Bull euphoria: all 250 assets trigger BUY signals
    raw_allocs = {
        f"STOCK_{i:03d}": float(rng3.uniform(0.03, 0.06))
        for i in range(N_universe)
    }
    raw_gross = sum(raw_allocs.values())

    norm_weights = normalizer.normalize_weights(raw_allocs, execution_mode="LONG_ONLY")
    norm_gross = sum(abs(w) for w in norm_weights.values())

    # Invariant: gross exposure never exceeds 100%
    gross_ok = (norm_gross <= 1.0 + 1e-12) and math.isclose(norm_gross, 1.0, abs_tol=1e-7)

    # Invariant: relative conviction proportionality
    conviction_ratios_preserved = True
    keys = list(raw_allocs.keys())
    for idx in range(0, len(keys) - 1, 20):
        tA, tB = keys[idx], keys[idx + 1]
        raw_ratio = raw_allocs[tA] / raw_allocs[tB]
        norm_ratio = norm_weights[tA] / norm_weights[tB]
        if not math.isclose(raw_ratio, norm_ratio, rel_tol=1e-5):
            conviction_ratios_preserved = False

    # Invariant: under-leveraged portfolio does not pump leverage
    sparse_allocs = {"AAPL": 0.05, "MSFT": 0.04, "NVDA": 0.06}
    sparse_norm = normalizer.normalize_weights(sparse_allocs, execution_mode="LONG_ONLY")
    sparse_gross = sum(abs(w) for w in sparse_norm.values())
    sparse_ok = math.isclose(sparse_gross, 0.15, abs_tol=1e-6) and (sparse_gross <= 1.0)

    t3_pass = gross_ok and conviction_ratios_preserved and sparse_ok
    print(f"  > Universe Size:            {N_universe} stocks")
    print(f"  > Raw Unnormalized Leverage:{raw_gross * 100:.1f}% ({raw_gross:.2f}x gross exposure)")
    print(f"  > Normalized Gross Leverage:{norm_gross * 100:.2f}% (Cap: <= 100.00%)")
    print(f"  > Conviction Preservation:  {'[PASS]' if conviction_ratios_preserved else '[FAIL]'}")
    print(f"  > Sub-100% Passthrough:     {'[PASS]' if sparse_ok else '[FAIL]'} (Sparse: {sparse_gross * 100:.1f}%)")
    print(f"  > Status:                   {'[PASS]' if t3_pass else '[FAIL]'}")
    if not t3_pass:
        failures += 1

    # ---------------------------------------------------------
    # TEST 4: Execution Mode Integrity (LONG_ONLY Exits to Cash f* = 0.0%)
    # ---------------------------------------------------------
    print("\n[TEST 4/4] Checking Execution Mode Integrity (LONG_ONLY Exits to Cash f* = 0.0%)...")
    # SELL signal in LONG_ONLY
    rec_sell = sizer.compute_recommendation(
        p_buy=0.10, p_hold=0.25, p_sell=0.65,
        b_upper=0.08, b_lower=0.04, entry_price=200.0,
        kappa=0.50, execution_mode="LONG_ONLY",
    )
    sell_cash_ok = (
        rec_sell.action == "SELL"
        and rec_sell.unconstrained_kelly_pct == 0.0
        and rec_sell.regime_scaled_kelly_pct == 0.0
        and rec_sell.l1_normalized_portfolio_allocation_pct == 0.0
    )

    # HOLD signal in LONG_ONLY
    rec_hold = sizer.compute_recommendation(
        p_buy=0.20, p_hold=0.60, p_sell=0.20,
        b_upper=0.08, b_lower=0.04, entry_price=100.0,
        kappa=0.50, execution_mode="LONG_ONLY",
    )
    hold_cash_ok = (
        rec_hold.action == "HOLD"
        and rec_hold.unconstrained_kelly_pct == 0.0
        and rec_hold.regime_scaled_kelly_pct == 0.0
        and rec_hold.l1_normalized_portfolio_allocation_pct == 0.0
    )

    # BUY signal in LONG_ONLY
    rec_buy = sizer.compute_recommendation(
        p_buy=0.65, p_hold=0.25, p_sell=0.10,
        b_upper=0.08, b_lower=0.04, entry_price=150.0,
        kappa=0.50, execution_mode="LONG_ONLY",
    )
    buy_ok = (
        rec_buy.action == "BUY"
        and rec_buy.unconstrained_kelly_pct > 0.0
        and rec_buy.regime_scaled_kelly_pct > 0.0
        and rec_buy.l1_normalized_portfolio_allocation_pct > 0.0
    )

    # 250-Stock Cross-Sectional Panel Non-Negativity Check
    rng4 = np.random.default_rng(999)
    panel_non_negative = True
    for _ in range(250):
        p_dir = rng4.dirichlet([1.0, 1.2, 1.0])
        r = sizer.compute_recommendation(
            p_buy=float(p_dir[0]), p_hold=float(p_dir[1]), p_sell=float(p_dir[2]),
            b_upper=0.08, b_lower=0.04, entry_price=100.0,
            kappa=0.50, execution_mode="LONG_ONLY",
        )
        if (
            r.unconstrained_kelly_pct < 0.0
            or r.regime_scaled_kelly_pct < 0.0
            or r.l1_normalized_portfolio_allocation_pct < 0.0
        ):
            panel_non_negative = False
        if r.action in ("SELL", "HOLD") and r.l1_normalized_portfolio_allocation_pct != 0.0:
            panel_non_negative = False

    # LONG_SHORT Diagnostic Mode Verification
    rec_ls_sell = sizer.compute_recommendation(
        p_buy=0.10, p_hold=0.20, p_sell=0.70,
        b_upper=0.08, b_lower=0.04, entry_price=200.0,
        kappa=0.50, execution_mode="LONG_SHORT",
    )
    ls_sell_ok = (rec_ls_sell.action == "SELL") and (rec_ls_sell.regime_scaled_kelly_pct < 0.0)

    t4_pass = sell_cash_ok and hold_cash_ok and buy_ok and panel_non_negative and ls_sell_ok
    print(f"  > Mode:                     LONG_ONLY (Institutional Cash Delivery)")
    print(f"  > SELL -> Cash (0.0%):      {'[PASS]' if sell_cash_ok else '[FAIL]'} (Alloc: {rec_sell.l1_normalized_portfolio_allocation_pct:.2f}%)")
    print(f"  > HOLD -> Cash (0.0%):      {'[PASS]' if hold_cash_ok else '[FAIL]'} (Alloc: {rec_hold.l1_normalized_portfolio_allocation_pct:.2f}%)")
    print(f"  > BUY -> Long (Alloc > 0):  {'[PASS]' if buy_ok else '[FAIL]'} (Alloc: {rec_buy.l1_normalized_portfolio_allocation_pct:.2f}%)")
    print(f"  > 250-Stock Panel Non-Neg:  {'[PASS]' if panel_non_negative else '[FAIL]'} (Strictly w_i >= 0.0)")
    print(f"  > LONG_SHORT Diagnostic:    {'[PASS]' if ls_sell_ok else '[FAIL]'} (Sell Short Alloc: {rec_ls_sell.regime_scaled_kelly_pct:.2f}%)")
    print(f"  > Status:                   {'[PASS]' if t4_pass else '[FAIL]'}")
    if not t4_pass:
        failures += 1

    elapsed = time.perf_counter() - t_start

    # ---------------------------------------------------------
    # GATE 4 SUMMARY TABLE REPORT
    # ---------------------------------------------------------
    print("\n" + "=" * 90)
    print("GATE 4 QUANTITATIVE FALSIFICATION BATTERY SUMMARY REPORT (QARE-MM v1.3)")
    print("=" * 90)
    header = f"| {'Criterion':<36} | {'Hurdle / Contract Invariant':<27} | {'Observed Value':<15} | {'State':<8} |"
    divider = "+" + "-" * 38 + "+" + "-" * 29 + "+" + "-" * 17 + "+" + "-" * 10 + "+"
    print(divider)
    print(header)
    print(divider)

    c1_state = "[PASS]" if t1_pass else "[FAIL]"
    c2_state = "[PASS]" if t2_pass else "[FAIL]"
    c3_state = "[PASS]" if t3_pass else "[FAIL]"
    c4_state = "[PASS]" if t4_pass else "[FAIL]"

    print(f"| {'1. Temperature Calibration':<36} | {'Delta Brier > 10.00%':<27} | {brier_drop * 100:>13.2f}% | {c1_state:<8} |")
    print(f"| {'2. Zero-Skill Kelly Allocation Guard':<36} | {'p_hat = p_bar ==> f* = 0.0%':<27} | {'0.00% (Strict)':>15} | {c2_state:<8} |")
    print(f"| {'3. Portfolio L1 Gross Leverage Cap':<36} | {'sum |w_i| <= 100.0%':<27} | {norm_gross * 100:>13.2f}% | {c3_state:<8} |")
    print(f"| {'4. Execution Mode Integrity':<36} | {'LONG_ONLY Exits to Cash':<27} | {'w_i >= 0.0%':>15} | {c4_state:<8} |")
    print(divider)

    print("\n" + "=" * 90)
    if failures == 0:
        print(f"GATE 4 FALSIFICATION BATTERY RESULT: PASSED (4/4) [{elapsed:.2f}s]")
    else:
        print(f"GATE 4 FALSIFICATION BATTERY RESULT: FAILED ({failures} failures) [{elapsed:.2f}s]")
    print("=" * 90 + "\n")

    return failures


if __name__ == "__main__":
    sys.exit(run_battery())
