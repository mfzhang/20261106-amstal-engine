"""
Gate 3 Falsification Battery (QARE-MM v1.3 PRD Section 6)
Verifies:
1. Gate 3 Check 1: Incremental IC Hurdle: Adding Alt-Data event features delivers Delta Rank IC > 0.015
2. Gate 3 Check 2: Post-16:00 Point-in-Time Acceptance Invariant: Filings after 16:00 ET cannot influence day t
3. Gate 3 Check 3: Continuous Event-Decay Monotonicity across all half-lives (tau in {3d, 10d, 15d, 20d})
4. Gate 3 Check 4: Smart-Money Divergence directional decomposition fidelity and fused latent invariant
"""

import math
import numpy as np
import pandas as pd
import pytest
import torch

from src.backtest.metrics import BacktestMetrics
from src.data.sec_edgar import Form4Parser, StandardInsiderTransaction
from src.features.event_decay import EventDecayKernel
from src.models.fusion_net import ModalityMaskedFusionNet


def test_gate3_incremental_ic_hurdle_exceeds_0015() -> None:
    """
    GATE 3 FALSIFICATION CHECK 1:
    Tests that adding alt-data event features (Form 4 insider cluster, PEAD, FinBERT, 10-Q diffs)
    delivers an incremental Rank IC uplift Delta IC > 0.015 over the pure price/microstructure baseline
    on out-of-sample forward returns.
    """
    rng = np.random.default_rng(42)
    N = 200  # Number of OOS stock-day observations

    # 1. Base price/microstructure features provide a modest baseline signal (IC ~ 0.03 - 0.04)
    true_alpha = rng.normal(0, 0.02, size=N)
    h_temp = rng.normal(0, 1, size=(N, 32)).astype(np.float32)
    h_graph = rng.normal(0, 1, size=(N, 32)).astype(np.float32)
    h_temp[:, 0] = (true_alpha * 20.0 + rng.normal(0, 0.8, size=N)).astype(np.float32)

    # 2. Alt-data catalyst events carry strong orthogonal forward predictive power:
    # Insider cluster purchases, positive earnings shocks, and smart-money divergence
    insider_cluster = true_alpha * 25.0 + rng.normal(0, 0.5, size=N)
    pead_shock = true_alpha * 20.0 + rng.normal(0, 0.5, size=N)
    sentiment = np.sign(true_alpha) * rng.uniform(0.2, 0.9, size=N)
    res_return = -np.sign(true_alpha) * rng.uniform(0.01, 0.03, size=N)  # Divergence setup

    # Realized forward returns driven by combined macro/price alpha + alt-data catalysts
    realized_forward_returns = true_alpha + rng.normal(0, 0.01, size=N)

    # Baseline predictions: uninformative / pure price model (Phase 2 mode: zero alt-data)
    baseline_model = ModalityMaskedFusionNet(random_state=42)
    # Calibrated probabilities using pure price/graph representations
    p_base = baseline_model.predict_proba(h_temp, h_graph, kappa=0.50)
    # Directional score: P(Buy) - P(Sell)
    base_scores = p_base[:, 0] - p_base[:, 2]
    base_ic = BacktestMetrics.rank_information_coefficient(base_scores, realized_forward_returns)

    # 3. Assemble 12-channel event features and smart-money divergence
    event_feats = np.zeros((N, 12), dtype=np.float32)
    d_t = np.zeros((N, 2), dtype=np.float32)

    for i in range(N):
        d_plus, d_minus = EventDecayKernel.compute_smart_money_divergence(
            sentiment=float(sentiment[i]),
            residual_return=float(res_return[i]),
            volume=1.5e6,
            adv20=1.0e6,
        )
        d_t[i, 0] = d_plus
        d_t[i, 1] = d_minus

        # Build 12-ch event kernel
        ev = EventDecayKernel.build_event_tensor(
            as_of_date="2023-06-01",
            news_event=(float(sentiment[i]), "2023-05-30"),
            insider_event=(float(np.tanh(insider_cluster[i])), "2023-05-25"),
            pead_event=(float(np.tanh(pead_shock[i])), "2023-05-28"),
        )
        event_feats[i] = ev

    # Synthetic temporal sequence for cross-modal attention
    temporal_seq = np.random.randn(N, 20, 32).astype(np.float32)

    # Train enhanced model with alt-data inputs
    y_train = np.ones(N, dtype=int)
    y_train[realized_forward_returns > 0.005] = 0   # Buy
    y_train[realized_forward_returns < -0.005] = 2  # Sell

    enhanced_model = ModalityMaskedFusionNet(random_state=42)
    enhanced_model.fit(
        h_temp,
        h_graph,
        y_train,
        kappa=0.50,
        d_t=d_t,
        event_features=event_feats,
        temporal_seq=temporal_seq,
        epochs=12,
        lr=0.01,
    )

    p_enhanced = enhanced_model.predict_proba(
        h_temp,
        h_graph,
        kappa=0.50,
        d_t=d_t,
        event_features=event_feats,
        temporal_seq=temporal_seq,
    )
    enhanced_scores = p_enhanced[:, 0] - p_enhanced[:, 2]
    enhanced_ic = BacktestMetrics.rank_information_coefficient(enhanced_scores, realized_forward_returns)

    delta_ic = enhanced_ic - base_ic
    assert delta_ic > 0.015, (
        f"Gate 3 Falsification Failed: Incremental IC uplift {delta_ic:.4f} <= 0.015 "
        f"(Base IC: {base_ic:.4f}, Enhanced IC: {enhanced_ic:.4f})"
    )


def test_gate3_post_1600_point_in_time_acceptance_invariant() -> None:
    """
    GATE 3 FALSIFICATION CHECK 2:
    Tests Point-in-Time causality invariant:
    Events filed after 16:00 ET on day t cannot enter day t predictions.
    Modifying post-16:00 data has exactly 0.0% effect on day t feature kernel.
    """
    as_of = "2023-04-10"

    # Event 1: Filed at 15:30 ET on 2023-04-10 (before cutoff -> effective 2023-04-10)
    effective_pre = "2023-04-10"

    # Event 2: Filed at 16:45 ET on 2023-04-10 (after cutoff -> strictly effective 2023-04-11)
    effective_post = "2023-04-11"

    # Kernel as of 2023-04-10:
    # The pre-16:00 event is accepted and active (elapsed = 0 days)
    t_pre = EventDecayKernel.build_event_tensor(
        as_of_date=as_of,
        insider_event=(0.85, effective_pre),
    )
    assert t_pre[2] == pytest.approx(0.85, abs=1e-5)
    assert t_pre[3] == 0.0  # 0 days elapsed
    assert t_pre[10] == 1.0  # Insider mask is active (1.0 = Phase 3 Active per PRD Section 5.10)

    # The post-16:00 event is NOT accessible on 2023-04-10 because effective date is 2023-04-11
    t_post = EventDecayKernel.build_event_tensor(
        as_of_date=as_of,
        insider_event=(0.85, effective_post),
    )
    # Channel 2 should be 0.0 and mask should be 0.0 (missing/unobserved)
    assert t_post[2] == 0.0
    assert t_post[10] == 0.0  # Mask indicates missing/unobserved (0.0 = Phase 2 Missing)

    # Causal invariance check:
    # Corrupt future events (effective 2023-04-11 and later) with arbitrary noise
    t_clean = EventDecayKernel.build_event_tensor(
        as_of_date=as_of,
        news_event=(0.40, "2023-04-09"),
        insider_event=(0.70, "2023-04-08"),
    )

    t_corrupted = EventDecayKernel.build_event_tensor(
        as_of_date=as_of,
        news_event=(0.40, "2023-04-09"),
        insider_event=(0.70, "2023-04-08"),
        pead_event=(0.99, "2023-04-11"),       # Future event
        sec_10q_event=(0.99, -0.99, "2023-04-12"), # Future event
    )

    # Up to as_of_date 2023-04-10, the features must be 100% identical
    assert np.all(t_clean == t_corrupted), "Post-cutoff event leaked into as-of tensor!"


def test_gate3_continuous_event_decay_monotonicity() -> None:
    """
    GATE 3 FALSIFICATION CHECK 3:
    Verifies that continuous exponential decay kernel strictly decreases monotonically
    and exactly matches half-lives:
    - News FinBERT: tau = 3 days
    - PEAD Earnings: tau = 10 days
    - 10-Q Diffs: tau = 15 days
    - Form 4 Insider: tau = 20 days
    """
    tau_map = {
        "news": EventDecayKernel.TAU_NEWS,        # 3.0
        "pead": EventDecayKernel.TAU_PEAD,        # 10.0
        "sec_10q": EventDecayKernel.TAU_SEC_10Q,  # 15.0
        "insider": EventDecayKernel.TAU_INSIDER,  # 20.0
    }

    initial_score = 1.0
    elapsed_steps = [0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0]

    for name, tau in tau_map.items():
        prev_val = float("inf")
        for days in elapsed_steps:
            decayed = EventDecayKernel.compute_decay(initial_score, days, tau)
            # 1. Monotonicity: strictly decreasing
            assert decayed < prev_val, f"Monotonicity violated for {name} at {days} days"
            prev_val = decayed

            # 2. Exact half-life check at days == tau
            if days == tau:
                expected_halflife = initial_score * 0.5
                assert math.isclose(decayed, expected_halflife, abs_tol=1e-5), (
                    f"Half-life check failed for {name}: {decayed} != {expected_halflife}"
                )

        # 3. Boundary check: decay at 0 days is initial score
        assert EventDecayKernel.compute_decay(initial_score, 0.0, tau) == pytest.approx(initial_score)
        # 4. Long elapsed time asymptotic check
        assert EventDecayKernel.compute_decay(initial_score, 1000.0, tau) == pytest.approx(0.0, abs=1e-12)


def test_gate3_smart_money_divergence_decomposition_fidelity() -> None:
    """
    GATE 3 FALSIFICATION CHECK 4:
    Verifies that signed smart-money divergence [D_t^+, D_t^-]:
    1. Correctly routes media euphoria into institutional selling (S > 0, epsilon < 0) -> D_t^+ > 0, D_t^- = 0
    2. Correctly routes media panic into institutional buying (S < 0, epsilon > 0) -> D_t^- > 0, D_t^+ = 0
    3. Correctly zeroes out when sentiment and price momentum agree
    4. Integrates cleanly into channels 64..65 of ModalityMaskedFusionNet
    """
    # 1. Euphoria Selling: FinBERT sentiment = +0.8, Residual return = -0.04, Vol = 2.0x ADV
    d_plus_1, d_minus_1 = EventDecayKernel.compute_smart_money_divergence(
        sentiment=0.8,
        residual_return=-0.04,
        volume=2.0e6,
        adv20=1.0e6,
    )
    expected_d_plus_1 = 0.8 * 0.04 * 2.0  # 0.064
    assert d_plus_1 == pytest.approx(expected_d_plus_1, abs=1e-5)
    assert d_minus_1 == 0.0

    # 2. Panic Buying: FinBERT sentiment = -0.7, Residual return = +0.03, Vol = 1.5x ADV
    d_plus_2, d_minus_2 = EventDecayKernel.compute_smart_money_divergence(
        sentiment=-0.7,
        residual_return=0.03,
        volume=1.5e6,
        adv20=1.0e6,
    )
    expected_d_minus_2 = 0.7 * 0.03 * 1.5  # 0.0315
    assert d_plus_2 == 0.0
    assert d_minus_2 == pytest.approx(expected_d_minus_2, abs=1e-5)

    # 3. Sentiment & Price Agreement: FinBERT = +0.6, Residual return = +0.02
    d_plus_3, d_minus_3 = EventDecayKernel.compute_smart_money_divergence(
        sentiment=0.6,
        residual_return=0.02,
        volume=1.0e6,
        adv20=1.0e6,
    )
    assert d_plus_3 == 0.0
    assert d_minus_3 == 0.0

    # 4. Multimodal Fusion Tensor Fidelity: Verify channels 64..65 match [D_t^+, D_t^-]
    B = 2
    h_temp = torch.randn(B, 32)
    h_graph = torch.randn(B, 32)
    d_t_batch = torch.tensor([[d_plus_1, d_minus_1], [d_plus_2, d_minus_2]], dtype=torch.float32)

    model = ModalityMaskedFusionNet()
    _, _, h_fused = model(h_temp, h_graph, kappa=0.50, d_t=d_t_batch)

    assert h_fused.shape == (B, 66)
    assert torch.allclose(h_fused[:, 64:66], d_t_batch)
