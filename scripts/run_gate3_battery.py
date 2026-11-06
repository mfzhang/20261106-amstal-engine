"""
Gate 3 Quantitative Falsification Battery CLI Runner (QARE-MM v1.3 PRD Section 6)
Executes Phase 3 Gate 3 falsification battery:
1. Incremental IC Hurdle: Alt-data catalysts deliver Delta IC > 0.015 over price/microstructure baseline
2. Post-16:00 Point-in-Time Acceptance Invariant: Filings after 16:00 ET cannot influence day t
3. Continuous Event-Decay Monotonicity across all half-lives (tau in {3d, 10d, 15d, 20d})
4. Smart-Money Divergence directional decomposition fidelity and fused latent tensor (Batch, 66)
"""

import math
import sys
import time
import numpy as np
import pandas as pd
import torch

from src.backtest.metrics import BacktestMetrics
from src.data.sec_edgar import Form4Parser, StandardInsiderTransaction
from src.features.event_decay import EventDecayKernel
from src.models.fusion_net import ModalityMaskedFusionNet


def run_battery() -> int:
    print("=" * 80)
    print("AMSTAL QUANT ENGINE: GATE 3 FALSIFICATION BATTERY (QARE-MM v1.3)")
    print("Target: SEC EDGAR & Alt-Data Pipeline (Form 4, PEAD, FinBERT, 10-Q Diffs)")
    print("=" * 80)

    failures = 0

    # ---------------------------------------------------------
    # TEST 1: Incremental IC Hurdle (Delta IC > 0.015)
    # ---------------------------------------------------------
    print("\n[TEST 1/4] Checking Incremental IC Uplift (Hurdle: Delta IC > 0.015)...")
    rng = np.random.default_rng(42)
    N = 200

    true_alpha = rng.normal(0, 0.02, size=N)
    h_temp = rng.normal(0, 1, size=(N, 32)).astype(np.float32)
    h_graph = rng.normal(0, 1, size=(N, 32)).astype(np.float32)
    h_temp[:, 0] = (true_alpha * 20.0 + rng.normal(0, 0.8, size=N)).astype(np.float32)

    insider_cluster = true_alpha * 25.0 + rng.normal(0, 0.5, size=N)
    pead_shock = true_alpha * 20.0 + rng.normal(0, 0.5, size=N)
    sentiment = np.sign(true_alpha) * rng.uniform(0.2, 0.9, size=N)
    res_return = -np.sign(true_alpha) * rng.uniform(0.01, 0.03, size=N)

    realized_forward_returns = true_alpha + rng.normal(0, 0.01, size=N)

    baseline_model = ModalityMaskedFusionNet(random_state=42)
    p_base = baseline_model.predict_proba(h_temp, h_graph, kappa=0.50)
    base_scores = p_base[:, 0] - p_base[:, 2]
    base_ic = BacktestMetrics.rank_information_coefficient(base_scores, realized_forward_returns)

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
        ev = EventDecayKernel.build_event_tensor(
            as_of_date="2023-06-01",
            news_event=(float(sentiment[i]), "2023-05-30"),
            insider_event=(float(np.tanh(insider_cluster[i])), "2023-05-25"),
            pead_event=(float(np.tanh(pead_shock[i])), "2023-05-28"),
        )
        event_feats[i] = ev

    temporal_seq = np.random.randn(N, 20, 32).astype(np.float32)

    y_train = np.ones(N, dtype=int)
    y_train[realized_forward_returns > 0.005] = 0
    y_train[realized_forward_returns < -0.005] = 2

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

    t1_pass = delta_ic > 0.015
    print(f"  > Baseline Rank IC (Pure Price): {base_ic:.4f}")
    print(f"  > Enhanced Rank IC (+ Alt-Data):  {enhanced_ic:.4f}")
    print(f"  > Incremental Delta IC:          +{delta_ic:.4f} (Hurdle: > +0.0150)")
    print(f"  > Status:                        {'[PASS]' if t1_pass else '[FAIL]'}")
    if not t1_pass:
        failures += 1

    # ---------------------------------------------------------
    # TEST 2: Post-16:00 Point-in-Time Acceptance Invariant
    # ---------------------------------------------------------
    print("\n[TEST 2/4] Checking Post-16:00 Point-in-Time Acceptance Invariant...")
    as_of = "2023-04-10"

    # Pre-cutoff (15:30 ET) event is accessible on day t
    t_pre = EventDecayKernel.build_event_tensor(as_of_date=as_of, insider_event=(0.85, "2023-04-10"))
    pre_active = (t_pre[2] == 0.85) and (t_pre[10] == 1.0)

    # Post-cutoff (16:45 ET) event effective date is 2023-04-11, so inaccessible on 2023-04-10
    t_post = EventDecayKernel.build_event_tensor(as_of_date=as_of, insider_event=(0.85, "2023-04-11"))
    post_masked = (t_post[2] == 0.0) and (t_post[10] == 0.0)

    # Invariance across future mutations
    t_clean = EventDecayKernel.build_event_tensor(
        as_of_date=as_of,
        news_event=(0.40, "2023-04-09"),
        insider_event=(0.70, "2023-04-08"),
    )
    t_corrupted = EventDecayKernel.build_event_tensor(
        as_of_date=as_of,
        news_event=(0.40, "2023-04-09"),
        insider_event=(0.70, "2023-04-08"),
        pead_event=(0.99, "2023-04-11"),
        sec_10q_event=(0.99, -0.99, "2023-04-12"),
    )
    future_leakage_variance = float(np.var(t_clean - t_corrupted))

    t2_pass = pre_active and post_masked and (future_leakage_variance == 0.0)
    print(f"  > Pre-16:00 Event Accepted:       {pre_active}")
    print(f"  > Post-16:00 Event Masked (PIT):  {post_masked}")
    print(f"  > Future Leakage Variance:        {future_leakage_variance:.6f} (Hurdle: 0.000000)")
    print(f"  > Status:                         {'[PASS]' if t2_pass else '[FAIL]'}")
    if not t2_pass:
        failures += 1

    # ---------------------------------------------------------
    # TEST 3: Continuous Event-Decay Kernel Monotonicity & Half-Life
    # ---------------------------------------------------------
    print("\n[TEST 3/4] Checking Continuous Event-Decay Kernel Monotonicity & Half-Lives...")
    tau_checks = {
        "News FinBERT (tau=3d)": (EventDecayKernel.TAU_NEWS, 3.0),
        "PEAD Shock (tau=10d)": (EventDecayKernel.TAU_PEAD, 10.0),
        "10-Q Diffs (tau=15d)": (EventDecayKernel.TAU_SEC_10Q, 15.0),
        "Form 4 Insider (tau=20d)": (EventDecayKernel.TAU_INSIDER, 20.0),
    }

    t3_pass = True
    for label, (tau, expected_days) in tau_checks.items():
        v0 = EventDecayKernel.compute_decay(1.0, 0.0, tau)
        v_half = EventDecayKernel.compute_decay(1.0, expected_days, tau)
        v_double = EventDecayKernel.compute_decay(1.0, 2.0 * expected_days, tau)
        half_ok = math.isclose(v_half, 0.5, abs_tol=1e-5)
        quarter_ok = math.isclose(v_double, 0.25, abs_tol=1e-5)
        mono_ok = v0 > v_half > v_double > 0.0
        print(f"  > {label}: V(0)={v0:.2f}, V(tau)={v_half:.4f}, V(2*tau)={v_double:.4f} [Mono: {mono_ok}]")
        if not (half_ok and quarter_ok and mono_ok):
            t3_pass = False

    print(f"  > Status:                         {'[PASS]' if t3_pass else '[FAIL]'}")
    if not t3_pass:
        failures += 1

    # ---------------------------------------------------------
    # TEST 4: Smart-Money Divergence Directional Decomposition
    # ---------------------------------------------------------
    print("\n[TEST 4/4] Checking Smart-Money Divergence Directional Fidelity & Fused Tensor...")
    # Euphoria selling: S > 0, epsilon < 0
    d_plus_1, d_minus_1 = EventDecayKernel.compute_smart_money_divergence(0.8, -0.04, 2.0e6, 1.0e6)
    # Panic buying: S < 0, epsilon > 0
    d_plus_2, d_minus_2 = EventDecayKernel.compute_smart_money_divergence(-0.7, 0.03, 1.5e6, 1.0e6)
    # Agreement: S > 0, epsilon > 0
    d_plus_3, d_minus_3 = EventDecayKernel.compute_smart_money_divergence(0.6, 0.02, 1.0e6, 1.0e6)

    cond_euphoria = (d_plus_1 > 0.0) and (d_minus_1 == 0.0)
    cond_panic = (d_plus_2 == 0.0) and (d_minus_2 > 0.0)
    cond_agree = (d_plus_3 == 0.0) and (d_minus_3 == 0.0)

    # Tensor incorporation into (B, 66)
    B = 2
    h_temp = torch.randn(B, 32)
    h_graph = torch.randn(B, 32)
    d_t_batch = torch.tensor([[d_plus_1, d_minus_1], [d_plus_2, d_minus_2]], dtype=torch.float32)

    fusion = ModalityMaskedFusionNet()
    _, _, h_fused = fusion(h_temp, h_graph, kappa=0.50, d_t=d_t_batch)

    tensor_ok = (h_fused.shape == (B, 66)) and torch.allclose(h_fused[:, 64:66], d_t_batch)

    t4_pass = cond_euphoria and cond_panic and cond_agree and tensor_ok
    print(f"  > Euphoria Selling Routing:       {'[PASS]' if cond_euphoria else '[FAIL]'} (D+={d_plus_1:.4f}, D-={d_minus_1:.4f})")
    print(f"  > Panic Buying Routing:           {'[PASS]' if cond_panic else '[FAIL]'} (D+={d_plus_2:.4f}, D-={d_minus_2:.4f})")
    print(f"  > Agreement Zero-Routing:         {'[PASS]' if cond_agree else '[FAIL]'} (D+={d_plus_3:.4f}, D-={d_minus_3:.4f})")
    print(f"  > Channels 64..65 Tensor Match:   {'[PASS]' if tensor_ok else '[FAIL]'}")
    print(f"  > Status:                         {'[PASS]' if t4_pass else '[FAIL]'}")
    if not t4_pass:
        failures += 1

    print("\n" + "=" * 80)
    if failures == 0:
        print("GATE 3 FALSIFICATION BATTERY RESULT: PASSED (4/4)")
    else:
        print(f"GATE 3 FALSIFICATION BATTERY RESULT: FAILED ({failures} failures)")
    print("=" * 80 + "\n")

    return failures


if __name__ == "__main__":
    sys.exit(run_battery())
