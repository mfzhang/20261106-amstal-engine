"""
Tests for Residualized Triple-Barrier Labeling & Sample Uniqueness
Verifies 1D ATR barrier thresholds and concurrency weight normalization.
"""

import numpy as np
from src.label.triple_barrier import TripleBarrierLabeler
from src.label.weights import UniquenessWeighter

def test_volatility_scaling():
    labeler = TripleBarrierLabeler(upper_multiplier=2.0, lower_multiplier=1.0, vertical_horizon=10)
    eps_history = np.full(20, 0.01)  # Constant 1% residuals
    vol_scale = labeler.compute_volatility_scale(eps_history)
    # sqrt(10) * sqrt(1/20 * 20 * 0.01^2) = sqrt(10) * 0.01 ~= 0.03162
    expected = np.sqrt(10) * 0.01
    assert np.isclose(vol_scale, expected, atol=1e-5)

def test_triple_barrier_hits():
    labeler = TripleBarrierLabeler(upper_multiplier=2.0, lower_multiplier=1.0, vertical_horizon=10)
    vol_scale = 0.02  # Upper = +0.04, Lower = -0.02
    
    # Path hitting upper barrier at step 3
    path_up = np.array([0.01, 0.015, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    res_up = labeler.label_path(current_idx=50, future_eps_path_10=path_up, vol_scale=vol_scale)
    assert res_up.label == 0  # Buy
    assert res_up.exit_horizon == 3

    # Path hitting lower barrier at step 2
    path_down = np.array([-0.01, -0.015, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    res_down = labeler.label_path(current_idx=50, future_eps_path_10=path_down, vol_scale=vol_scale)
    assert res_down.label == 2  # Sell
    assert res_down.exit_horizon == 2

    # Path expiring at vertical barrier
    path_hold = np.zeros(10)
    res_hold = labeler.label_path(current_idx=50, future_eps_path_10=path_hold, vol_scale=vol_scale)
    assert res_hold.label == 1  # Hold
    assert res_hold.exit_horizon == 10

def test_uniqueness_weight_normalization():
    total_len = 100
    # Create overlapping intervals
    intervals = [
        (10, 20),
        (15, 25),
        (18, 28),
        (50, 60),
    ]
    weights = UniquenessWeighter.compute_uniqueness_weights(total_len, intervals)
    assert len(weights) == len(intervals)
    assert np.isclose(np.mean(weights), 1.0, atol=1e-6)
