"""
Post-Hoc Probability Calibration via Temperature Scaling (QARE-MM v1.3)
Fits temperature parameter T > 0 by minimizing Negative Log-Likelihood (NLL)
on the 126-day validation fold to produce well-calibrated ternary probabilities.
"""

from typing import Optional, Union
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp


class TemperatureScaler:
    """
    Learns post-hoc temperature parameter T > 0 on validation fold predictions.
    Scales logits: z_cal = z / T, ensuring probabilities reflect true empirical frequencies.
    """

    def __init__(self, bounds: tuple[float, float] = (0.05, 5.0)) -> None:
        self.bounds = bounds
        self.temperature_: float = 1.0
        self.is_fitted_: bool = False

    def fit(self, raw_probs: np.ndarray, y_val: np.ndarray) -> float:
        """
        Fits temperature parameter T > 0 minimizing NLL on validation probabilities and labels.
        """
        probs = np.asarray(raw_probs, dtype=np.float64)
        y = np.asarray(y_val, dtype=int)
        if probs.ndim == 1:
            probs = probs.reshape(1, -1)
        n_samples = len(y)

        if n_samples == 0:
            self.temperature_ = 1.0
            self.is_fitted_ = True
            return 1.0

        # Clip raw probabilities to prevent log(0)
        clipped_probs = np.clip(probs, 1e-12, 1.0 - 1e-12)
        logits = np.log(clipped_probs)

        def nll_objective(T_val: float) -> float:
            if T_val <= 0:
                return 1e9
            scaled_logits = logits / T_val
            log_denom = logsumexp(scaled_logits, axis=1)
            target_logits = scaled_logits[np.arange(n_samples), y]
            log_likelihood = target_logits - log_denom
            return float(-np.mean(log_likelihood))

        res = minimize_scalar(nll_objective, bounds=self.bounds, method="bounded")
        self.temperature_ = float(res.x)
        self.is_fitted_ = True
        return self.temperature_

    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        """
        Transforms raw probabilities using the fitted temperature parameter.
        """
        probs = np.asarray(raw_probs, dtype=np.float64)
        if len(probs) == 0 or abs(self.temperature_ - 1.0) < 1e-5:
            return probs

        is_1d = (probs.ndim == 1)
        if is_1d:
            probs = probs.reshape(1, -1)

        clipped_probs = np.clip(probs, 1e-12, 1.0 - 1e-12)
        logits = np.log(clipped_probs)
        scaled_logits = logits / self.temperature_
        log_denom = logsumexp(scaled_logits, axis=1, keepdims=True)
        res = np.exp(scaled_logits - log_denom)
        return res.squeeze(0) if is_1d else res

    def fit_transform(self, raw_probs: np.ndarray, y_val: np.ndarray) -> np.ndarray:
        self.fit(raw_probs, y_val)
        return self.transform(raw_probs)
