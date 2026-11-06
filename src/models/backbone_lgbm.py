"""
Sample-Weighted LightGBM Backbone Classifier (QARE-MM v1.3)
Consumes (Batch, 105) structured multi-scale summary features with intra-ticker uniqueness sample weights.
Predicts ternary triple-barrier outcomes {0: Buy, 1: Hold, 2: Sell} with post-hoc Temperature Scaling calibration.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp


class LightGBMBackbone:
    """
    Sample-Weighted LightGBM Classifier for (Batch, 105) multi-scale tabular features.
    Supports early stopping on validation fold and post-hoc Temperature Scaling.
    """

    NUM_CLASSES = 3

    def __init__(
        self,
        n_estimators: int = 150,
        learning_rate: float = 0.03,
        num_leaves: int = 31,
        max_depth: int = -1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        n_jobs: int = 1,
        random_state: int = 42,
    ) -> None:
        self.params: Dict[str, Any] = {
            "objective": "multiclass",
            "num_class": self.NUM_CLASSES,
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            "verbose": -1,
            "n_jobs": n_jobs,
        }
        self.model: Optional[Any] = None
        self.temperature_: float = 1.0
        self.is_fitted_: bool = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
        eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        early_stopping_rounds: Optional[int] = 15,
    ) -> "LightGBMBackbone":
        """
        Fits multiclass booster with intra-ticker sample uniqueness weights u_{i, t}.
        """
        X_tr = np.asarray(X_train, dtype=np.float64)
        y_tr = np.asarray(y_train, dtype=int)
        weights = np.asarray(sample_weights, dtype=np.float64) if sample_weights is not None else None

        try:
            import lightgbm as lgb

            self.model = lgb.LGBMClassifier(**self.params)
            callbacks = []
            if eval_set is not None and early_stopping_rounds:
                callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False))

            fit_kwargs: Dict[str, Any] = {"sample_weight": weights}
            if eval_set is not None:
                fit_kwargs["eval_set"] = [eval_set]
                if callbacks:
                    fit_kwargs["callbacks"] = callbacks

            self.model.fit(X_tr, y_tr, **fit_kwargs)
        except ImportError:
            # Fallback to HistGradientBoostingClassifier if lightgbm not installed
            from sklearn.ensemble import HistGradientBoostingClassifier

            self.model = HistGradientBoostingClassifier(
                max_iter=self.params["n_estimators"],
                learning_rate=self.params["learning_rate"],
                random_state=self.params["random_state"],
            )
            self.model.fit(X_tr, y_tr, sample_weight=weights)

        self.is_fitted_ = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns raw predicted class probabilities (Batch, 3)."""
        if not self.is_fitted_ or self.model is None:
            raise RuntimeError("Model has not been fitted.")
        X_arr = np.asarray(X, dtype=np.float64)
        probs = self.model.predict_proba(X_arr)
        # Normalize to strictly sum to 1.0
        row_sums = probs.sum(axis=1, keepdims=True)
        return probs / np.maximum(row_sums, 1e-12)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns predicted class labels (Batch,) in {0, 1, 2}."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def fit_temperature(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """
        Fits post-hoc Temperature parameter T > 0 on validation fold to optimize probability calibration
        and minimize negative log-likelihood (Brier/ECE loss).
        """
        raw_probs = self.predict_proba(X_val)
        # Convert probabilities to unnormalized logits
        logits = np.log(np.clip(raw_probs, 1e-12, 1.0 - 1e-12))
        y_v = np.asarray(y_val, dtype=int)
        n_samples = len(y_v)

        def nll_objective(T_val: float) -> float:
            if T_val <= 0:
                return 1e9
            scaled = logits / T_val
            log_denom = logsumexp(scaled, axis=1)
            # Log probability of true classes
            target_logits = scaled[np.arange(n_samples), y_v]
            log_likelihood = target_logits - log_denom
            return float(-np.mean(log_likelihood))

        res = minimize_scalar(nll_objective, bounds=(0.05, 5.0), method="bounded")
        self.temperature_ = float(res.x)
        return self.temperature_

    def predict_calibrated_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns temperature-calibrated class probabilities (Batch, 3) summing strictly to 1.0:
        p_k = exp(z_k / T) / sum_j exp(z_j / T)
        """
        raw_probs = self.predict_proba(X)
        if abs(self.temperature_ - 1.0) < 1e-5:
            return raw_probs

        logits = np.log(np.clip(raw_probs, 1e-12, 1.0 - 1e-12))
        scaled_logits = logits / self.temperature_
        log_denom = logsumexp(scaled_logits, axis=1, keepdims=True)
        calibrated = np.exp(scaled_logits - log_denom)
        return calibrated
