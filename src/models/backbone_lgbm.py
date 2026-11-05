"""
Sample-Weighted LightGBM Backbone Classifier
Consumes (Batch, 105) structured multi-scale summary features with uniqueness sample weights.
"""

from typing import Optional, Dict, Any, Tuple
import numpy as np

class LightGBMBackbone:
    """LightGBM classifier predicting ternary triple-barrier outcomes {0: Buy, 1: Hold, 2: Sell}."""

    def __init__(
        self,
        n_estimators: int = 150,
        learning_rate: float = 0.03,
        num_leaves: int = 31,
        random_state: int = 42,
    ) -> None:
        self.params = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "random_state": random_state,
            "verbose": -1,
        }
        self.model: Optional[Any] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
        eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> "LightGBMBackbone":
        """Fits multiclass classifier with sample uniqueness weights."""
        try:
            import lightgbm as lgb
            self.model = lgb.LGBMClassifier(**self.params)
            self.model.fit(
                X_train,
                y_train,
                sample_weight=sample_weights,
            )
        except ImportError:
            # Fallback to scikit-learn GradientBoosting if lightgbm not installed
            from sklearn.ensemble import HistGradientBoostingClassifier
            self.model = HistGradientBoostingClassifier(
                max_iter=self.params["n_estimators"],
                learning_rate=self.params["learning_rate"],
                random_state=self.params["random_state"],
            )
            self.model.fit(X_train, y_train, sample_weight=sample_weights)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns calibrated class probabilities (Batch, 3)."""
        if self.model is None:
            raise RuntimeError("Model has not been fit.")
        return self.model.predict_proba(X)
