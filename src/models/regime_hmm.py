"""
Forward-Filtered 3-State Gaussian HMM (QARE-MM v1.3)
Implements causal log-space forward filtering with zero forward-backward lookahead leak.
Enforces invariant state ordering by SPY return variance (State 0: Low-Vol, State 1: Med-Vol, State 2: High-Vol).
"""

from typing import Dict, Optional, Tuple
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm


class RegimeHMM:
    """3-State Volatility Regime HMM with causal forward recursion and variance sorting."""

    NUM_STATES = 3
    STATE_LABELS = {
        0: "LOW_VOLATILITY",
        1: "MEDIUM_VOLATILITY",
        2: "HIGH_VOLATILITY",
    }
    # Macro regime Kelly scaling multipliers
    REGIME_MULTIPLIERS = {
        0: 0.50,  # Half-Kelly in low volatility
        1: 0.35,  # Moderate Kelly in medium volatility
        2: 0.25,  # Quarter-Kelly in high volatility / turbulent regime
    }

    def __init__(
        self,
        n_iter: int = 30,
        random_state: int = 42,
        standardize: bool = True,
        n_fits: int = 10,
    ) -> None:
        self.n_iter = n_iter
        self.random_state = random_state
        self.standardize = standardize
        self.n_fits = n_fits
        self.mean_: Optional[np.ndarray] = None         # (2,)
        self.std_: Optional[np.ndarray] = None          # (2,)
        self.means_: Optional[np.ndarray] = None        # (3, 2)
        self.covars_: Optional[np.ndarray] = None       # (3, 2) or (3, 2, 2)
        self.transmat_: Optional[np.ndarray] = None     # (3, 3)
        self.startprob_: Optional[np.ndarray] = None    # (3,)
        self.state_order_: Optional[np.ndarray] = None  # 1D index mapping (3,)

    def transform_observation(self, x: np.ndarray) -> np.ndarray:
        """Applies fitted standardization to an input observation."""
        x_arr = np.asarray(x, dtype=np.float64)
        if self.standardize and self.mean_ is not None and self.std_ is not None:
            return (x_arr - self.mean_) / self.std_
        return x_arr

    def fit(self, X_train: np.ndarray) -> "RegimeHMM":
        """
        Fits 3-state Gaussian HMM on 2D series [SPY_returns, VIX].
        Standardizes input with fold-scoped statistics, runs multi-restart EM,
        and sorts states strictly by SPY return variance component Sigma_{k, rr}.
        """
        X = np.asarray(X_train, dtype=np.float64)
        if self.standardize:
            self.mean_ = np.mean(X, axis=0)
            std = np.std(X, axis=0)
            self.std_ = np.where(std < 1e-8, 1.0, std)
            X_proc = (X - self.mean_) / self.std_
        else:
            self.mean_ = np.zeros(X.shape[1], dtype=np.float64)
            self.std_ = np.ones(X.shape[1], dtype=np.float64)
            X_proc = X

        try:
            from hmmlearn.hmm import GaussianHMM

            best_score = -np.inf
            best_hmm = None

            for i in range(max(1, self.n_fits)):
                seed = self.random_state + i
                raw_hmm = GaussianHMM(
                    n_components=self.NUM_STATES,
                    covariance_type="diag",
                    n_iter=self.n_iter,
                    tol=1e-3,
                    random_state=seed,
                )
                try:
                    raw_hmm.fit(X_proc)
                    score = raw_hmm.score(X_proc)
                    if score > best_score or best_hmm is None:
                        best_score = score
                        best_hmm = raw_hmm
                except Exception:
                    continue

            if best_hmm is None:
                raise RuntimeError("Failed to fit GaussianHMM on any restart.")

            raw_hmm = best_hmm

            # Extract 1D variance component for SPY return (feature index 0)
            if raw_hmm.covars_.ndim == 3:
                variances = raw_hmm.covars_[:, 0, 0]
            else:
                variances = raw_hmm.covars_[:, 0]

            self.state_order_ = np.argsort(variances).astype(int)

            self.means_ = raw_hmm.means_[self.state_order_]
            self.covars_ = raw_hmm.covars_[self.state_order_]
            self.startprob_ = raw_hmm.startprob_[self.state_order_]

            # Reorder transition matrix rows and columns
            trans = raw_hmm.transmat_[self.state_order_, :]
            self.transmat_ = trans[:, self.state_order_]
        except ImportError:
            # Deterministic fallback parameters
            self.means_ = np.array([[0.05, -0.5], [0.0, 0.0], [-0.1, 1.2]])
            self.covars_ = np.array([[0.5, 0.5], [1.0, 1.0], [2.5, 2.5]])
            self.transmat_ = np.array([
                [0.92, 0.06, 0.02],
                [0.05, 0.90, 0.05],
                [0.03, 0.07, 0.90],
            ])
            self.startprob_ = np.array([0.7, 0.2, 0.1])
            self.state_order_ = np.array([0, 1, 2])

        return self

    def forward_filter_step(
        self,
        x_t: np.ndarray,
        log_alpha_prev: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Executes one causal log-space forward recursion step:
        log_alpha_tilde(k) = log P(x_t | S_t = k) + logsumexp_j (log_alpha_prev(j) + log A_{j, k})
        p_t(k) = exp(log_alpha_tilde(k) - logsumexp_j log_alpha_tilde(j))
        """
        if self.means_ is None or self.covars_ is None or self.transmat_ is None:
            raise RuntimeError("GaussianHMM must be fit before filtering.")

        log_emission = np.zeros(self.NUM_STATES)
        for k in range(self.NUM_STATES):
            # Extract diagonal variance components
            if self.covars_.ndim == 3:
                cov_diag = np.diagonal(self.covars_[k])
            else:
                cov_diag = self.covars_[k]
            scale = np.sqrt(np.maximum(cov_diag, 1e-8))
            log_emission[k] = np.sum(norm.logpdf(x_t, loc=self.means_[k], scale=scale))

        log_trans = np.log(np.maximum(self.transmat_, 1e-12))
        log_alpha_tilde = np.zeros(self.NUM_STATES)
        for k in range(self.NUM_STATES):
            log_alpha_tilde[k] = log_emission[k] + logsumexp(log_alpha_prev + log_trans[:, k])

        # Normalize in log-space
        log_norm = logsumexp(log_alpha_tilde)
        log_alpha_t = log_alpha_tilde - log_norm
        prob_t = np.exp(log_alpha_t)

        return prob_t, log_alpha_t

    def filter_sequence(self, X: np.ndarray) -> np.ndarray:
        """Causally forward-filters sequence X with zero lookahead leak."""
        if self.means_ is None or self.covars_ is None or self.transmat_ is None or self.startprob_ is None:
            raise RuntimeError("GaussianHMM must be fit before filtering.")

        X_arr = np.asarray(X, dtype=np.float64)
        if self.standardize and self.mean_ is not None and self.std_ is not None:
            X_arr = (X_arr - self.mean_) / self.std_

        T = len(X_arr)
        probs = np.zeros((T, self.NUM_STATES))
        log_alpha = np.log(np.maximum(self.startprob_, 1e-12))

        for t in range(T):
            prob_t, log_alpha = self.forward_filter_step(X_arr[t], log_alpha)
            probs[t] = prob_t

        return probs

    def get_regime_multiplier(self, probs_t: np.ndarray) -> float:
        """Computes expected macro regime multiplier kappa in [0.25, 0.50]."""
        p = np.asarray(probs_t, dtype=np.float64)
        kappa = (
            p[0] * self.REGIME_MULTIPLIERS[0]
            + p[1] * self.REGIME_MULTIPLIERS[1]
            + p[2] * self.REGIME_MULTIPLIERS[2]
        )
        return float(np.clip(kappa, 0.25, 0.50))
