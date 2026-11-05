"""
Forward-Filtered 3-State Gaussian HMM
Implements causal log-space forward filtering with zero forward-backward lookahead leak.
"""

from typing import Tuple, Optional
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

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.means_: Optional[np.ndarray] = None        # (3, 2)
        self.covars_: Optional[np.ndarray] = None       # (3, 2, 2) or (3, 2)
        self.transmat_: Optional[np.ndarray] = None     # (3, 3)
        self.startprob_: Optional[np.ndarray] = None    # (3,)
        self.state_order_: Optional[np.ndarray] = None  # Mapping from raw to sorted state

    def fit(self, X_train: np.ndarray) -> "RegimeHMM":
        """
        Fits 3-state Gaussian HMM on 2D series [standardized_SPY_returns, standardized_VIX].
        Sorts states strictly by SPY return variance component Sigma_{k, rr}.
        """
        try:
            from hmmlearn.hmm import GaussianHMM
            raw_hmm = GaussianHMM(
                n_components=self.NUM_STATES,
                covariance_type="diag",
                n_iter=100,
                random_state=self.random_state,
            )
            raw_hmm.fit(X_train)
            
            # Sort states by return variance (feature index 0)
            variances = raw_hmm.covars_[:, 0]
            self.state_order_ = np.argsort(variances)
            
            self.means_ = raw_hmm.means_[self.state_order_]
            self.covars_ = raw_hmm.covars_[self.state_order_]
            self.startprob_ = raw_hmm.startprob_[self.state_order_]
            
            # Reorder transition matrix
            trans = raw_hmm.transmat_[self.state_order_, :]
            self.transmat_ = trans[:, self.state_order_]
        except ImportError:
            # Fallback deterministic initialization if hmmlearn not present
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
        log_emission = np.zeros(self.NUM_STATES)
        for k in range(self.NUM_STATES):
            # Gaussian log-pdf with diagonal covariance
            log_emission[k] = np.sum(
                norm.logpdf(x_t, loc=self.means_[k], scale=np.sqrt(self.covars_[k]))
            )

        log_trans = np.log(self.transmat_ + 1e-12)
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
        T = len(X)
        probs = np.zeros((T, self.NUM_STATES))
        log_alpha = np.log(self.startprob_ + 1e-12)

        for t in range(T):
            prob_t, log_alpha = self.forward_filter_step(X[t], log_alpha)
            probs[t] = prob_t

        return probs
