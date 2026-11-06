"""
Continuous Event-Decay Kernel & Smart-Money Divergence Engine (QARE-MM v1.3 PRD Section 5.6 & 5.10)
Transforms asynchronous qualitative events into stationary continuous signals:
1. Decayed FinBERT Sentiment & Age (half-life tau = 3 days)
2. Decayed Form 4 Insider Clusters & Age (half-life tau = 20 days)
3. Decayed PEAD Earnings Shocks & Age (half-life tau = 10 days)
4. Decayed 10-Q Redline Delta & MD&A Catalysts & Staleness (half-life tau = 15 days)
5. Explicit Modality Missingness Masks (3 binary indicators)
Total Output Vector Dimension: exactly 12 channels.

Also computes Signed Smart-Money Divergence [D_t^+, D_t^-] for channels 64..65 of (Batch, 66).
"""

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


@dataclass
class EventObservation:
    """Represents a Point-in-Time catalyst event."""
    event_type: str  # "news", "insider", "pead", "sec_10q"
    effective_date: str  # YYYY-MM-DD post-16:00 cutoff
    magnitude: float  # Raw catalyst intensity in [-1.0, +1.0]


class EventDecayKernel:
    """
    Computes the 12-channel continuous event-decay feature vector:
    X_event(t) = X_0 * exp(-ln(2) / tau * delta_d)
    """

    NUM_CHANNELS = 12

    # Half-lives in trading days (PRD Section 5.6)
    TAU_NEWS = 3.0
    TAU_INSIDER = 20.0
    TAU_PEAD = 10.0
    TAU_SEC_10Q = 15.0

    LN2 = math.log(2.0)

    @classmethod
    def compute_decay(cls, initial_val: float, elapsed_days: float, tau: float) -> float:
        """Applies continuous exponential half-life decay."""
        if elapsed_days < 0 or tau <= 0:
            return 0.0
        decay_factor = math.exp(- (cls.LN2 / tau) * elapsed_days)
        return float(initial_val * decay_factor)

    @classmethod
    def build_event_tensor(
        cls,
        as_of_date: Union[str, pd.Timestamp],
        news_event: Optional[Tuple[float, Union[str, pd.Timestamp]]] = None,     # (sentiment, date)
        insider_event: Optional[Tuple[float, Union[str, pd.Timestamp]]] = None,  # (cluster_score, date)
        pead_event: Optional[Tuple[float, Union[str, pd.Timestamp]]] = None,     # (sue_shock, date)
        sec_10q_event: Optional[Tuple[float, float, Union[str, pd.Timestamp]]] = None, # (risk_delta, mda_score, date)
        max_age_cap: float = 63.0,
    ) -> np.ndarray:
        """
        Builds the exact 12-channel event feature vector as of as_of_date.
        Respects Point-in-Time causality: events occurring strictly after as_of_date are excluded.
        """
        current_dt = pd.Timestamp(as_of_date)
        features = np.zeros(cls.NUM_CHANNELS, dtype=np.float32)

        # Modality Missingness Masks: default 1.0 (missing)
        mask_news = 1.0
        mask_insider = 1.0
        mask_10q = 1.0

        # Channel 0-1: FinBERT Sentiment & Age
        if news_event is not None:
            score, dt_str = news_event
            evt_dt = pd.Timestamp(dt_str)
            if evt_dt <= current_dt:
                elapsed = max(0.0, (current_dt - evt_dt).days)
                if elapsed <= max_age_cap:
                    features[0] = cls.compute_decay(score, elapsed, cls.TAU_NEWS)
                    features[1] = elapsed
                    mask_news = 0.0

        # Channel 2-3: Form 4 Insider Cluster & Age
        if insider_event is not None:
            score, dt_str = insider_event
            evt_dt = pd.Timestamp(dt_str)
            if evt_dt <= current_dt:
                elapsed = max(0.0, (current_dt - evt_dt).days)
                if elapsed <= max_age_cap:
                    features[2] = cls.compute_decay(score, elapsed, cls.TAU_INSIDER)
                    features[3] = elapsed
                    mask_insider = 0.0

        # Channel 4-5: PEAD Earnings Shock & Age
        if pead_event is not None:
            sue, dt_str = pead_event
            evt_dt = pd.Timestamp(dt_str)
            if evt_dt <= current_dt:
                elapsed = max(0.0, (current_dt - evt_dt).days)
                if elapsed <= max_age_cap:
                    features[4] = cls.compute_decay(sue, elapsed, cls.TAU_PEAD)
                    features[5] = elapsed

        # Channel 6-8: 10-Q Item 1A Risk Delta, MD&A Catalyst Score & Staleness Age
        if sec_10q_event is not None:
            risk_delta, mda_score, dt_str = sec_10q_event
            evt_dt = pd.Timestamp(dt_str)
            if evt_dt <= current_dt:
                elapsed = max(0.0, (current_dt - evt_dt).days)
                if elapsed <= max_age_cap:
                    features[6] = cls.compute_decay(risk_delta, elapsed, cls.TAU_SEC_10Q)
                    features[7] = cls.compute_decay(mda_score, elapsed, cls.TAU_SEC_10Q)
                    features[8] = elapsed
                    mask_10q = 0.0

        # Channel 9-11: Modality Missingness Masks
        features[9] = mask_news
        features[10] = mask_insider
        features[11] = mask_10q

        return features

    @classmethod
    def compute_smart_money_divergence(
        cls,
        sentiment: float,
        residual_return: float,
        volume: float,
        adv20: float,
    ) -> Tuple[float, float]:
        """
        Computes Signed Smart-Money Divergence [D_t^+, D_t^-] (PRD Section 5.10):
        D_t^+: Media Euphoria into Institutional Selling / Distribution (S > 0 and epsilon < 0)
        D_t^-: Media Panic into Institutional Buying / Accumulation (S < 0 and epsilon > 0)
        """
        vol_ratio = volume / max(adv20, 1e-5)

        if sentiment > 0.0 and residual_return < 0.0:
            d_plus = float(sentiment * (-residual_return) * vol_ratio)
            d_minus = 0.0
        elif sentiment < 0.0 and residual_return > 0.0:
            d_plus = 0.0
            d_minus = float((-sentiment) * residual_return * vol_ratio)
        else:
            d_plus = 0.0
            d_minus = 0.0

        return float(d_plus), float(d_minus)
