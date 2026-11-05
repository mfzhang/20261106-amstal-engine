"""
Microstructure Feature Extractor & Hybrid Dual-Mode Engine
Produces:
1. 3D Sequence Tensor: (Batch, 126, 15)
2. 2D Structured Tabular Summary: (Batch, 105)
"""

from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from scipy.stats import rankdata

class MicrostructureFeatureExtractor:
    """Computes 15 microstructure channels and multi-scale summaries."""

    NUM_CHANNELS = 15
    SUMMARY_PER_CHANNEL = 7
    NUM_SUMMARY_FEATURES = 105

    def __init__(self, lookback_window: int = 126) -> None:
        self.lookback_window = lookback_window

    @staticmethod
    def rank_normalize(values: np.ndarray) -> np.ndarray:
        """Cross-sectional rank normalization into [-1.0, +1.0]."""
        valid_mask = ~np.isnan(values)
        n_valid = np.sum(valid_mask)
        if n_valid <= 1:
            return np.zeros_like(values)
        
        ranks = np.zeros_like(values, dtype=float)
        ranks[valid_mask] = rankdata(values[valid_mask], method="average")
        # Formula: 2 * (rank - 1) / (N - 1) - 1.0
        normalized = 2.0 * (ranks - 1.0) / (n_valid - 1.0) - 1.0
        normalized[~valid_mask] = 0.0
        return normalized

    @staticmethod
    def rogers_satchell_volatility(
        high: np.ndarray,
        low: np.ndarray,
        open_: np.ndarray,
        close: np.ndarray,
    ) -> np.ndarray:
        """Rogers-Satchell volatility proxy for daily OHLCV bars."""
        term1 = np.log(high / close) * np.log(high / open_)
        term2 = np.log(low / close) * np.log(low / open_)
        return np.sqrt(np.maximum(term1 + term2, 0.0))

    @staticmethod
    def clv_volume_skew(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        adv20: np.ndarray,
    ) -> np.ndarray:
        """Volume-scaled Close Location Value (CLV) skew fallback."""
        clv = (2.0 * close - high - low) / (high - low + 1e-8)
        vol_ratio = np.log(volume / (adv20 + 1e-5) + 1e-5)
        return clv * vol_ratio

    @staticmethod
    def typical_price_deviation(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> np.ndarray:
        """Deviation of close from typical price (H+L+C)/3."""
        tp = (high + low + close) / 3.0
        return (close - tp) / (tp + 1e-8)

    def extract_summary_105(self, sequence_15: np.ndarray) -> np.ndarray:
        """
        Maps (126, 15) sequence matrix into 105 structured summary features:
        For each of 15 channels:
        1. Current value x_t
        2. EWMA(span=5)
        3. EWMA(span=21)
        4. EWMA(span=63)
        5. EWMA(span=126)
        6. Rolling standard deviation (21d)
        7. Rolling linear trend slope (21d)
        """
        T, C = sequence_15.shape
        summaries = []

        for ch in range(C):
            series = pd.Series(sequence_15[:, ch])
            curr_val = series.iloc[-1]
            ewma5 = series.ewm(span=5).mean().iloc[-1]
            ewma21 = series.ewm(span=21).mean().iloc[-1]
            ewma63 = series.ewm(span=63).mean().iloc[-1]
            ewma126 = series.ewm(span=126).mean().iloc[-1]
            std21 = series.iloc[-21:].std() if T >= 21 else 0.0
            
            # 21-day OLS slope
            if T >= 21:
                y = series.iloc[-21:].values
                x = np.arange(21)
                slope = np.polyfit(x, y, 1)[0]
            else:
                slope = 0.0

            summaries.extend([
                curr_val,
                ewma5,
                ewma21,
                ewma63,
                ewma126,
                0.0 if np.isnan(std21) else std21,
                0.0 if np.isnan(slope) else slope,
            ])

        return np.array(summaries, dtype=float)
