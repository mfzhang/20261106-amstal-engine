"""
Microstructure Feature Extractor & Hybrid Dual-Mode Engine (QARE-MM v1.3)
Generates:
1. 3D Sequence Tensor: (Batch, 126, 15) for Attention-GRU
2. 2D Structured Tabular Summary: (Batch, 105) for Sample-Weighted LightGBM
3. Daily Cross-Sectional Rank Normalization: [-1.0, +1.0] across active universe assets
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.stats import rankdata


CHANNEL_NAMES: List[str] = [
    "daily_log_return",              # Ch 0
    "idiosyncratic_residual_return", # Ch 1
    "overnight_log_return",          # Ch 2
    "intraday_log_return",           # Ch 3
    "residual_momentum_5d",          # Ch 4
    "residual_momentum_20d",         # Ch 5
    "residual_momentum_60d",         # Ch 6
    "relative_volume_ratio",         # Ch 7
    "amihud_illiquidity_z",          # Ch 8
    "atr_volatility_ratio",          # Ch 9
    "idiosyncratic_volatility_20d",  # Ch 10
    "parkinson_volatility",          # Ch 11
    "realized_intraday_volatility",  # Ch 12 (Dual-Mode: Hourly or Rogers-Satchell)
    "closing_power",                 # Ch 13 (Dual-Mode: Hour 7/1 Volume or CLV Skew)
    "close_to_vwap_deviation",       # Ch 14 (Dual-Mode: VWAP or Typical Price)
]

SUMMARY_STATS: List[str] = [
    "current",
    "ewma_5d",
    "ewma_21d",
    "ewma_63d",
    "ewma_126d",
    "std_21d",
    "slope_21d",
]


class MicrostructureFeatureExtractor:
    """
    Computes 15 rank-normalized microstructure channels and 105-channel multi-scale summaries.
    Supports Hybrid Dual-Mode Fallback (hourly intraday bars when available, daily OHLCV proxies otherwise).
    """

    NUM_CHANNELS = 15
    SUMMARY_PER_CHANNEL = 7
    NUM_SUMMARY_FEATURES = 105

    def __init__(self, lookback_window: int = 126) -> None:
        self.lookback_window = lookback_window
        # Precomputed slope regression weights for N=21: w_k = (k - 10) / 770
        k = np.arange(21, dtype=np.float64)
        self._slope_weights_21 = (k - 10.0) / 770.0

    @staticmethod
    def rank_normalize(values: np.ndarray) -> np.ndarray:
        """
        Cross-sectional rank normalization into [-1.0, +1.0] across active universe assets on day t.
        Formula: 2 * (rank(x) - 1) / (M_valid - 1) - 1.0
        Missing values and NaNs are mapped post-ranking to 0.0 (the neutral cross-sectional median).
        """
        arr = np.asarray(values, dtype=np.float64)
        valid_mask = ~np.isnan(arr)
        n_valid = int(np.sum(valid_mask))
        if n_valid <= 1:
            return np.zeros_like(arr, dtype=np.float64)

        ranks = np.zeros_like(arr, dtype=np.float64)
        ranks[valid_mask] = rankdata(arr[valid_mask], method="average")
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

    @staticmethod
    def parkinson_volatility(
        high: np.ndarray,
        low: np.ndarray,
    ) -> np.ndarray:
        """Parkinson volatility: (ln(H / L))^2 / (4 * ln 2)."""
        hl_ratio = np.maximum(high / np.maximum(low, 1e-8), 1.0)
        return (np.log(hl_ratio) ** 2) / (4.0 * np.log(2.0))

    @staticmethod
    def amihud_illiquidity_z(
        returns: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        window: int = 60,
    ) -> np.ndarray:
        """60-day rolling Z-score of Amihud illiquidity |r_t| / (C_t * V_t)."""
        dollar_vol = close * volume + 1e-8
        illiq = np.abs(returns) / dollar_vol

        series = pd.Series(illiq)
        rolling_mean = series.rolling(window, min_periods=10).mean().values
        rolling_std = series.rolling(window, min_periods=10).std().values

        z = (illiq - rolling_mean) / (rolling_std + 1e-8)
        z = np.nan_to_num(z, nan=0.0)
        return np.clip(z, -5.0, 5.0)

    @staticmethod
    def atr_volatility_ratio(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        window: int = 14,
    ) -> np.ndarray:
        """ATR_14 / Close ratio."""
        n = len(close)
        tr = np.zeros(n, dtype=np.float64)
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]

        for i in range(n):
            hl = high[i] - low[i]
            hc = abs(high[i] - prev_close[i])
            lc = abs(low[i] - prev_close[i])
            tr[i] = max(hl, hc, lc)

        atr = pd.Series(tr).rolling(window, min_periods=1).mean().values
        return atr / (close + 1e-8)

    def compute_raw_channels_df(
        self,
        df_ohlcv: pd.DataFrame,
        residuals: pd.Series,
        hourly_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Computes the 15 raw microstructure channels for a single stock time series.
        Automatically utilizes hourly bars for Channels 12-14 if provided; otherwise applies Dual-Mode Fallback.
        """
        df = df_ohlcv.copy()
        if not isinstance(df.index, pd.DatetimeIndex) and "timestamp" in df.columns:
            df.index = pd.to_datetime(df["timestamp"])

        close = df["close"].values
        open_ = df["open"].values
        high = df["high"].values
        low = df["low"].values
        volume = df["volume"].values
        n = len(df)

        prev_close = np.roll(close, 1)
        prev_close[0] = open_[0]

        # Ch 0: Daily Log-Return
        ch0 = np.log(close / np.maximum(prev_close, 1e-8))
        ch0[0] = 0.0

        # Ch 1: Idiosyncratic Residual Return
        aligned_res = residuals.reindex(df.index).fillna(0.0).values
        ch1 = aligned_res

        # Ch 2: Overnight Log-Return: ln(Open_t / Close_{t-1})
        ch2 = np.log(open_ / np.maximum(prev_close, 1e-8))
        ch2[0] = 0.0

        # Ch 3: Intraday Session Log-Return: ln(Close_t / Open_t)
        ch3 = np.log(close / np.maximum(open_, 1e-8))

        # Ch 4-6: Residual Momentums (5d, 20d, 60d rolling sums of epsilon)
        res_series = pd.Series(ch1, index=df.index)
        ch4 = res_series.rolling(5, min_periods=1).sum().values
        ch5 = res_series.rolling(20, min_periods=1).sum().values
        ch6 = res_series.rolling(60, min_periods=1).sum().values

        # Ch 7: Relative Volume Ratio: V_t / ADV_20
        adv20 = pd.Series(volume, index=df.index).rolling(20, min_periods=1).mean().values
        ch7 = volume / (adv20 + 1e-8)

        # Ch 8: Amihud Illiquidity Z-score (60d)
        ch8 = self.amihud_illiquidity_z(ch0, close, volume, window=60)

        # Ch 9: ATR Volatility Ratio: ATR_14 / C_t
        ch9 = self.atr_volatility_ratio(high, low, close, window=14)

        # Ch 10: Idiosyncratic Volatility (20d): sqrt(252 * mean(eps^2))
        res_sq = res_series**2
        ch10 = np.sqrt(252.0 * res_sq.rolling(20, min_periods=1).mean().values)

        # Ch 11: Parkinson Volatility
        ch11 = self.parkinson_volatility(high, low)

        # Dual-Mode Channels 12-14
        has_hourly = hourly_df is not None and not hourly_df.empty
        if has_hourly:
            # Aggregate hourly metrics per calendar date
            ch12 = np.zeros(n, dtype=np.float64)
            ch13 = np.zeros(n, dtype=np.float64)
            ch14 = np.zeros(n, dtype=np.float64)
            # Group hourly by date
            hourly_dates = hourly_df.index.normalize()
            for i, dt in enumerate(df.index.normalize()):
                h_sub = hourly_df[hourly_dates == dt]
                if len(h_sub) >= 2:
                    h_ret = np.log(h_sub["close"] / h_sub["close"].shift(1)).dropna().values
                    ch12[i] = np.sqrt(np.sum(h_ret**2)) if len(h_ret) > 0 else 0.0
                    v_first = h_sub["volume"].iloc[0]
                    v_last = h_sub["volume"].iloc[-1]
                    ch13[i] = np.log((v_last + 1e-5) / (v_first + 1e-5))
                    vwap_t = (
                        np.sum(h_sub["close"] * h_sub["volume"]) / (np.sum(h_sub["volume"]) + 1e-8)
                    )
                    ch14[i] = (close[i] - vwap_t) / (vwap_t + 1e-8)
                else:
                    # Fallback for days lacking full hourly bars
                    ch12[i] = self.rogers_satchell_volatility(high[i : i + 1], low[i : i + 1], open_[i : i + 1], close[i : i + 1])[0]
                    ch13[i] = self.clv_volume_skew(high[i : i + 1], low[i : i + 1], close[i : i + 1], volume[i : i + 1], adv20[i : i + 1])[0]
                    ch14[i] = self.typical_price_deviation(high[i : i + 1], low[i : i + 1], close[i : i + 1])[0]
        else:
            # Automatic Daily Fallbacks
            ch12 = self.rogers_satchell_volatility(high, low, open_, close)
            ch13 = self.clv_volume_skew(high, low, close, volume, adv20)
            if "vwap" in df.columns:
                ch14 = (close - df["vwap"].values) / (df["vwap"].values + 1e-8)
            else:
                ch14 = self.typical_price_deviation(high, low, close)

        data_dict = {
            CHANNEL_NAMES[0]: ch0,
            CHANNEL_NAMES[1]: ch1,
            CHANNEL_NAMES[2]: ch2,
            CHANNEL_NAMES[3]: ch3,
            CHANNEL_NAMES[4]: ch4,
            CHANNEL_NAMES[5]: ch5,
            CHANNEL_NAMES[6]: ch6,
            CHANNEL_NAMES[7]: ch7,
            CHANNEL_NAMES[8]: ch8,
            CHANNEL_NAMES[9]: ch9,
            CHANNEL_NAMES[10]: ch10,
            CHANNEL_NAMES[11]: ch11,
            CHANNEL_NAMES[12]: ch12,
            CHANNEL_NAMES[13]: ch13,
            CHANNEL_NAMES[14]: ch14,
        }
        return pd.DataFrame(data_dict, index=df.index)

    def rank_normalize_panel(
        self,
        panel_raw_dict: Dict[str, pd.DataFrame],
    ) -> Dict[str, pd.DataFrame]:
        """
        Cross-sectional rank normalization across all universe tickers day by day into [-1.0, +1.0].
        Missing universe assets on any date are imputed post-ranking to 0.0.
        """
        if not panel_raw_dict:
            return {}

        tickers = sorted(list(panel_raw_dict.keys()))
        all_dates = sorted(
            list(set.union(*[set(df.index) for df in panel_raw_dict.values()]))
        )

        # Initialize normalized result dataframes
        normalized_dict = {
            t: pd.DataFrame(index=panel_raw_dict[t].index, columns=CHANNEL_NAMES, dtype=np.float64)
            for t in tickers
        }

        # For each channel, build a cross-sectional panel (dates x tickers)
        for col in CHANNEL_NAMES:
            panel_col = pd.DataFrame(index=all_dates, columns=tickers, dtype=np.float64)
            for t in tickers:
                if col in panel_raw_dict[t].columns:
                    panel_col[t] = panel_raw_dict[t][col]

            # Cross-sectional rank normalize row by row
            norm_matrix = np.zeros_like(panel_col.values, dtype=np.float64)
            for row_idx in range(len(all_dates)):
                row_vals = panel_col.values[row_idx, :]
                norm_matrix[row_idx, :] = self.rank_normalize(row_vals)

            norm_df = pd.DataFrame(norm_matrix, index=all_dates, columns=tickers)

            for t in tickers:
                sub_dates = panel_raw_dict[t].index
                normalized_dict[t][col] = norm_df.loc[sub_dates, t].values

        return normalized_dict

    def extract_summary_105(self, sequence_15: np.ndarray) -> np.ndarray:
        """
        Maps (126, 15) sequence matrix into 105 structured summary features for LightGBM:
        For each of 15 channels:
        1. Current cross-sectional value: x_t
        2. Short-term EWMA (span=5d)
        3. Monthly EWMA (span=21d)
        4. Quarterly EWMA (span=63d)
        5. Half-Year EWMA (span=126d)
        6. Monthly Volatility (21-day rolling standard deviation)
        7. Monthly Trend Velocity (21-day linear regression slope)
        
        Also accepts 3D tensor (Batch, 126, 15) returning (Batch, 105).
        """
        seq = np.asarray(sequence_15, dtype=np.float64)
        if seq.ndim == 3:
            return np.array([self.extract_summary_105(item) for item in seq], dtype=np.float64)

        if seq.ndim != 2 or seq.shape[1] != self.NUM_CHANNELS:
            raise ValueError(f"Expected sequence of shape (T, 15), got {seq.shape}")

        t_len, n_ch = seq.shape
        df = pd.DataFrame(seq)

        curr_vals = seq[-1, :]
        ewma5 = df.ewm(span=5).mean().iloc[-1].values
        ewma21 = df.ewm(span=21).mean().iloc[-1].values
        ewma63 = df.ewm(span=63).mean().iloc[-1].values
        ewma126 = df.ewm(span=126).mean().iloc[-1].values

        if t_len >= 21:
            std21 = df.iloc[-21:].std(ddof=1).fillna(0.0).values
            # Fast vectorized dot product: slope = sum(w * y)
            slope21 = np.sum(df.iloc[-21:].values * self._slope_weights_21[:, None], axis=0)
        else:
            std21 = np.zeros(n_ch, dtype=np.float64)
            slope21 = np.zeros(n_ch, dtype=np.float64)

        # Stack into 105 features: for each channel, 7 statistics
        stacked = np.column_stack([curr_vals, ewma5, ewma21, ewma63, ewma126, std21, slope21])
        return stacked.flatten()

    def build_sequence_tensor(
        self,
        normalized_df: pd.DataFrame,
    ) -> np.ndarray:
        """
        Converts normalized 15-channel DataFrame into sliding windows of shape (N, 126, 15)
        for Attention-GRU consumption.
        """
        values = normalized_df[CHANNEL_NAMES].values
        t_len = len(values)
        w = self.lookback_window
        if t_len < w:
            return np.empty((0, w, self.NUM_CHANNELS), dtype=np.float64)

        num_samples = t_len - w + 1
        tensor = np.zeros((num_samples, w, self.NUM_CHANNELS), dtype=np.float64)
        for i in range(num_samples):
            tensor[i] = values[i : i + w, :]
        return tensor

    def build_lgbm_features(
        self,
        normalized_df: pd.DataFrame,
    ) -> np.ndarray:
        """
        Converts normalized 15-channel DataFrame into multi-scale LightGBM features (N, 105).
        """
        tensor = self.build_sequence_tensor(normalized_df)
        if len(tensor) == 0:
            return np.empty((0, self.NUM_SUMMARY_FEATURES), dtype=np.float64)
        return self.extract_summary_105(tensor)
