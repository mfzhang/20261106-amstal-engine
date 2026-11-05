"""
Intra-Ticker Sample Uniqueness Weighting & Concurrency (QARE-MM v1.3)
Computes average inverse concurrency weights to penalize overlapping barrier observations
and enforces cross-ticker calendar-date purging and embargo boundaries:
- Intra-ticker uniqueness: u_{i, t} = (1 / L) * sum_{tau=t}^{t_end} (1 / c_{i, tau})
- Fold mean-normalization: (1 / N) * sum(u_{i, t}) = 1.0 to preserve gradient scale
- Purging: drops training observations whose barrier evaluation window crosses split_date
- Embargo: enforces a 10-day buffer gap before out-of-sample evaluation begins
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class UniquenessWeighter:
    """Calculates intra-ticker concurrency, mean-normalized uniqueness weights, and purge/embargo splits."""

    @staticmethod
    def compute_concurrency(
        total_timeline_length: int,
        observation_intervals: List[Tuple[int, int]],
    ) -> np.ndarray:
        """
        Computes concurrency array c_t counting active barrier intervals at each timeline step.
        """
        concurrency = np.zeros(total_timeline_length, dtype=int)
        for t_start, t_end in observation_intervals:
            s = max(0, int(t_start))
            e = min(int(t_end) + 1, total_timeline_length)
            if e > s:
                concurrency[s:e] += 1
        return np.maximum(concurrency, 1)

    @classmethod
    def compute_uniqueness_weights(
        cls,
        total_timeline_length: int,
        observation_intervals: List[Tuple[int, int]],
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Computes u_i,t = (1 / L) * sum_{tau=t}^{t_end} (1 / c_tau).
        Optionally mean-normalizes across the sample fold so sum(u) / N = 1.0.
        """
        if not observation_intervals:
            return np.empty(0, dtype=np.float64)

        concurrency = cls.compute_concurrency(total_timeline_length, observation_intervals)
        inv_concurrency = 1.0 / concurrency

        weights = []
        for t_start, t_end in observation_intervals:
            s = max(0, int(t_start))
            e = min(int(t_end) + 1, total_timeline_length)
            length = max(e - s, 1)
            u_it = float(np.mean(inv_concurrency[s:e]))
            weights.append(u_it)

        weights_arr = np.array(weights, dtype=np.float64)

        if normalize and len(weights_arr) > 0:
            mean_w = np.mean(weights_arr)
            if mean_w > 0:
                weights_arr = weights_arr / mean_w

        return weights_arr

    @classmethod
    def weight_labeled_df(
        cls,
        labeled_df: pd.DataFrame,
        normalize: bool = True,
    ) -> pd.DataFrame:
        """
        Takes a DataFrame output from TripleBarrierLabeler.label_series,
        computes intra-ticker uniqueness weights, and attaches 'uniqueness_weight'.
        """
        df = labeled_df.copy()
        if df.empty:
            df["uniqueness_weight"] = np.empty(0, dtype=np.float64)
            return df

        if "start_idx" in df.columns and "end_idx" in df.columns:
            intervals = list(zip(df["start_idx"].values, df["end_idx"].values))
            max_idx = int(df["end_idx"].max()) + 2
        else:
            # Map index positions to integer timeline offsets
            dates = sorted(list(set(df["t_start"].tolist() + df["t_end"].tolist())))
            date_to_idx = {d: i for i, d in enumerate(dates)}
            intervals = [(date_to_idx[r["t_start"]], date_to_idx[r["t_end"]]) for _, r in df.iterrows()]
            max_idx = len(dates) + 2

        weights = cls.compute_uniqueness_weights(max_idx, intervals, normalize=normalize)
        df["uniqueness_weight"] = weights
        return df

    @staticmethod
    def purge_and_embargo(
        labeled_df: pd.DataFrame,
        split_date: Union[str, pd.Timestamp],
        embargo_days: int = 10,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Strict Cross-Ticker Calendar-Date Purging & Embargo Splitting.
        - Training set: All observations where t_end < split_date (purges any training sample
          whose barrier horizon extends into or across split_date to eliminate forward-looking leak).
        - Test set: All observations where t_start >= split_date + embargo_days (enforces 10-day
          buffer embargo so feature lookback cannot contaminate test fold).
        """
        split_dt = pd.to_datetime(split_date)
        if labeled_df.empty:
            return labeled_df.copy(), labeled_df.copy()

        df = labeled_df.copy()
        # Guarantee t_start and t_end are DatetimeIndex compatible
        t_start_col = pd.to_datetime(df["t_start"] if "t_start" in df.columns else df.index)
        t_end_col = pd.to_datetime(df["t_end"])

        # Purging rule: Train samples must strictly resolve BEFORE split_date
        train_mask = t_end_col < split_dt
        train_df = df[train_mask].copy()

        # Embargo rule: Test samples must start AFTER split_date + embargo_days
        # If business days, shift by business days; otherwise timedelta
        embargo_dt = split_dt + pd.Timedelta(days=embargo_days)
        test_mask = t_start_col >= embargo_dt
        test_df = df[test_mask].copy()

        return train_df, test_df
