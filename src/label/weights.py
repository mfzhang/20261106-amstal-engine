"""
Intra-Ticker Sample Uniqueness Weighting & Concurrency
Computes average inverse concurrency weights to penalize overlapping barrier observations.
"""

from typing import List, Tuple
import numpy as np

class UniquenessWeighter:
    """Calculates intra-ticker concurrency and mean-normalized uniqueness weights."""

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
            concurrency[t_start : min(t_end + 1, total_timeline_length)] += 1
        return np.maximum(concurrency, 1)

    @classmethod
    def compute_uniqueness_weights(
        cls,
        total_timeline_length: int,
        observation_intervals: List[Tuple[int, int]],
    ) -> np.ndarray:
        """
        Computes u_i,t = (1 / L) * sum_{tau=t}^{t_end} (1 / c_tau).
        Mean-normalizes across the sample fold so sum(u) / N = 1.0.
        """
        concurrency = cls.compute_concurrency(total_timeline_length, observation_intervals)
        inv_concurrency = 1.0 / concurrency
        
        weights = []
        for t_start, t_end in observation_intervals:
            end_bound = min(t_end + 1, total_timeline_length)
            length = max(end_bound - t_start, 1)
            u_it = np.mean(inv_concurrency[t_start:end_bound])
            weights.append(u_it)

        weights_arr = np.array(weights, dtype=float)
        if len(weights_arr) == 0:
            return weights_arr

        # Mean-normalization to preserve gradient / learning rate scale
        mean_weight = np.mean(weights_arr)
        if mean_weight > 0:
            weights_arr = weights_arr / mean_weight
        return weights_arr
