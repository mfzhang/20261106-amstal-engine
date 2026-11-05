"""
Locked Walk-Forward Cross-Validation Geometry
Enforces:
[246d Warm-Up] -> [756d Train] -> [10d Purge] -> [126d Val] -> [10d Purge] -> [63d OOS Test]
"""

from dataclasses import dataclass
from typing import List, Generator
import numpy as np

@dataclass(frozen=True)
class FoldIndices:
    fold_id: int
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    train_end_idx: int
    val_end_idx: int
    test_end_idx: int

class WalkForwardSplitter:
    """Generates walk-forward splits strictly purging observations crossing boundary cutoffs."""

    WARMUP_DAYS: int = 246
    TRAIN_DAYS: int = 756
    PURGE_BUFFER: int = 10
    VAL_DAYS: int = 126
    TEST_DAYS: int = 63

    def __init__(self, step_days: int = 63) -> None:
        self.step_days = step_days

    def split(
        self,
        total_days: int,
        barrier_end_indices: np.ndarray,
    ) -> Generator[FoldIndices, None, None]:
        """
        Generates purged folds stepping forward by step_days.
        Any sample whose barrier_end_idx > cutoff is purged from that partition.
        """
        required_minimum = (
            self.WARMUP_DAYS + self.TRAIN_DAYS + self.PURGE_BUFFER +
            self.VAL_DAYS + self.PURGE_BUFFER + self.TEST_DAYS
        )
        if total_days < required_minimum:
            raise ValueError(f"Timeline length {total_days} less than minimum required {required_minimum}")

        fold_id = 0
        current_train_start = self.WARMUP_DAYS

        while True:
            train_end = current_train_start + self.TRAIN_DAYS
            val_start = train_end + self.PURGE_BUFFER
            val_end = val_start + self.VAL_DAYS
            test_start = val_end + self.PURGE_BUFFER
            test_end = test_start + self.TEST_DAYS

            if test_end > total_days:
                break

            # Purge rule: t_end <= cutoff
            train_mask = (
                (np.arange(total_days) >= current_train_start) &
                (np.arange(total_days) < train_end) &
                (barrier_end_indices < train_end)
            )
            val_mask = (
                (np.arange(total_days) >= val_start) &
                (np.arange(total_days) < val_end) &
                (barrier_end_indices < val_end)
            )
            test_mask = (
                (np.arange(total_days) >= test_start) &
                (np.arange(total_days) < test_end)
            )

            yield FoldIndices(
                fold_id=fold_id,
                train_indices=np.where(train_mask)[0],
                val_indices=np.where(val_mask)[0],
                test_indices=np.where(test_mask)[0],
                train_end_idx=train_end,
                val_end_idx=val_end,
                test_end_idx=test_end,
            )

            current_train_start += self.step_days
            fold_id += 1
