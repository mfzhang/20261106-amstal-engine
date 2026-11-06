"""
Locked Walk-Forward Cross-Validation Geometry
Enforces:
[246d Warm-Up] -> [756d Train] -> [10d Purge] -> [126d Val] -> [10d Purge] -> [63d OOS Test]
"""

from dataclasses import dataclass
from typing import Generator, List, Optional
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
    """
    Generates walk-forward splits strictly purging observations crossing boundary cutoffs.
    Enforces locked geometry:
    [246d Warm-Up] -> [756d Train] -> [10d Purge] -> [126d Val] -> [10d Purge] -> [63d OOS Test]
    """

    WARMUP_DAYS: int = 246
    TRAIN_DAYS: int = 756
    PURGE_BUFFER: int = 10
    VAL_DAYS: int = 126
    TEST_DAYS: int = 63

    def __init__(
        self,
        warmup_days: int = 246,
        train_days: int = 756,
        purge_buffer: int = 10,
        val_days: int = 126,
        test_days: int = 63,
        step_days: int = 63,
    ) -> None:
        self.warmup_days = warmup_days
        self.train_days = train_days
        self.purge_buffer = purge_buffer
        self.val_days = val_days
        self.test_days = test_days
        self.step_days = step_days

    @property
    def min_required_days(self) -> int:
        return (
            self.warmup_days
            + self.train_days
            + self.purge_buffer
            + self.val_days
            + self.purge_buffer
            + self.test_days
        )

    def split_panel(
        self,
        sample_days: np.ndarray,
        barrier_end_days: np.ndarray,
        total_days: Optional[int] = None,
    ) -> Generator[FoldIndices, None, None]:
        """
        Generates purged folds for panel observations where sample_days and barrier_end_days
        denote the entry and resolution day index for each row.
        """
        sample_days = np.asarray(sample_days, dtype=int)
        barrier_end_days = np.asarray(barrier_end_days, dtype=int)

        if total_days is None:
            max_day = max(
                int(sample_days.max()) if len(sample_days) > 0 else 0,
                int(barrier_end_days.max()) if len(barrier_end_days) > 0 else 0,
            )
            total_days = max_day + 1

        if total_days < self.min_required_days:
            raise ValueError(
                f"Timeline length {total_days} less than minimum required {self.min_required_days}"
            )

        fold_id = 0
        current_train_start = self.warmup_days

        while True:
            train_end = current_train_start + self.train_days
            val_start = train_end + self.purge_buffer
            val_end = val_start + self.val_days
            test_start = val_end + self.purge_buffer
            test_end = test_start + self.test_days

            if test_end > total_days:
                break

            # Strict purge rules:
            # 1. Train samples must enter within [train_start, train_end) and resolve <= train_end
            train_mask = (
                (sample_days >= current_train_start)
                & (sample_days < train_end)
                & (barrier_end_days <= train_end)
            )
            # 2. Val samples must enter within [val_start, val_end) and resolve <= val_end
            val_mask = (
                (sample_days >= val_start)
                & (sample_days < val_end)
                & (barrier_end_days <= val_end)
            )
            # 3. Test samples must enter within [test_start, test_end)
            test_mask = (sample_days >= test_start) & (sample_days < test_end)

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

    def split(
        self,
        total_days: int,
        barrier_end_indices: np.ndarray,
    ) -> Generator[FoldIndices, None, None]:
        """
        Generates purged folds for 1D timeline index.
        """
        return self.split_panel(
            sample_days=np.arange(total_days),
            barrier_end_days=barrier_end_indices,
            total_days=total_days,
        )
