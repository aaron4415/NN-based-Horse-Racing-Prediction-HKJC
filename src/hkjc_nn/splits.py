"""Time-based splitting.

The report is emphatic: **do not randomly split**.  A random split puts the
same horse — and often the same race meeting — into both the training and the
test set, which inflates the score and hides overfitting.  Every split in
this project is therefore a cut on the calendar.

Races are assigned to exactly one split, and the boundaries are chosen so that
the test set is *later* than everything the model has seen.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimeSplits:
    """Boolean masks over the rows of a frame."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    train_end: pd.Timestamp
    val_end: pd.Timestamp

    def sizes(self) -> dict[str, int]:
        return {
            "train": int(self.train.sum()),
            "val": int(self.val.sum()),
            "test": int(self.test.sum()),
        }

    def check_disjoint(self) -> None:
        total = self.train.sum() + self.val.sum() + self.test.sum()
        if total != len(self.train):
            raise AssertionError("splits overlap or do not cover every row")
        if self.train.any() and self.val.any():
            assert not (self.train & self.val).any()


def time_splits(
    frame: pd.DataFrame,
    *,
    train_until: str = "2022-12-31",
    val_until: str = "2023-12-31",
    date_column: str = "meeting_date",
) -> TimeSplits:
    """Cut a frame into train / validation / test on the calendar.

    * train:  ``meeting_date <= train_until``
    * val:    ``train_until < meeting_date <= val_until``
    * test:   ``meeting_date > val_until``
    """
    dates = pd.to_datetime(frame[date_column])
    train_end = pd.Timestamp(train_until)
    val_end = pd.Timestamp(val_until)

    train = (dates <= train_end).to_numpy()
    val = ((dates > train_end) & (dates <= val_end)).to_numpy()
    test = (dates > val_end).to_numpy()

    splits = TimeSplits(train=train, val=val, test=test, train_end=train_end, val_end=val_end)
    splits.check_disjoint()
    return splits


def describe_splits(frame: pd.DataFrame, splits: TimeSplits, date_column: str = "meeting_date") -> str:
    """Human-readable summary of the split boundaries."""
    dates = pd.to_datetime(frame[date_column])
    lines = []
    for name, mask in (("train", splits.train), ("val", splits.val), ("test", splits.test)):
        if not mask.any():
            lines.append(f"{name:5s}  (empty)")
            continue
        sub = dates[mask]
        lines.append(
            f"{name:5s}  rows={int(mask.sum()):6d}  "
            f"dates {sub.min().date()} .. {sub.max().date()}"
        )
    return "\n".join(lines)
