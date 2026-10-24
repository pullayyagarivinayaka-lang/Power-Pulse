"""AC-04: chronological train/validation/test split with purge gaps (no adjacent-interval leakage)."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class SplitError(ValueError):
    pass


@dataclass(frozen=True)
class SplitInfo:
    method: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    purge_hours: float
    n_train: int
    n_val: int
    n_test: int


def chronological_split(df: pd.DataFrame, train_frac: float, val_frac: float, purge_hours: float,
                        min_sessions_per_appliance: int = 1):
    """Split by time (not randomly). Boundaries are quantiles of session start time.

    A purge window of ``purge_hours`` is removed on both sides of each boundary, and any session whose
    *end* crosses into the next period is dropped, so no session overlaps two sets.
    """
    if df.empty:
        raise SplitError("No sessions to split")
    d = df.sort_values("start_time").reset_index(drop=True)
    b1 = d["start_time"].quantile(train_frac, interpolation="nearest")
    b2 = d["start_time"].quantile(train_frac + val_frac, interpolation="nearest")
    gap = pd.Timedelta(hours=purge_hours)

    train = d[d["end_time"] <= b1 - gap]
    val = d[(d["start_time"] >= b1 + gap) & (d["end_time"] <= b2 - gap)]
    test = d[d["start_time"] >= b2 + gap]
    for name, part in (("train", train), ("validation", val), ("test", test)):
        if part.empty:
            raise SplitError(f"{name} split is empty; need a longer recording or a smaller purge window")

    counts = test.groupby("appliance").size()
    thin = [a for a in d["appliance"].unique() if counts.get(a, 0) < min_sessions_per_appliance
            or (train["appliance"] == a).sum() < min_sessions_per_appliance]
    info = SplitInfo("chronological (time-ordered) with purge gaps", str(b1 - gap), str(b1 + gap), str(b2 - gap),
                     str(b2 + gap), purge_hours, len(train), len(val), len(test))
    return train.copy(), val.copy(), test.copy(), info, thin


def assert_no_overlap(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    """Raise if any set overlaps in time with a later set (used in tests and the pipeline)."""
    if not (train["end_time"].max() < val["start_time"].min()):
        raise SplitError("train overlaps validation in time")
    if not (val["end_time"].max() < test["start_time"].min()):
        raise SplitError("validation overlaps test in time")
