"""Leakage-safe feature engineering for session-level energy regression.

Features are limited to things known *before/at the start* of a session plus the declared duration:
appliance, duration, rated power, rated-power baseline, time-of-day / weekday / month, and (optional)
the mean power of the appliance's PREVIOUS sessions. Measured mean/peak power of the session being
predicted are deliberately NOT features (they would leak the target: energy = mean_power * duration).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .baseline import add_baseline

TARGET = "measured_energy_kwh"
CATEGORICAL = ["appliance"]
NUMERIC_BASE = ["duration_h", "rated_power_w", "baseline_kwh", "hour_sin", "hour_cos",
                "dayofweek", "is_weekend", "month"]
HISTORY = "hist_mean_power_w"


def feature_columns(use_history: bool) -> list[str]:
    return CATEGORICAL + NUMERIC_BASE + ([HISTORY] if use_history else [])


def time_features(start_times: pd.Series, tz: str) -> pd.DataFrame:
    local = pd.DatetimeIndex(start_times).tz_convert(tz)
    hour = local.hour + local.minute / 60.0
    return pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * hour / 24.0), "hour_cos": np.cos(2 * np.pi * hour / 24.0),
        "dayofweek": local.dayofweek.astype(int), "is_weekend": (local.dayofweek >= 5).astype(int),
        "month": local.month.astype(int)}, index=start_times.index)


def build_features(sessions: pd.DataFrame, rated_power_w: dict[str, float], tz: str,
                   use_history: bool = True, history_window: int = 20) -> pd.DataFrame:
    df = add_baseline(sessions.sort_values("start_time").reset_index(drop=True), rated_power_w)
    df = pd.concat([df, time_features(df["start_time"], tz)], axis=1)
    if use_history:
        # mean power of the previous N sessions of the same appliance (strictly earlier -> shift(1))
        df[HISTORY] = (df.groupby("appliance", sort=False)["mean_power_w"]
                         .transform(lambda s: s.shift(1).rolling(history_window, min_periods=1).mean()))
        df[HISTORY] = df[HISTORY].fillna(df["rated_power_w"])   # first session: fall back to rated power
    return df


def assert_no_target_leakage(cols: list[str]) -> None:
    banned = {TARGET, "mean_power_w", "peak_power_w", "n_samples", "end_time"}
    leaked = banned.intersection(cols)
    if leaked:
        raise ValueError(f"Target-leaking feature(s) in feature set: {sorted(leaked)}")
