"""AC-04 leakage-safe split and features."""
import numpy as np
import pandas as pd
import pytest

from src.features import HISTORY, assert_no_target_leakage, build_features, feature_columns
from src.split import SplitError, assert_no_overlap, chronological_split


def _sessions(n=400):
    start = pd.Timestamp("2024-01-01", tz="UTC") + pd.to_timedelta(np.arange(n) * 3, unit="h")
    return pd.DataFrame({"appliance": ["a", "b"] * (n // 2), "start_time": start,
                         "end_time": start + pd.Timedelta(hours=1), "duration_h": 1.0,
                         "measured_energy_kwh": np.linspace(0.1, 1, n), "mean_power_w": np.linspace(100, 1000, n),
                         "peak_power_w": 1200.0, "n_samples": 60})


def test_split_is_chronological_and_non_overlapping():
    train, val, test, info, _ = chronological_split(_sessions(), 0.7, 0.15, purge_hours=24)
    assert_no_overlap(train, val, test)
    assert train["end_time"].max() < val["start_time"].min() <= val["end_time"].max() < test["start_time"].min()
    assert len(train) > len(val) and len(test) > 0
    assert (val["start_time"].min() - train["end_time"].max()) >= pd.Timedelta(hours=23)


def test_split_rejects_too_short_data():
    with pytest.raises(SplitError):
        chronological_split(_sessions(6), 0.7, 0.15, purge_hours=1000)


def test_leaky_columns_are_refused():
    with pytest.raises(ValueError, match="leaking"):
        assert_no_target_leakage(["duration_h", "mean_power_w"])
    assert_no_target_leakage(feature_columns(True))


def test_history_feature_uses_only_past_sessions():
    f = build_features(_sessions(40), {"a": 150.0, "b": 200.0}, "UTC", use_history=True, history_window=3)
    a = f[f["appliance"] == "a"].reset_index(drop=True)
    assert a.loc[0, HISTORY] == 150.0                                   # first session -> rated-power fallback
    assert a.loc[1, HISTORY] == pytest.approx(a.loc[0, "mean_power_w"])  # only the previous session
    assert a.loc[2, HISTORY] == pytest.approx(a.loc[:1, "mean_power_w"].mean())


def test_time_features_use_local_timezone():
    s = _sessions(4).assign(start_time=pd.Timestamp("2024-07-01 23:30", tz="UTC"))
    f = build_features(s, {"a": 1.0, "b": 1.0}, "Asia/Kolkata", use_history=False)   # 05:00 next day local
    assert f.loc[0, "dayofweek"] == 1 and f.loc[0, "month"] == 7
