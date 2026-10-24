"""AC-01: invalid values rejected/handled by documented rules; FR-02 validation."""
import numpy as np
import pandas as pd
import pytest

from src.preprocess import DataQualityError, clean_power_series


def _series(vals, freq="60s", tz="UTC"):
    return pd.Series(vals, index=pd.date_range("2024-01-01", periods=len(vals), freq=freq, tz=tz), dtype="float64")


def test_negative_nan_and_high_values_are_dropped_and_counted():
    s = _series([100] * 95 + [-5, np.nan, 50000, 100, 100])
    out, rep = clean_power_series(s, "x", 300, 20000, 0.05)
    assert rep["n_negative"] == 1 and rep["n_nan_or_non_numeric"] == 1 and rep["n_above_max_valid"] == 1
    assert (out >= 0).all() and len(out) == 97


def test_too_many_invalid_readings_rejects_series():
    s = _series([100] * 80 + [-1] * 20)
    with pytest.raises(DataQualityError, match="invalid"):
        clean_power_series(s, "x", 300, 20000, 0.05)


def test_duplicates_removed_and_sorted():
    s = _series([10, 20, 30, 40])
    s = pd.concat([s, s.iloc[[1]]]).iloc[[3, 0, 1, 4, 2]]
    out, rep = clean_power_series(s, "x", 300, 20000, 0.5)
    assert rep["n_duplicate_timestamps"] == 1
    assert out.index.is_monotonic_increasing and not out.index.has_duplicates


def test_timezone_naive_timestamps_rejected():
    s = _series([1, 2, 3], tz=None)
    with pytest.raises(DataQualityError, match="timezone"):
        clean_power_series(s, "x", 300, 20000, 0.05)


def test_gap_is_reported():
    idx = pd.DatetimeIndex(list(pd.date_range("2024-01-01 00:00", periods=10, freq="60s", tz="UTC"))
                           + list(pd.date_range("2024-01-01 05:00", periods=10, freq="60s", tz="UTC")))
    out, rep = clean_power_series(pd.Series(100.0, index=idx), "x", 300, 20000, 0.05)
    assert rep["n_gaps_over_max"] == 1 and rep["longest_gap_h"] > 4 and rep["coverage_pct"] < 10
