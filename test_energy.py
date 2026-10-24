"""AC-02: energy integration vs hand-calculated cases. FR-03: appliance-aware sessions."""
import numpy as np
import pandas as pd
import pytest

from src.config import ApplianceConfig
from src.energy import detect_sessions, integrate_energy_kwh, sample_energy_kwh


def _s(vals, freq="60s", start="2024-01-01"):
    return pd.Series(np.asarray(vals, float), index=pd.date_range(start, periods=len(vals), freq=freq, tz="UTC"))


def _cfg(thr=50, min_on=60, merge=0):
    return ApplianceConfig("app", "Appliance1", thr, min_on, merge, 1000, "test")


def test_1000w_for_one_hour_is_one_kwh():
    res = integrate_energy_kwh(_s([1000] * 3600, "1s"), max_gap_s=10)
    assert res.energy_kwh == pytest.approx(1.0, abs=1e-9) and res.coverage == pytest.approx(1.0)


def test_500w_for_two_hours_minute_sampling_is_one_kwh():
    assert integrate_energy_kwh(_s([500] * 120), 120).energy_kwh == pytest.approx(1.0, abs=1e-9)


def test_8_second_sampling_matches_hand_calc():
    # 250 W for 30 minutes at 8 s = 225 samples -> 0.125 kWh
    assert integrate_energy_kwh(_s([250] * 225, "8s"), 60).energy_kwh == pytest.approx(0.125, abs=1e-9)


def test_gap_longer_than_max_is_not_integrated():
    idx = pd.DatetimeIndex(list(pd.date_range("2024-01-01 00:00", periods=60, freq="60s", tz="UTC"))
                           + list(pd.date_range("2024-01-01 10:00", periods=60, freq="60s", tz="UTC")))
    res = integrate_energy_kwh(pd.Series(1000.0, index=idx), max_gap_s=120)
    # Block 1: 59 valid 1-min intervals (the last sample before the gap has unknown duration -> 0).
    # Block 2: 59 intervals + last sample (median interval 60 s) = 60. Total 119 min at 1 kW.
    assert res.energy_kwh == pytest.approx(119 / 60, abs=1e-9)
    assert res.n_gap_intervals == 1 and res.coverage < 0.2


def test_negative_power_raises():
    with pytest.raises(ValueError, match="negative"):
        integrate_energy_kwh(_s([100, -1, 100]), 120)


def test_sample_energy_sums_to_integral():
    s = _s(np.random.default_rng(0).uniform(0, 2000, 500))
    assert sample_energy_kwh(s, 120).sum() == pytest.approx(integrate_energy_kwh(s, 120).energy_kwh)


def test_session_energy_and_duration():
    vals = [0] * 10 + [2000] * 30 + [0] * 10                      # 2 kW for 30 min = 1 kWh
    sess = detect_sessions(_s(vals), _cfg(), 120)
    assert len(sess) == 1
    assert sess.loc[0, "measured_energy_kwh"] == pytest.approx(1.0, abs=1e-9)
    assert sess.loc[0, "duration_h"] == pytest.approx(0.5, abs=1e-9)
    assert sess.loc[0, "peak_power_w"] == 2000


def test_short_off_gap_merges_only_when_allowed():
    vals = [1000] * 20 + [0] * 5 + [1000] * 20                    # 5-minute OFF between runs
    assert len(detect_sessions(_s(vals), _cfg(merge=600), 120)) == 1
    assert len(detect_sessions(_s(vals), _cfg(merge=60), 120)) == 2


def test_merged_session_includes_low_power_energy_between_runs():
    vals = [1000] * 10 + [0] * 5 + [1000] * 10
    sess = detect_sessions(_s(vals), _cfg(merge=600), 120)
    assert sess.loc[0, "measured_energy_kwh"] == pytest.approx(20 / 60, abs=1e-9)   # 1 kW * 20 min


def test_sessions_shorter_than_min_on_are_dropped():
    assert detect_sessions(_s([0] * 5 + [1000] * 3 + [0] * 5), _cfg(min_on=600), 120).empty


def test_sessions_split_at_data_gaps():
    idx = pd.DatetimeIndex(list(pd.date_range("2024-01-01 00:00", periods=20, freq="60s", tz="UTC"))
                           + list(pd.date_range("2024-01-01 03:00", periods=20, freq="60s", tz="UTC")))
    sess = detect_sessions(pd.Series(1000.0, index=idx), _cfg(merge=10 ** 6), 120)
    assert len(sess) == 2
