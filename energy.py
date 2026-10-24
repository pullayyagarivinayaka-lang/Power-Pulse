"""FR-03 / FR-04: operating-interval detection and energy integration.

Integration method ("left rectangle"): each reading P_i (W) is assumed constant until the next
reading, so  E_kWh = sum_i (P_i / 1000) * (dt_i / 3600)  where dt_i is the seconds until the next
sample. The last sample uses the median sampling interval. Intervals longer than ``max_gap_s`` are
treated as missing data and contribute zero energy (never interpolated).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ApplianceConfig

INTEGRATION_METHOD = "left_rectangle: sum(power_kW * interval_h); gaps > max_gap_s excluded"


@dataclass(frozen=True)
class EnergyResult:
    energy_kwh: float
    covered_hours: float
    span_hours: float
    coverage: float            # covered / span in [0, 1]
    n_gap_intervals: int
    method: str = INTEGRATION_METHOD


def _seconds(index: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray((index - index[0]) / pd.Timedelta(seconds=1), dtype="float64")


def interval_seconds(t_s: np.ndarray) -> np.ndarray:
    """Seconds until the next sample for each sample; last sample gets the median interval."""
    if len(t_s) == 0:
        return np.array([], dtype="float64")
    if len(t_s) == 1:
        return np.array([0.0])
    d = np.diff(t_s)
    return np.append(d, np.median(d))


def _validate_power(power_w: np.ndarray) -> None:
    if not np.all(np.isfinite(power_w)):
        raise ValueError("power_w contains NaN/inf; clean the series first (AC-01)")
    if np.any(power_w < 0):
        raise ValueError("power_w contains negative values; clean the series first (AC-01)")


def integrate_energy_kwh(series_w: pd.Series, max_gap_s: float) -> EnergyResult:
    """Measured energy (kWh) of a power series given in watts."""
    if len(series_w) == 0:
        return EnergyResult(0.0, 0.0, 0.0, 1.0, 0)
    if not series_w.index.is_monotonic_increasing or series_w.index.has_duplicates:
        raise ValueError("timestamps must be strictly increasing (clean the series first)")
    p = series_w.to_numpy(dtype="float64")
    _validate_power(p)
    dts = interval_seconds(_seconds(series_w.index))
    valid = dts <= max_gap_s
    covered_s = float(dts[valid].sum())
    span_s = float(dts.sum())
    energy = float(np.sum(p[valid] * dts[valid]) / 3.6e6)       # W*s -> kWh
    return EnergyResult(energy, covered_s / 3600.0, span_s / 3600.0,
                        covered_s / span_s if span_s > 0 else 1.0, int((~valid).sum()))


def sample_energy_kwh(series_w: pd.Series, max_gap_s: float) -> pd.Series:
    """Energy (kWh) attributed to each individual sample (0 where the following gap is too long)."""
    p = series_w.to_numpy(dtype="float64")
    _validate_power(p)
    dts = interval_seconds(_seconds(series_w.index))
    dts = np.where(dts <= max_gap_s, dts, 0.0)
    return pd.Series(p * dts / 3.6e6, index=series_w.index, name="energy_kwh")


SESSION_COLUMNS = ["appliance", "start_time", "end_time", "duration_h", "measured_energy_kwh",
                   "mean_power_w", "peak_power_w", "n_samples"]


def detect_sessions(series_w: pd.Series, cfg: ApplianceConfig, max_gap_s: float) -> pd.DataFrame:
    """Derive operating sessions with appliance-aware thresholds (FR-03).

    A sample is ON when power > ``on_threshold_w``. Consecutive ON samples form a run; runs are
    split at data gaps, then merged when the OFF period between them is <= ``merge_gap_s`` (and no
    data gap lies between). Sessions shorter than ``min_on_s`` are dropped. Energy of a merged
    session includes the low-power samples between its runs.
    """
    if len(series_w) < 2:
        return pd.DataFrame(columns=SESSION_COLUMNS)
    p = series_w.to_numpy(dtype="float64")
    _validate_power(p)
    t = _seconds(series_w.index)
    dts = interval_seconds(t)
    valid = dts <= max_gap_s
    dts_v = np.where(valid, dts, 0.0)
    gap_idx = np.flatnonzero(~valid)

    on = p > cfg.on_threshold_w
    if not on.any():
        return pd.DataFrame(columns=SESSION_COLUMNS)
    pad = np.concatenate([[0], on.astype(np.int8), [0]])
    edges = np.diff(pad)
    run_starts, run_ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1  # inclusive

    segments: list[tuple[int, int]] = []
    for a, b in zip(run_starts, run_ends):
        lo, hi = np.searchsorted(gap_idx, a, "left"), np.searchsorted(gap_idx, b, "left")
        seg_start = a
        for g in gap_idx[lo:hi]:                 # split run where the interval after sample g is a gap
            segments.append((seg_start, int(g)))
            seg_start = int(g) + 1
        segments.append((seg_start, int(b)))

    merged: list[list[int]] = []
    for a, b in segments:
        if merged:
            pa, pb = merged[-1]
            off_s = t[a] - (t[pb] + dts_v[pb])
            lo, hi = np.searchsorted(gap_idx, pb, "left"), np.searchsorted(gap_idx, a, "left")
            if hi == lo and off_s <= cfg.merge_gap_s:
                merged[-1][1] = b
                continue
        merged.append([a, b])

    cum_e = np.concatenate([[0.0], np.cumsum(p * dts_v)])   # W*s
    cum_d = np.concatenate([[0.0], np.cumsum(dts_v)])       # s
    rows = []
    for a, b in merged:
        dur_s = cum_d[b + 1] - cum_d[a]
        if dur_s < cfg.min_on_s or dur_s <= 0:
            continue
        e_ws = cum_e[b + 1] - cum_e[a]
        rows.append((cfg.name, series_w.index[a], series_w.index[b] + pd.Timedelta(seconds=float(dts_v[b])),
                     dur_s / 3600.0, e_ws / 3.6e6, e_ws / dur_s, float(p[a:b + 1].max()), int(b - a + 1)))
    return pd.DataFrame(rows, columns=SESSION_COLUMNS)


def detect_all_sessions(series: dict[str, pd.Series], cfg) -> pd.DataFrame:
    frames = [detect_sessions(s, cfg.appliances[name], cfg.dataset.max_gap_s) for name, s in series.items()]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise ValueError("No operating sessions detected for any appliance; check thresholds / units")
    return pd.concat(frames, ignore_index=True).sort_values("start_time").reset_index(drop=True)
