"""FR-01: load appliance-level power readings + channel->appliance metadata.

Every loader returns ``dict[appliance_name -> pd.Series]`` of power in **watts**, indexed by a
tz-aware UTC DatetimeIndex. Channel names are resolved through the config mapping, never guessed
from the channel number (AC-03).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import AppConfig, ConfigError

log = logging.getLogger(__name__)


class IngestError(ValueError):
    """Raised when the raw file does not match the expected schema."""


def _to_utc_index(df: pd.DataFrame, cfg: AppConfig) -> pd.DatetimeIndex:
    d = cfg.dataset
    if d.unix_col and d.unix_col in df.columns:
        # REFIT 'Unix' column = seconds since epoch (UTC)
        return pd.DatetimeIndex(pd.to_datetime(df[d.unix_col], unit="s", utc=True))
    if d.timestamp_col not in df.columns:
        raise IngestError(f"Neither '{d.unix_col}' nor '{d.timestamp_col}' found in file columns: {list(df.columns)[:15]}")
    ts = pd.to_datetime(df[d.timestamp_col], errors="coerce")
    if ts.isna().any():
        raise IngestError(f"{int(ts.isna().sum())} timestamps in '{d.timestamp_col}' could not be parsed")
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(d.timezone, ambiguous="NaT", nonexistent="NaT")
    return pd.DatetimeIndex(ts.dt.tz_convert("UTC"))


def load_refit_wide(csv_path: str | Path, cfg: AppConfig) -> tuple[dict[str, pd.Series], dict]:
    """Load a REFIT-style wide CSV (Time, Unix, Aggregate, Appliance1..N, Issues)."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise IngestError(f"Raw data file not found: {csv_path}")
    d = cfg.dataset
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    missing = [a.channel for a in cfg.appliances.values() if a.channel not in header]
    if missing:
        raise IngestError(f"Mapped channel(s) {missing} not present in {csv_path.name}. Columns found: {header}")

    time_cols = [c for c in (d.unix_col, d.timestamp_col) if c and c in header]
    if not time_cols:
        raise IngestError(f"No timestamp column ('{d.unix_col}' / '{d.timestamp_col}') in {csv_path.name}")
    usecols = list(dict.fromkeys(time_cols + [a.channel for a in cfg.appliances.values()]
                                 + ([d.issues_col] if d.issues_col and d.issues_col in header else [])))
    dtypes = {a.channel: "float32" for a in cfg.appliances.values()}
    df = pd.read_csv(csv_path, usecols=usecols, dtype=dtypes)
    index = _to_utc_index(df, cfg)

    n_issue_rows = 0
    issue_mask = None
    if d.issues_col and d.issues_col in df.columns:
        issue_mask = (pd.to_numeric(df[d.issues_col], errors="coerce").fillna(0).to_numpy() > 0)
        n_issue_rows = int(issue_mask.sum())

    series: dict[str, pd.Series] = {}
    for name, a in cfg.appliances.items():
        values = pd.to_numeric(df[a.channel], errors="coerce").to_numpy(dtype="float64")
        if issue_mask is not None:
            values = np.where(issue_mask, np.nan, values)  # rows flagged by the dataset authors
        series[name] = pd.Series(values, index=index, name=name)

    meta = {"file": csv_path.name, "n_rows": int(len(df)), "n_issue_rows_flagged_by_source": n_issue_rows,
            "channels": {n: a.channel for n, a in cfg.appliances.items()}, "units": "W"}
    if not d.mapping_verified:
        log.warning("Channel mapping is NOT marked verified in the config. Verify against the dataset "
                    "metadata before trusting appliance labels (AC-03).")
    return series, meta


def load_long(csv_path: str | Path, cfg: AppConfig) -> tuple[dict[str, pd.Series], dict]:
    """Load a long-format CSV with columns: timestamp, appliance, power_w (watts)."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise IngestError(f"Raw data file not found: {csv_path}")
    d = cfg.dataset
    df = pd.read_csv(csv_path)
    for col in (d.timestamp_col, d.appliance_col, d.power_col):
        if col not in df.columns:
            raise IngestError(f"Column '{col}' missing in {csv_path.name}; found {list(df.columns)}")
    index = _to_utc_index(df, cfg)
    df = df.assign(_ts=index)
    series: dict[str, pd.Series] = {}
    for name in cfg.appliances:
        sub = df[df[d.appliance_col] == name]
        if sub.empty:
            raise IngestError(f"Appliance '{name}' has no rows in {csv_path.name}")
        series[name] = pd.Series(pd.to_numeric(sub[d.power_col], errors="coerce").to_numpy(float),
                                 index=pd.DatetimeIndex(sub["_ts"]), name=name)
    meta = {"file": csv_path.name, "n_rows": int(len(df)), "units": "W",
            "channels": {n: n for n in cfg.appliances}}
    return series, meta


def load_raw(csv_path: str | Path, cfg: AppConfig) -> tuple[dict[str, pd.Series], dict]:
    if cfg.dataset.format == "refit_wide":
        return load_refit_wide(csv_path, cfg)
    if cfg.dataset.format == "long":
        return load_long(csv_path, cfg)
    raise ConfigError(f"Unsupported dataset.format: {cfg.dataset.format}")
