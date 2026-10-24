"""FR-02 / AC-01: validate and clean power series, and produce a data-quality report.

Rules (documented, applied in this order, all counted in the report):
  1. Non-numeric or NaN readings            -> dropped
  2. Negative readings (W)                  -> dropped (physically invalid for these loads)
  3. Readings above ``max_valid_w`` (W)     -> dropped (implausible / sensor fault)
  4. Duplicate timestamps                   -> first occurrence kept
  5. Out-of-order timestamps                -> sorted
If more than ``max_invalid_fraction`` of readings are invalid the series is REJECTED (no silent defaults).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class DataQualityError(ValueError):
    """Raised when a series has too many invalid readings to be used."""


def clean_power_series(s: pd.Series, name: str, max_gap_s: float, max_valid_w: float,
                       max_invalid_fraction: float) -> tuple[pd.Series, dict]:
    if not isinstance(s.index, pd.DatetimeIndex):
        raise DataQualityError(f"{name}: index must be a DatetimeIndex")
    if s.index.tz is None:
        raise DataQualityError(f"{name}: timestamps must be timezone-aware (UTC) to avoid DST ambiguity")

    n_raw = int(len(s))
    if n_raw == 0:
        raise DataQualityError(f"{name}: no readings")
    vals = pd.to_numeric(s, errors="coerce")
    nan_mask = vals.isna()
    neg_mask = (vals < 0) & ~nan_mask
    high_mask = (vals > max_valid_w) & ~nan_mask
    bad = nan_mask | neg_mask | high_mask
    n_invalid = int(bad.sum())
    if n_invalid / n_raw > max_invalid_fraction:
        raise DataQualityError(
            f"{name}: {n_invalid}/{n_raw} readings invalid ({100 * n_invalid / n_raw:.1f}%) "
            f"> allowed {100 * max_invalid_fraction:.1f}% (NaN={int(nan_mask.sum())}, "
            f"negative={int(neg_mask.sum())}, >{max_valid_w:g}W={int(high_mask.sum())})")

    out = vals[~bad].astype("float64")
    out = out.sort_index()
    dup_mask = out.index.duplicated(keep="first")
    n_dup = int(dup_mask.sum())
    out = out[~dup_mask]
    if len(out) < 2:
        raise DataQualityError(f"{name}: fewer than 2 valid readings after cleaning")

    t = (out.index - out.index[0]) / pd.Timedelta(seconds=1)
    dt = np.diff(np.asarray(t, dtype="float64"))
    gap_mask = dt > max_gap_s
    span_s = float(t.to_numpy()[-1] - t.to_numpy()[0])
    covered_s = float(dt[~gap_mask].sum())
    report = {
        "appliance": name, "units": "W",
        "n_raw": n_raw, "n_nan_or_non_numeric": int(nan_mask.sum()), "n_negative": int(neg_mask.sum()),
        "n_above_max_valid": int(high_mask.sum()), "n_duplicate_timestamps": n_dup, "n_kept": int(len(out)),
        "median_interval_s": float(np.median(dt)), "n_gaps_over_max": int(gap_mask.sum()),
        "longest_gap_h": float(dt.max() / 3600.0), "start_utc": str(out.index[0]), "end_utc": str(out.index[-1]),
        "span_days": span_s / 86400.0, "coverage_pct": 100.0 * covered_s / span_s if span_s > 0 else 100.0,
        "min_w": float(out.min()), "max_w": float(out.max()), "mean_w": float(out.mean()),
    }
    return out, report


def clean_all(series: dict[str, pd.Series], cfg) -> tuple[dict[str, pd.Series], list[dict]]:
    d = cfg.dataset
    cleaned, reports = {}, []
    for name, s in series.items():
        c, r = clean_power_series(s, name, d.max_gap_s, d.max_valid_w, d.max_invalid_fraction)
        cleaned[name] = c
        reports.append(r)
    return cleaned, reports


def quality_report_markdown(reports: list[dict], ingest_meta: dict, cfg) -> str:
    d = cfg.dataset
    lines = ["# Data quality report", ""]
    if d.synthetic:
        lines += ["> **SYNTHETIC DATA** - demonstration only. These numbers are not a measured benchmark (risk R-05).", ""]
    lines += [
        "## Source", "",
        f"- Dataset: {d.name}", f"- Source: {d.source}", f"- Licence: {d.licence}", f"- Version: {d.version}",
        f"- File: {ingest_meta.get('file')}  ({ingest_meta.get('n_rows')} rows)",
        f"- Units: power in watts (W); timestamps converted to UTC; local features use `{d.timezone}`",
        f"- Channel mapping verified against source metadata: **{d.mapping_verified}**",
        f"- Rows flagged by dataset authors as having issues: {ingest_meta.get('n_issue_rows_flagged_by_source', 'n/a')}",
        f"- Max sample gap integrated: {d.max_gap_s:g} s; max valid reading: {d.max_valid_w:g} W", "",
        "## Per-appliance profile", "",
        "| Appliance | Kept | NaN | Negative | > max | Duplicates | Median dt (s) | Gaps | Longest gap (h) | Coverage % | Mean W | Max W |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in reports:
        lines.append(f"| {r['appliance']} | {r['n_kept']} | {r['n_nan_or_non_numeric']} | {r['n_negative']} | "
                     f"{r['n_above_max_valid']} | {r['n_duplicate_timestamps']} | {r['median_interval_s']:.1f} | "
                     f"{r['n_gaps_over_max']} | {r['longest_gap_h']:.2f} | {r['coverage_pct']:.1f} | "
                     f"{r['mean_w']:.1f} | {r['max_w']:.0f} |")
    lines += ["", "## Policies applied", "",
              "- NaN / negative / implausibly high readings are dropped and counted above; a series is rejected if the invalid share is too high.",
              "- Duplicate timestamps: first kept. Out-of-order rows: sorted.",
              "- Sample gaps longer than the max gap are **not integrated** (energy is under-counted there, never interpolated); sessions are split at gaps.",
              "- Unresolved items (licence, release version, mapping verification, tariff) remain **TBD** until confirmed by the team.", ""]
    return "\n".join(lines)
