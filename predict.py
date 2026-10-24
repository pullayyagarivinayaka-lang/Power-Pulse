"""FR-07 / AC-06: inference interface with an explicit output contract.

Every result states units, status, provenance (measured | calculated | ml_estimated), assumptions,
model version and warnings. Missing/invalid inputs return status="error" - there are no silent defaults.

    python -m src.predict --artifact-dir artifacts/synthetic --appliance washing_machine --duration-h 1.2
    python -m src.predict --artifact-dir artifacts/synthetic --appliance fridge --trace my_trace.csv
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

from .baseline import baseline_energy_kwh
from .energy import INTEGRATION_METHOD, integrate_energy_kwh
from .features import time_features
from .preprocess import DataQualityError, clean_power_series
from .tariff import Tariff, load_tariff

SCHEMA_VERSION = "1.0"
MAX_RATED_POWER_W = 20000.0


def load_artifacts(artifact_dir: str | Path) -> dict:
    d = Path(artifact_dir)
    meta_path = d / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found. Run the pipeline first (see README).")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    model = None
    if (d / "model.joblib").exists():
        model = joblib.load(d / "model.joblib")["model"]
    return {"metadata": meta, "model": model}


def _result(appliance, status, errors, warnings, meta, **extra) -> dict:
    base = {"schema_version": SCHEMA_VERSION, "status": status, "errors": errors, "warnings": warnings,
            "appliance": appliance, "units": {"energy": "kWh", "power": "W", "duration": "h"},
            "model_version": (meta or {}).get("model_version"),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "disclaimer": "Estimate for decision support only; not a substitute for your electricity meter or bill."}
    base.update(extra)
    return base


def estimate_appliance(appliance: str, duration_h: float, art: dict, rated_power_w: float | None = None,
                       start_time: datetime | None = None, tariff: Tariff | None = None,
                       monthly_total_kwh: float | None = None) -> dict:
    """Estimate energy (kWh) and cost for one operating session of ``appliance`` lasting ``duration_h``."""
    meta, model = art["metadata"], art["model"]
    errors: list[str] = []
    warnings: list[str] = []
    if appliance not in meta["appliances"]:
        return _result(appliance, "error", [f"Unknown appliance '{appliance}'. Known: {sorted(meta['appliances'])}"], [], meta,
                       energy_kwh=None, energy_provenance=None)
    ap = meta["appliances"][appliance]
    try:
        duration_h = float(duration_h)
    except (TypeError, ValueError):
        errors.append("duration_h must be a number (hours)")
        duration_h = float("nan")
    if not math.isfinite(duration_h) or duration_h <= 0:
        errors.append("duration_h must be a finite number > 0 (hours)")
    rated = ap["rated_power_w"] if rated_power_w is None else rated_power_w
    if rated_power_w is None:
        warnings.append(f"rated_power_w not supplied; using configured value {rated:g} W "
                        f"(source: {ap['rated_power_source']}).")
    try:
        rated = float(rated)
    except (TypeError, ValueError):
        errors.append("rated_power_w must be a number (watts)")
        rated = float("nan")
    if not math.isfinite(rated) or rated <= 0 or rated > MAX_RATED_POWER_W:
        errors.append(f"rated_power_w must be in (0, {MAX_RATED_POWER_W:g}] watts")
    if errors:
        return _result(appliance, "error", errors, warnings, meta, energy_kwh=None, energy_provenance=None)

    base_kwh = baseline_energy_kwh(rated, duration_h)
    stats = meta["appliance_stats"].get(appliance, {})
    if duration_h > stats.get("duration_h_p99", float("inf")):
        warnings.append(f"duration_h exceeds the 99th percentile of training sessions "
                        f"({stats['duration_h_p99']:.2f} h): low confidence for the ML estimate.")
    if abs(rated - ap["rated_power_w"]) / ap["rated_power_w"] > 0.3:
        warnings.append("rated_power_w differs >30% from the value used in training: low confidence for the ML estimate.")
    if meta.get("synthetic"):
        warnings.append("Model was trained on SYNTHETIC demo data; do not use outputs as real estimates.")

    tz = meta["timezone"]
    if start_time is None:
        start_time = datetime.now(ZoneInfo(tz))
        warnings.append("start_time not supplied; the current local time was used for time-of-day features.")
    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=ZoneInfo(tz))

    ml_kwh = None
    if model is not None:
        st = pd.Series([pd.Timestamp(start_time).tz_convert("UTC")])
        row = {"appliance": appliance, "duration_h": duration_h, "rated_power_w": rated, "baseline_kwh": base_kwh,
               **time_features(st, tz).iloc[0].to_dict()}
        if "hist_mean_power_w" in meta["feature_cols"]:
            row["hist_mean_power_w"] = stats.get("reference_hist_power_w", rated)
            warnings.append("History feature taken from the training-data reference for this appliance, "
                            "not from your own household.")
        ml_kwh = float(max(model.predict(pd.DataFrame([row])[meta["feature_cols"]])[0], 0.0))
        energy, prov = ml_kwh, "ml_estimated"
        method = f"{meta['selected_model']} (validated against rated-power baseline)"
    else:
        energy, prov = base_kwh, "calculated"
        method = "rated_power_W * duration_h / 1000 (theoretical; ML not justified on validation data)"
        warnings.append("Theoretical estimate from rated power; real consumption can differ for cycling/variable loads.")

    cost, currency, placeholder = None, None, None
    if tariff is not None:
        cost, w = tariff.cost_of_energy(energy, monthly_total_kwh)
        warnings += w
        currency, placeholder = tariff.currency, tariff.is_placeholder
        if placeholder:
            warnings.append("Tariff is a PLACEHOLDER; replace it with your provider's current tariff before relying on cost.")
    return _result(appliance, "warning" if warnings else "ok", [], warnings, meta,
                   inputs={"duration_h": duration_h, "rated_power_w": rated,
                           "start_time_local": pd.Timestamp(start_time).isoformat()},
                   energy_kwh=round(energy, 6), energy_provenance=prov, method=method,
                   baseline_energy_kwh=round(base_kwh, 6),
                   ml_energy_kwh=None if ml_kwh is None else round(ml_kwh, 6),
                   cost=None if cost is None else round(cost, 4), currency=currency, tariff_is_placeholder=placeholder,
                   assumptions=[f"rated_power_source: {ap['rated_power_source']}",
                                "cost = tariff-based estimate; slab tariffs use the month's average rate"])


def estimate_from_trace(appliance: str, trace: pd.DataFrame, art: dict | None = None, tariff: Tariff | None = None,
                        max_gap_s: float = 300.0, max_valid_w: float = 20000.0) -> dict:
    """Measured energy from a user-supplied power trace. Columns: ``timestamp`` + ``power_w`` (or ``power_kw``)."""
    meta = (art or {}).get("metadata")
    errors: list[str] = []
    if "timestamp" not in trace.columns or not ({"power_w", "power_kw"} & set(trace.columns)):
        return _result(appliance, "error", ["trace must have columns 'timestamp' and 'power_w' (or 'power_kw')"], [], meta,
                       energy_kwh=None, energy_provenance=None)
    p = pd.to_numeric(trace["power_w"] if "power_w" in trace.columns else trace["power_kw"] * 1000.0, errors="coerce")
    ts = pd.to_datetime(trace["timestamp"], errors="coerce", utc=True)
    if ts.isna().any():
        return _result(appliance, "error", [f"{int(ts.isna().sum())} unparseable timestamps"], [], meta,
                       energy_kwh=None, energy_provenance=None)
    warnings = ["Timestamps without a timezone were interpreted as UTC."]
    try:
        s, q = clean_power_series(pd.Series(p.to_numpy(float), index=pd.DatetimeIndex(ts)), appliance, max_gap_s,
                                  max_valid_w, 0.05)
    except DataQualityError as exc:
        return _result(appliance, "error", [str(exc)], warnings, meta, energy_kwh=None, energy_provenance=None)
    res = integrate_energy_kwh(s, max_gap_s)
    if res.coverage < 0.95:
        warnings.append(f"Only {100 * res.coverage:.1f}% of the time span had usable data; energy is under-counted.")
    cost, currency = None, None
    if tariff is not None:
        cost, w = tariff.cost_of_energy(res.energy_kwh)
        warnings += w
        currency = tariff.currency
        if tariff.is_placeholder:
            warnings.append("Tariff is a PLACEHOLDER; replace it before relying on cost.")
    return _result(appliance, "warning" if warnings else "ok", [], warnings, meta,
                   energy_kwh=round(res.energy_kwh, 6), energy_provenance="measured", method=res.method,
                   span_hours=round(res.span_hours, 4), data_coverage_pct=round(100 * res.coverage, 2),
                   cost=None if cost is None else round(cost, 4), currency=currency, data_quality=q)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="PowerPulse Model 2 - appliance energy & cost estimate")
    ap.add_argument("--artifact-dir", default="artifacts/refit")
    ap.add_argument("--appliance", required=True)
    ap.add_argument("--duration-h", type=float, help="operating duration in hours")
    ap.add_argument("--rated-power-w", type=float, help="rated power in watts (optional)")
    ap.add_argument("--start", help="session start, ISO format e.g. 2026-10-24T19:30 (local time)")
    ap.add_argument("--trace", help="CSV with columns timestamp,power_w for a MEASURED estimate")
    ap.add_argument("--tariff", default=None, help="tariff JSON (optional)")
    ap.add_argument("--monthly-total-kwh", type=float, default=None, help="household monthly kWh (slab tariffs)")
    args = ap.parse_args(argv)

    art = load_artifacts(args.artifact_dir)
    tariff = load_tariff(args.tariff) if args.tariff else None
    if args.trace:
        out = estimate_from_trace(args.appliance, pd.read_csv(args.trace), art, tariff)
    else:
        if args.duration_h is None:
            ap.error("--duration-h is required unless --trace is given")
        start = datetime.fromisoformat(args.start) if args.start else None
        out = estimate_appliance(args.appliance, args.duration_h, art, args.rated_power_w, start, tariff, args.monthly_total_kwh)
    print(json.dumps(out, indent=2, default=str))
    return 0 if out["status"] != "error" else 2


if __name__ == "__main__":
    sys.exit(main())
