"""FR-05: transparent rated-power baseline.  E_kWh = rated_power_W * duration_h / 1000."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def baseline_energy_kwh(rated_power_w: float, duration_h: float) -> float:
    """Theoretical energy (kWh) from rated power (W) and operating duration (h)."""
    if not (math.isfinite(rated_power_w) and math.isfinite(duration_h)):
        raise ValueError("rated_power_w and duration_h must be finite numbers")
    if rated_power_w <= 0:
        raise ValueError("rated_power_w must be > 0 (watts)")
    if duration_h < 0:
        raise ValueError("duration_h must be >= 0 (hours)")
    return rated_power_w * duration_h / 1000.0


def add_baseline(sessions: pd.DataFrame, rated_power_w_by_appliance: dict[str, float]) -> pd.DataFrame:
    out = sessions.copy()
    unknown = set(out["appliance"]) - set(rated_power_w_by_appliance)
    if unknown:
        raise ValueError(f"No rated power configured for: {sorted(unknown)}")
    out["rated_power_w"] = out["appliance"].map(rated_power_w_by_appliance).astype(float)
    out["baseline_kwh"] = out["rated_power_w"].to_numpy() * out["duration_h"].to_numpy() / 1000.0
    return out
