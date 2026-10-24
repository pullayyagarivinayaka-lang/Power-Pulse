"""FR-07 (measured side): appliance-wise monthly energy and cost from measured traces, with provenance."""
from __future__ import annotations

import pandas as pd

from .energy import sample_energy_kwh
from .tariff import Tariff


def monthly_appliance_summary(series: dict[str, pd.Series], max_gap_s: float, tz: str, tariff: Tariff) -> pd.DataFrame:
    parts = []
    for name, s in series.items():
        e = sample_energy_kwh(s, max_gap_s)
        month = e.index.tz_convert(tz).tz_localize(None).to_period("M").astype(str)
        parts.append(e.groupby(month).sum().rename("energy_kwh").reset_index().rename(columns={"index": "month"})
                     .assign(appliance=name))
    df = pd.concat(parts, ignore_index=True)
    df.columns = ["month", "energy_kwh", "appliance"]
    df["month_total_kwh"] = df.groupby("month")["energy_kwh"].transform("sum")
    df["share_pct"] = 100 * df["energy_kwh"] / df["month_total_kwh"].where(df["month_total_kwh"] > 0)
    df["cost"] = [tariff.cost_of_energy(e, t)[0] for e, t in zip(df["energy_kwh"], df["month_total_kwh"])]
    df["currency"] = tariff.currency
    df["tariff_is_placeholder"] = tariff.is_placeholder
    df["provenance"] = "measured (integrated from power trace); cost = tariff-based estimate"
    return df[["month", "appliance", "energy_kwh", "share_pct", "cost", "currency", "tariff_is_placeholder",
               "provenance"]].sort_values(["month", "energy_kwh"], ascending=[True, False]).reset_index(drop=True)
