"""Tariff arithmetic (kWh -> cost). Tariffs are always user-supplied; the shipped one is a PLACEHOLDER."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


class TariffError(ValueError):
    pass


@dataclass(frozen=True)
class Tariff:
    name: str
    currency: str
    type: str                                # "flat" | "slab"
    rate_per_kwh: float | None = None
    slabs: tuple[tuple[float | None, float], ...] = ()   # (up_to_kwh or None, rate_per_kwh)
    fixed_charge_per_month: float = 0.0
    is_placeholder: bool = True

    def monthly_cost(self, monthly_kwh: float, include_fixed: bool = False) -> float:
        """Total energy charge for a month's consumption (progressive slabs)."""
        if monthly_kwh < 0:
            raise TariffError("energy (kWh) must be >= 0")
        if self.type == "flat":
            cost = monthly_kwh * float(self.rate_per_kwh)
        else:
            cost, prev = 0.0, 0.0
            for up_to, rate in self.slabs:
                top = monthly_kwh if up_to is None else min(monthly_kwh, up_to)
                if top > prev:
                    cost += (top - prev) * rate
                    prev = top
                if up_to is None or monthly_kwh <= up_to:
                    break
        return cost + (self.fixed_charge_per_month if include_fixed else 0.0)

    def cost_of_energy(self, kwh: float, monthly_total_kwh: float | None = None) -> tuple[float, list[str]]:
        """Cost of ``kwh`` of one appliance/session.

        Flat: kwh * rate. Slab: the average rate of the month (monthly cost / monthly kWh) is applied,
        so appliance costs add up to the monthly energy charge. If ``monthly_total_kwh`` is not given,
        ``kwh`` itself is treated as the whole month (a warning is returned).
        """
        warnings: list[str] = []
        if self.type == "flat":
            return self.monthly_cost(kwh), warnings
        if monthly_total_kwh is None or monthly_total_kwh <= 0:
            monthly_total_kwh = kwh
            warnings.append("Slab tariff: monthly total not supplied; cost computed as if this energy were the "
                            "whole month's consumption.")
        if monthly_total_kwh <= 0:
            return 0.0, warnings
        avg_rate = self.monthly_cost(monthly_total_kwh) / monthly_total_kwh
        return kwh * avg_rate, warnings


def load_tariff(path: str | Path) -> Tariff:
    p = Path(path)
    if not p.exists():
        raise TariffError(f"Tariff file not found: {p}")
    r = json.loads(p.read_text(encoding="utf-8"))
    try:
        ttype = r["type"]
        if ttype == "flat":
            rate = float(r["rate_per_kwh"])
            if rate < 0:
                raise TariffError("rate_per_kwh must be >= 0")
            t = Tariff(r["name"], r["currency"], "flat", rate_per_kwh=rate,
                       fixed_charge_per_month=float(r.get("fixed_charge_per_month", 0.0)),
                       is_placeholder=bool(r.get("is_placeholder", True)))
        elif ttype == "slab":
            slabs = tuple((None if s["up_to_kwh"] is None else float(s["up_to_kwh"]), float(s["rate_per_kwh"]))
                          for s in r["slabs"])
            if not slabs or slabs[-1][0] is not None:
                raise TariffError("Last slab must have up_to_kwh = null (open-ended)")
            bounds = [s[0] for s in slabs[:-1]]
            if bounds != sorted(bounds):
                raise TariffError("Slab upper bounds must be increasing")
            t = Tariff(r["name"], r["currency"], "slab", slabs=slabs,
                       fixed_charge_per_month=float(r.get("fixed_charge_per_month", 0.0)),
                       is_placeholder=bool(r.get("is_placeholder", True)))
        else:
            raise TariffError(f"Unknown tariff type '{ttype}'")
    except KeyError as exc:
        raise TariffError(f"Tariff file missing field {exc}") from exc
    return t
