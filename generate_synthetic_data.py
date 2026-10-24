"""Generate SYNTHETIC REFIT-like data (Time, Unix, Aggregate, Appliance1..9, Issues) at 1-minute sampling.

PURPOSE: let you run and test the whole pipeline before the real REFIT download is available.
This is NOT measured data. Any metric computed from it is a demonstration, not a benchmark (risk R-05).
Appliance behaviour is invented (e.g. fridge power varies with season, heater duty-cycle with month) so
that a learner has *something* to find beyond the rated-power baseline.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CHANNELS = ["fridge", "chest_freezer", "upright_freezer", "tumble_dryer", "washing_machine",
            "dishwasher", "computer", "television", "electric_heater"]
PER_DAY = 1440  # 1-minute samples


def _cycling(n, rng, on_mean, off_mean, power, seasonal):
    out = np.full(n, 2.0)  # standby W
    i = 0
    while i < n:
        on = max(3, int(rng.normal(on_mean, 2)))
        off = max(8, int(rng.normal(off_mean, 5)))
        out[i:i + on] = rng.normal(power * seasonal[i:i + on].mean(), 4, size=len(out[i:i + on]))
        i += on + off
    return np.clip(out, 0, None)


def generate(days: int, start: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = days * PER_DAY
    idx = pd.date_range(start, periods=n, freq="60s", tz="UTC")
    month = np.asarray(idx.month)
    summer = 1.0 + 0.15 * np.cos((month - 7) / 12 * 2 * np.pi)       # fridge works harder in summer
    P = {c: np.zeros(n) for c in CHANNELS}
    P["fridge"] = _cycling(n, rng, 12, 28, 90, summer)
    P["chest_freezer"] = _cycling(n, rng, 15, 25, 110, summer)
    P["upright_freezer"] = _cycling(n, rng, 14, 30, 100, summer)

    def put(name, day, minute, profile):
        s = day * PER_DAY + int(minute)
        e = min(s + len(profile), n)
        if s < n:
            P[name][s:e] += profile[: e - s]

    for d in range(days):
        m = idx[d * PER_DAY].month
        dow = idx[d * PER_DAY].dayofweek
        weekend = dow >= 5
        if rng.random() < (0.45 if weekend else 0.30):                  # tumble dryer
            dur = int(rng.integers(55, 100))
            prof = rng.normal(2300, 60, dur) * (rng.random(dur) > 0.15) + 150
            put("tumble_dryer", d, rng.integers(8, 20) * 60 + rng.integers(0, 60), prof)
        if rng.random() < (0.55 if weekend else 0.35):                  # washing machine
            heat = rng.normal(2050, 40, int(rng.integers(12, 20)))
            motor = rng.uniform(120, 320, int(rng.integers(40, 70)))
            spin = rng.normal(420, 30, int(rng.integers(8, 14)))
            put("washing_machine", d, rng.integers(7, 21) * 60 + rng.integers(0, 60), np.concatenate([heat, motor, spin]))
        if rng.random() < 0.60:                                         # dishwasher after dinner
            prof = np.concatenate([rng.normal(1800, 30, int(rng.integers(18, 26))),
                                   rng.normal(120, 10, int(rng.integers(45, 70))),
                                   rng.normal(1000, 40, int(rng.integers(10, 18)))])
            put("dishwasher", d, rng.integers(19, 22) * 60 + rng.integers(0, 60), prof)
        if rng.random() < 0.90:                                         # computer
            dur = int(rng.integers(120, 360))
            base = rng.uniform(70, 130)
            put("computer", d, rng.integers(17, 20) * 60 + rng.integers(0, 60), np.clip(rng.normal(base, 12, dur), 20, None))
        if rng.random() < 0.95:                                         # television
            dur = int(rng.integers(120, 300))
            put("television", d, rng.integers(18, 21) * 60 + rng.integers(0, 60), np.clip(rng.normal(rng.uniform(60, 110), 6, dur), 15, None))
        if m in (11, 12, 1, 2, 3) and rng.random() < 0.80:              # heater: thermostat duty by month
            duty = 0.7 if m in (12, 1) else 0.4
            dur = int(rng.integers(180, 360))
            blocks = np.repeat(rng.random(dur // 10 + 1) < duty, 10)[:dur]
            put("electric_heater", d, rng.integers(17, 20) * 60, np.where(blocks, rng.normal(2000, 25, dur), 3.0))

    unix = ((idx - pd.Timestamp("1970-01-01", tz="UTC")).total_seconds()).astype("int64")
    df = pd.DataFrame({"Time": idx.tz_localize(None).strftime("%Y-%m-%d %H:%M:%S"), "Unix": unix})
    agg = sum(P.values()) + rng.uniform(60, 120, n)
    df["Aggregate"] = np.round(agg, 1)
    for i, c in enumerate(CHANNELS, start=1):
        df[f"Appliance{i}"] = np.round(P[c], 1)
    df["Issues"] = 0
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/synthetic_house.csv")
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate(a.days, a.start, a.seed)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows of SYNTHETIC data to {out}")


if __name__ == "__main__":
    main()
