"""Configuration loading and validation (units are explicit in every field name)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


class ConfigError(ValueError):
    """Raised when a config file is missing fields or contains invalid values."""


@dataclass(frozen=True)
class ApplianceConfig:
    name: str
    channel: str
    on_threshold_w: float      # power above this (W) = appliance is operating
    min_on_s: float            # sessions shorter than this (s) are discarded
    merge_gap_s: float         # OFF periods shorter than this (s) are merged into one session
    rated_power_w: float       # nameplate / assumed rated power (W)
    rated_power_source: str


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    source: str
    licence: str
    version: str
    format: str                # "refit_wide" | "long"
    timezone: str
    mapping_verified: bool
    synthetic: bool
    max_gap_s: float           # sample gaps longer than this (s) are NOT integrated
    max_valid_w: float         # readings above this (W) are rejected as invalid
    max_invalid_fraction: float
    timestamp_col: str = "Time"
    unix_col: str | None = "Unix"
    issues_col: str | None = "Issues"
    appliance_col: str = "appliance"   # only used for format == "long"
    power_col: str = "power_w"         # only used for format == "long"


@dataclass(frozen=True)
class ModelingConfig:
    seed: int = 42
    train_frac: float = 0.70
    val_frac: float = 0.15
    purge_hours: float = 24.0
    use_history_feature: bool = True
    history_window: int = 20
    min_improvement_pct: float = 5.0
    min_sessions_per_appliance: int = 30


@dataclass(frozen=True)
class AppConfig:
    dataset: DatasetConfig
    appliances: dict[str, ApplianceConfig]
    modeling: ModelingConfig
    path: str = ""


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


def load_config(path: str | Path) -> AppConfig:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    for key in ("dataset", "appliances"):
        _require(key in raw, f"Config is missing the '{key}' section")

    d = raw["dataset"]
    try:
        dataset = DatasetConfig(
            name=d["name"], source=d["source"], licence=d["licence"], version=d["version"],
            format=d["format"], timezone=d["timezone"],
            mapping_verified=bool(d["mapping_verified"]), synthetic=bool(d.get("synthetic", False)),
            max_gap_s=float(d["max_gap_s"]), max_valid_w=float(d["max_valid_w"]),
            max_invalid_fraction=float(d["max_invalid_fraction"]),
            timestamp_col=d.get("timestamp_col", "Time"), unix_col=d.get("unix_col"),
            issues_col=d.get("issues_col"),
            appliance_col=d.get("appliance_col", "appliance"), power_col=d.get("power_col", "power_w"),
        )
    except KeyError as exc:
        raise ConfigError(f"Dataset config is missing field {exc}") from exc
    _require(dataset.format in ("refit_wide", "long"), "dataset.format must be 'refit_wide' or 'long'")
    _require(dataset.max_gap_s > 0, "dataset.max_gap_s must be > 0")
    _require(dataset.max_valid_w > 0, "dataset.max_valid_w must be > 0")
    _require(0 <= dataset.max_invalid_fraction < 1, "dataset.max_invalid_fraction must be in [0, 1)")

    appliances: dict[str, ApplianceConfig] = {}
    for name, a in raw["appliances"].items():
        try:
            appliances[name] = ApplianceConfig(
                name=name, channel=a["channel"], on_threshold_w=float(a["on_threshold_w"]),
                min_on_s=float(a["min_on_s"]), merge_gap_s=float(a["merge_gap_s"]),
                rated_power_w=float(a["rated_power_w"]), rated_power_source=a["rated_power_source"],
            )
        except KeyError as exc:
            raise ConfigError(f"Appliance '{name}' is missing field {exc}") from exc
        ap = appliances[name]
        _require(ap.rated_power_w > 0, f"{name}: rated_power_w must be > 0 (W)")
        _require(ap.on_threshold_w >= 0, f"{name}: on_threshold_w must be >= 0 (W)")
        _require(ap.min_on_s >= 0 and ap.merge_gap_s >= 0, f"{name}: durations must be >= 0 (s)")
    _require(len(appliances) > 0, "At least one appliance must be configured")
    channels = [a.channel for a in appliances.values()]
    _require(len(set(channels)) == len(channels),
             "Two appliances map to the same channel; fix the channel mapping (AC-03)")

    m = raw.get("modeling", {})
    modeling = ModelingConfig(**{k: v for k, v in m.items() if not k.startswith("_")})
    _require(0 < modeling.train_frac < 1 and 0 < modeling.val_frac < 1
             and modeling.train_frac + modeling.val_frac < 1, "train_frac + val_frac must be < 1")
    return AppConfig(dataset=dataset, appliances=appliances, modeling=modeling, path=str(path))
