"""AC-03: channels mapped from config metadata, never guessed; FR-01/FR-02 schema checks."""
import json

import pandas as pd
import pytest

from src.config import ConfigError, load_config
from src.ingest import IngestError, load_refit_wide


def _write_cfg(tmp_path, appliances=None, verified=True):
    cfg = json.loads(open("configs/synthetic_demo.json").read())
    cfg["dataset"]["mapping_verified"] = verified
    if appliances is not None:
        cfg["appliances"] = appliances
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    return p


def _app(channel):
    return {"channel": channel, "on_threshold_w": 10, "min_on_s": 60, "merge_gap_s": 0, "rated_power_w": 100, "rated_power_source": "t"}


def _csv(tmp_path):
    p = tmp_path / "raw.csv"
    pd.DataFrame({"Time": ["2024-01-01 00:00:00", "2024-01-01 00:01:00", "2024-01-01 00:02:00"],
                  "Unix": [1704067200, 1704067260, 1704067320], "Aggregate": [100, 100, 100],
                  "Appliance1": [10.0, 20.0, 30.0], "Appliance2": [0.0, 0.0, 5.0], "Issues": [0, 1, 0]}).to_csv(p, index=False)
    return p


def test_duplicate_channel_mapping_rejected(tmp_path):
    with pytest.raises(ConfigError, match="same channel"):
        load_config(_write_cfg(tmp_path, {"a": _app("Appliance1"), "b": _app("Appliance1")}))


def test_missing_channel_in_file_is_an_explicit_error(tmp_path):
    cfg = load_config(_write_cfg(tmp_path, {"a": _app("Appliance1"), "b": _app("Appliance7")}))
    with pytest.raises(IngestError, match="Appliance7"):
        load_refit_wide(_csv(tmp_path), cfg)


def test_mapping_comes_from_config_and_issue_rows_are_masked(tmp_path):
    cfg = load_config(_write_cfg(tmp_path, {"fridge": _app("Appliance2"), "tv": _app("Appliance1")}))
    series, meta = load_refit_wide(_csv(tmp_path), cfg)
    assert series["tv"].iloc[0] == 10.0 and series["fridge"].iloc[2] == 5.0     # not by channel order
    assert pd.isna(series["tv"].iloc[1]) and meta["n_issue_rows_flagged_by_source"] == 1
    assert str(series["tv"].index.tz) == "UTC"


def test_unverified_mapping_warns(tmp_path, caplog):
    cfg = load_config(_write_cfg(tmp_path, {"a": _app("Appliance1")}, verified=False))
    with caplog.at_level("WARNING"):
        load_refit_wide(_csv(tmp_path), cfg)
    assert "NOT marked verified" in caplog.text


def test_refit_template_config_loads_and_is_unverified():
    cfg = load_config("configs/refit_house1.json")
    assert cfg.dataset.mapping_verified is False and len(cfg.appliances) == 9
