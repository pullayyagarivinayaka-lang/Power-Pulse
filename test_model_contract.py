"""E2E + output-contract tests (FR-07, AC-05, AC-06) on a small SYNTHETIC dataset."""
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from scripts.generate_synthetic_data import generate
from src.pipeline import run
from src.predict import estimate_appliance, estimate_from_trace, load_artifacts
from src.tariff import Tariff, load_tariff

REQUIRED = {"schema_version", "status", "errors", "warnings", "appliance", "units", "model_version",
            "generated_at_utc", "energy_kwh", "energy_provenance", "disclaimer"}


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("run")
    raw = root / "raw.csv"
    generate(days=200, start="2023-06-01", seed=1).to_csv(raw, index=False)
    out = run("configs/synthetic_demo.json", str(raw), "configs/tariff.json", "t", str(root), families=["gradient_boosting"])
    return root, out


def test_pipeline_outputs_exist(run_dir):
    root, out = run_dir
    for f in ("reports/t/data_quality.md", "reports/t/evaluation.md", "reports/t/metrics.json",
              "reports/t/appliance_energy_summary.csv", "artifacts/t/metadata.json"):
        assert (root / f).exists(), f
    assert "baseline" in out["test"] and "gradient_boosting" in out["test"]          # AC-05: compared with baseline
    for m in out["test"].values():
        assert m["mae_kwh"] >= 0 and m["rmse_kwh"] >= m["mae_kwh"] - 1e-12


def test_reports_flag_synthetic_data(run_dir):
    root, _ = run_dir
    assert "SYNTHETIC" in (root / "reports/t/evaluation.md").read_text()
    assert json.loads((root / "artifacts/t/metadata.json").read_text())["synthetic"] is True


def test_summary_marks_measured_provenance_and_cost(run_dir):
    root, _ = run_dir
    s = pd.read_csv(root / "reports/t/appliance_energy_summary.csv")
    assert s["provenance"].str.startswith("measured").all() and (s["cost"] >= 0).all()
    assert s.groupby("month")["share_pct"].sum().round(3).between(99.9, 100.1).all()


def test_estimate_contract_ok(run_dir):
    root, _ = run_dir
    art = load_artifacts(root / "artifacts/t")
    r = estimate_appliance("washing_machine", 1.2, art, start_time=datetime(2024, 1, 10, 19, 0), tariff=load_tariff("configs/tariff.json"))
    assert REQUIRED <= set(r) and r["status"] in ("ok", "warning")
    assert r["energy_kwh"] > 0 and r["baseline_energy_kwh"] == pytest.approx(2000 * 1.2 / 1000)
    assert r["energy_provenance"] in ("ml_estimated", "calculated")
    assert r["cost"] == pytest.approx(r["energy_kwh"] * 7.0, abs=1e-3) and r["tariff_is_placeholder"] is True
    assert any("SYNTHETIC" in w for w in r["warnings"])


@pytest.mark.parametrize("kwargs,needle", [
    ({"appliance": "toaster", "duration_h": 1}, "Unknown appliance"),
    ({"appliance": "fridge", "duration_h": -1}, "duration_h"),
    ({"appliance": "fridge", "duration_h": float("nan")}, "duration_h"),
    ({"appliance": "fridge", "duration_h": 1, "rated_power_w": 0}, "rated_power_w"),
    ({"appliance": "fridge", "duration_h": 1, "rated_power_w": 1e7}, "rated_power_w"),
])
def test_invalid_inputs_return_explicit_errors(run_dir, kwargs, needle):
    root, _ = run_dir
    r = estimate_appliance(art=load_artifacts(root / "artifacts/t"), **kwargs)
    assert r["status"] == "error" and r["energy_kwh"] is None and any(needle in e for e in r["errors"])


def test_baseline_only_fallback_is_labelled_calculated(run_dir):
    root, _ = run_dir
    art = load_artifacts(root / "artifacts/t")
    art["model"] = None
    r = estimate_appliance("computer", 3, art, rated_power_w=150, start_time=datetime(2024, 1, 10, 19, 0))
    assert r["energy_provenance"] == "calculated" and r["energy_kwh"] == pytest.approx(0.45)


def test_long_duration_triggers_low_confidence_warning(run_dir):
    root, _ = run_dir
    r = estimate_appliance("television", 500, load_artifacts(root / "artifacts/t"), start_time=datetime(2024, 1, 10, 19, 0))
    assert any("99th percentile" in w for w in r["warnings"])


def test_trace_estimate_is_measured_and_matches_hand_calc(run_dir):
    root, _ = run_dir
    ts = pd.date_range("2024-01-01", periods=120, freq="60s", tz="UTC")
    r = estimate_from_trace("fridge", pd.DataFrame({"timestamp": ts.astype(str), "power_w": 500.0}), load_artifacts(root / "artifacts/t"),
                            Tariff("t", "INR", "flat", rate_per_kwh=10.0, is_placeholder=False))
    assert r["energy_provenance"] == "measured" and r["energy_kwh"] == pytest.approx(1.0, abs=1e-6)
    assert r["cost"] == pytest.approx(10.0, abs=1e-3)


def test_trace_requires_explicit_units(run_dir):
    root, _ = run_dir
    r = estimate_from_trace("fridge", pd.DataFrame({"timestamp": ["2024-01-01"], "power": [5]}), load_artifacts(root / "artifacts/t"))
    assert r["status"] == "error" and "power_w" in r["errors"][0]
