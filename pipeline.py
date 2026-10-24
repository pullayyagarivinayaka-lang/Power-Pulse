"""End-to-end pipeline: ingest -> validate -> sessions/kWh labels -> features -> split -> baseline & ML
-> evaluate -> save artifacts + reports.

    python -m src.pipeline --config configs/synthetic_demo.json --raw data/raw/synthetic_house.csv \
        --tariff configs/tariff.json --run-name synthetic
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from .config import load_config
from .energy import INTEGRATION_METHOD, detect_all_sessions
from .evaluate import evaluate_models, evaluation_markdown, make_figures, predict_all
from .features import build_features, feature_columns
from .ingest import load_raw
from .preprocess import clean_all, quality_report_markdown
from .split import assert_no_overlap, chronological_split
from .summary import monthly_appliance_summary
from .tariff import load_tariff
from .train import train_and_select

log = logging.getLogger("powerpulse.m02")


def run(config_path: str, raw_path: str, tariff_path: str, run_name: str, root: str = ".",
        families: list[str] | None = None) -> dict:
    root_p = Path(root)
    cfg = load_config(config_path)
    tariff = load_tariff(tariff_path)
    m = cfg.modeling
    np.random.seed(m.seed)
    reports_dir = root_p / "reports" / run_name
    art_dir = root_p / "artifacts" / run_name
    proc_dir = root_p / "data" / "processed" / run_name
    for p in (reports_dir, art_dir, proc_dir):
        p.mkdir(parents=True, exist_ok=True)

    log.info("[1/7] Ingest %s", raw_path)
    series, ingest_meta = load_raw(raw_path, cfg)

    log.info("[2/7] Validate & clean (units W, UTC timestamps)")
    series, qreports = clean_all(series, cfg)
    (reports_dir / "data_quality.md").write_text(quality_report_markdown(qreports, ingest_meta, cfg), encoding="utf-8")
    (reports_dir / "data_quality.json").write_text(json.dumps(qreports, indent=2), encoding="utf-8")

    log.info("[3/7] Detect operating sessions and integrate measured kWh")
    sessions = detect_all_sessions(series, cfg)
    rated = {n: a.rated_power_w for n, a in cfg.appliances.items()}
    log.info("      %d sessions across %d appliances", len(sessions), sessions["appliance"].nunique())

    log.info("[4/7] Build leakage-safe features + rated-power baseline")
    feats = build_features(sessions, rated, cfg.dataset.timezone, m.use_history_feature, m.history_window)
    feat_cols = feature_columns(m.use_history_feature)
    feats.to_csv(proc_dir / "sessions_features.csv", index=False)

    log.info("[5/7] Chronological split (purged)")
    train, val, test, split_info, thin = chronological_split(
        feats, m.train_frac, m.val_frac, m.purge_hours, m.min_sessions_per_appliance)
    assert_no_overlap(train, val, test)
    log.info("      train=%d val=%d test=%d", len(train), len(val), len(test))

    log.info("[6/7] Train candidates, select on validation")
    tr = train_and_select(train, val, feat_cols, m.seed, m.min_improvement_pct, families)
    log.info("      selected=%s (ML improvement on val %.1f%%)", tr.selected, tr.improvement_pct)

    log.info("[7/7] Evaluate on held-out test period, write reports and artifacts")
    preds = predict_all(tr.models, test, feat_cols)
    results = evaluate_models(preds, cfg.dataset.timezone)
    figs = make_figures(preds, results, tr.selected, reports_dir / "figures")
    sig = json.dumps({"cfg": Path(config_path).read_text(), "n": len(sessions), "seed": m.seed,
                      "start": str(sessions["start_time"].min())}, sort_keys=True)
    version = f"pp-m02-{datetime.now(timezone.utc):%Y%m%d}-{hashlib.sha1(sig.encode()).hexdigest()[:8]}"
    (reports_dir / "evaluation.md").write_text(evaluation_markdown(results, tr, split_info, cfg, version, thin), encoding="utf-8")
    (reports_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    summary = monthly_appliance_summary(series, cfg.dataset.max_gap_s, cfg.dataset.timezone, tariff)
    summary.to_csv(reports_dir / "appliance_energy_summary.csv", index=False)

    stats = {}
    for a, g in feats.groupby("appliance"):
        stats[a] = {"n_sessions": int(len(g)), "duration_h_p99": float(g["duration_h"].quantile(0.99)),
                    "median_energy_kwh": float(g["measured_energy_kwh"].median()),
                    "reference_hist_power_w": float(g["mean_power_w"].tail(m.history_window).mean())}
    metadata = {
        "model_version": version, "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_model": tr.selected, "ml_justified": tr.ml_justified, "improvement_pct_on_validation": tr.improvement_pct,
        "feature_cols": feat_cols, "validation_mae_kwh": tr.validation_mae, "best_params": tr.best_params,
        "test_metrics": results["overall_session"], "split": split_info.__dict__, "seed": m.seed,
        "dataset": cfg.dataset.__dict__, "timezone": cfg.dataset.timezone, "synthetic": cfg.dataset.synthetic,
        "appliances": {n: a.__dict__ for n, a in cfg.appliances.items()}, "appliance_stats": stats,
        "integration_method": INTEGRATION_METHOD, "units": {"power": "W", "energy": "kWh", "duration": "h"},
        "library_versions": {"pandas": pd.__version__, "numpy": np.__version__, "scikit-learn": sklearn.__version__},
    }
    (art_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    if tr.ml_justified:
        joblib.dump({"model": tr.models[tr.selected], "feature_cols": feat_cols, "model_version": version}, art_dir / "model.joblib")
    elif (art_dir / "model.joblib").exists():
        (art_dir / "model.joblib").unlink()      # never leave a stale model next to a baseline-only decision
    log.info("Done. Reports: %s | Artifacts: %s", reports_dir, art_dir)
    return {"version": version, "selected": tr.selected, "validation_mae": tr.validation_mae,
            "test": results["overall_session"], "daily": results["daily_appliance"], "figures": figs}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="PowerPulse Model 2 pipeline")
    ap.add_argument("--config", required=True)
    ap.add_argument("--raw", required=True, help="path to raw CSV (see data/README.md)")
    ap.add_argument("--tariff", default="configs/tariff.json")
    ap.add_argument("--run-name", default="refit", help="outputs go to reports|artifacts|data/processed/<run-name>")
    ap.add_argument("--root", default=".")
    ap.add_argument("--families", nargs="+", default=None,
                    choices=["random_forest", "gradient_boosting", "hybrid_gradient_boosting"],
                    help="restrict candidate models (e.g. skip random_forest on very large REFIT files)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    out = run(args.config, args.raw, args.tariff, args.run_name, args.root, args.families)
    print(json.dumps({k: out[k] for k in ("version", "selected", "validation_mae", "test")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
