"""AC-05: MAE/RMSE of ML models vs. the rated-power baseline on the held-out test period,
with per-appliance and daily error analysis, plus a human-readable evaluation report."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .features import TARGET


def metrics(y_true, y_pred) -> dict:
    y, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = p - y
    return {"mae_kwh": float(np.mean(np.abs(err))), "rmse_kwh": float(np.sqrt(np.mean(err ** 2))),
            "bias_kwh": float(np.mean(err)), "mean_measured_kwh": float(np.mean(y)),
            "mae_pct_of_mean": float(100 * np.mean(np.abs(err)) / np.mean(y)) if np.mean(y) > 0 else float("nan"),
            "n": int(len(y))}


def predict_all(models: dict, test: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = test[["appliance", "start_time", TARGET, "baseline_kwh"]].copy()
    out["pred_baseline"] = test["baseline_kwh"].to_numpy()
    for fam, m in models.items():
        out[f"pred_{fam}"] = np.clip(m.predict(test[feature_cols]), 0, None)
    return out


def evaluate_models(preds: pd.DataFrame, tz: str) -> dict:
    cols = [c for c in preds.columns if c.startswith("pred_")]
    overall = {c[5:]: metrics(preds[TARGET], preds[c]) for c in cols}
    per_app = {c[5:]: {a: metrics(g[TARGET], g[c]) for a, g in preds.groupby("appliance")} for c in cols}
    day = preds.assign(day=pd.DatetimeIndex(preds["start_time"]).tz_convert(tz).tz_localize(None).normalize())
    daily = day.groupby(["appliance", "day"])[[TARGET] + cols].sum().reset_index()
    daily_m = {c[5:]: metrics(daily[TARGET], daily[c]) for c in cols}
    return {"overall_session": overall, "per_appliance_session": per_app, "daily_appliance": daily_m}


def make_figures(preds: pd.DataFrame, results: dict, selected: str, out_dir) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    model_key = selected if selected in results["overall_session"] else "baseline"
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(preds[TARGET], preds[f"pred_{model_key}"], s=6, alpha=0.4)
    lim = [0, max(preds[TARGET].max(), preds[f"pred_{model_key}"].max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=1, label="perfect")
    ax.set(xlabel="Measured energy per session (kWh)", ylabel="Predicted energy per session (kWh)",
           title=f"Test set: {model_key} vs measured", xlim=lim, ylim=lim)
    ax.legend()
    p = out_dir / "pred_vs_measured.png"; fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); paths.append(str(p))

    apps = sorted(preds["appliance"].unique())
    fams = list(results["overall_session"])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    w = 0.8 / len(fams)
    for i, f in enumerate(fams):
        ax.bar(np.arange(len(apps)) + i * w, [results["per_appliance_session"][f][a]["mae_kwh"] for a in apps], w, label=f)
    ax.set_xticks(np.arange(len(apps)) + 0.4 - w / 2); ax.set_xticklabels(apps, rotation=30, ha="right")
    ax.set(ylabel="MAE per session (kWh)", title="Test-set error by appliance (lower is better)")
    ax.legend(fontsize=8)
    p = out_dir / "mae_by_appliance.png"; fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); paths.append(str(p))
    return paths


def evaluation_markdown(results: dict, train_result, split_info, cfg, model_version: str, thin: list[str]) -> str:
    d = cfg.dataset
    L = ["# Evaluation report", ""]
    if d.synthetic:
        L += ["> **SYNTHETIC DATA** - this is a pipeline demonstration, NOT a measured benchmark (risk R-05). "
              "Do not quote these numbers as model performance.", ""]
    L += [f"- Model version: `{model_version}`", f"- Dataset: {d.name} (version: {d.version}; licence: {d.licence})",
          f"- Seed: {cfg.modeling.seed}", f"- Split: {split_info.method}; purge window {split_info.purge_hours:g} h",
          f"  - train: {split_info.n_train} sessions (ends {split_info.train_end})",
          f"  - validation: {split_info.n_val} sessions ({split_info.val_start} -> {split_info.val_end})",
          f"  - test: {split_info.n_test} sessions (starts {split_info.test_start}); never used for tuning",
          f"- Target: measured session energy (kWh) = integral of measured power over the session",
          f"- Features: appliance, duration, rated power, baseline, time-of-day/weekday/month"
          + (f", history mean power of previous {cfg.modeling.history_window} sessions" if cfg.modeling.use_history_feature else ""),
          "", "## Model selection (validation set)", "",
          "| Model | Validation MAE (kWh) | Best params |", "|---|---|---|"]
    for k, v in train_result.validation_mae.items():
        L.append(f"| {k} | {v:.5f} | {train_result.best_params.get(k, '-')} |")
    L += ["", f"Best ML vs baseline on validation: **{train_result.improvement_pct:+.1f}%** MAE improvement "
              f"(threshold {cfg.modeling.min_improvement_pct:g}%).",
          f"**Decision: `{train_result.selected}`** - " + ("ML is justified." if train_result.ml_justified else
          "ML is NOT justified; the validated rated-power baseline is deployed and ML is deferred (see M9)."),
          "", "## Test-set results (session level)", "",
          "| Model | MAE (kWh) | RMSE (kWh) | Bias (kWh) | MAE % of mean |", "|---|---|---|---|---|"]
    for k, m in results["overall_session"].items():
        L.append(f"| {k} | {m['mae_kwh']:.5f} | {m['rmse_kwh']:.5f} | {m['bias_kwh']:+.5f} | {m['mae_pct_of_mean']:.1f}% |")
    L += ["", "## Test-set results (daily energy per appliance)", "", "| Model | MAE (kWh/day) | RMSE (kWh/day) |", "|---|---|---|"]
    for k, m in results["daily_appliance"].items():
        L.append(f"| {k} | {m['mae_kwh']:.4f} | {m['rmse_kwh']:.4f} |")
    L += ["", "## Per-appliance MAE on test (kWh per session)", ""]
    fams = list(results["overall_session"])
    L += ["| Appliance | n | " + " | ".join(fams) + " |", "|---|---|" + "---|" * len(fams)]
    for a in sorted(results["per_appliance_session"][fams[0]]):
        n = results["per_appliance_session"][fams[0]][a]["n"]
        L.append(f"| {a} | {n} | " + " | ".join(f"{results['per_appliance_session'][f][a]['mae_kwh']:.5f}" for f in fams) + " |")
    L += ["", "![pred vs measured](figures/pred_vs_measured.png)", "", "![MAE by appliance](figures/mae_by_appliance.png)", "",
          "## Limitations and open items (TBD)", ""]
    if thin:
        L.append(f"- Appliances with too few sessions in train or test (interpret with care): {', '.join(thin)}")
    L += ["- Rated powers in the config are assumptions unless replaced with nameplate values (risk R-01).",
          "- On/off thresholds are per-appliance assumptions and need validation (risk R-02).",
          "- UK-recorded data does not represent Indian appliances/usage; validate with local data before production claims (risk R-03).",
          "- Single-household chronological holdout; hold-out-by-household not evaluated unless more houses are added.",
          "- Tariff is a placeholder unless replaced; cost is decision support, not a bill.",
          "- Dataset licence/version, owners, approvals: TBD until confirmed by the project lead.", ""]
    return "\n".join(L)
