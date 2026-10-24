"""FR-06: train candidate regressors and decide whether ML is justified vs. the baseline.

Protocol (no test-set tuning):
  1. For each model family, fit every hyper-parameter combo on TRAIN and score MAE on VALIDATION.
  2. Keep the best combo per family; compare the best family with the rated-power baseline on VALIDATION.
  3. ML is "justified" only if it beats the baseline by >= ``min_improvement_pct`` on validation.
  4. Re-fit each family's chosen combo on TRAIN+VALIDATION (the test set is untouched until evaluate.py).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .features import TARGET, assert_no_target_leakage
from .models import MODEL_FAMILIES, build_estimator, param_combinations

log = logging.getLogger(__name__)


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


@dataclass
class TrainResult:
    models: dict                       # family -> fitted estimator (train+val)
    best_params: dict                  # family -> params chosen on validation
    validation_mae: dict               # family -> val MAE (train-only fit); includes "baseline"
    selected: str                      # family name or "baseline"
    ml_justified: bool
    improvement_pct: float
    feature_cols: list = field(default_factory=list)


def train_and_select(train: pd.DataFrame, val: pd.DataFrame, feature_cols: list[str], seed: int,
                     min_improvement_pct: float, families: list[str] | None = None) -> TrainResult:
    assert_no_target_leakage(feature_cols)
    families = families or MODEL_FAMILIES
    Xtr, ytr = train[feature_cols], train[TARGET]
    Xva, yva = val[feature_cols], val[TARGET]

    val_mae = {"baseline": mae(yva, val["baseline_kwh"])}
    best_params: dict = {}
    for fam in families:
        best = (np.inf, None)
        for params in param_combinations(fam):
            est = build_estimator(fam, params, feature_cols, seed).fit(Xtr, ytr)
            score = mae(yva, est.predict(Xva))
            log.info("  %-26s %s -> val MAE %.5f kWh", fam, params, score)
            if score < best[0]:
                best = (score, params)
        val_mae[fam], best_params[fam] = best

    best_family = min(families, key=lambda f: val_mae[f])
    base = val_mae["baseline"]
    improvement = 100.0 * (base - val_mae[best_family]) / base if base > 0 else 0.0
    justified = improvement >= min_improvement_pct
    selected = best_family if justified else "baseline"

    both = pd.concat([train, val]).sort_values("start_time")
    models = {fam: build_estimator(fam, best_params[fam], feature_cols, seed).fit(both[feature_cols], both[TARGET])
              for fam in families}
    return TrainResult(models, best_params, val_mae, selected, justified, improvement, feature_cols)
