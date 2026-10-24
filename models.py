"""Candidate regressors: Random Forest, Gradient Boosting, and a hybrid baseline + ML residual model."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .features import CATEGORICAL

PARAM_GRIDS = {
    "random_forest": {"n_estimators": [200], "min_samples_leaf": [3, 10]},
    "gradient_boosting": {"max_iter": [200, 400], "learning_rate": [0.05, 0.1], "max_leaf_nodes": [15, 31]},
}
MODEL_FAMILIES = ["random_forest", "gradient_boosting", "hybrid_gradient_boosting"]


class ResidualRegressor(BaseEstimator, RegressorMixin):
    """Hybrid (option C): prediction = rated-power baseline + ML-learned correction (>= 0)."""

    def __init__(self, estimator, baseline_col: str = "baseline_kwh"):
        self.estimator = estimator
        self.baseline_col = baseline_col

    def fit(self, X: pd.DataFrame, y):
        base = X[self.baseline_col].to_numpy(float)
        self.estimator_ = clone(self.estimator).fit(X, np.asarray(y, float) - base)
        return self

    def predict(self, X: pd.DataFrame):
        base = X[self.baseline_col].to_numpy(float)
        return np.clip(base + self.estimator_.predict(X), 0.0, None)


def _pipeline(core, numeric_cols: list[str]) -> Pipeline:
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
                             ("num", "passthrough", numeric_cols)])
    return Pipeline([("pre", pre), ("reg", core)])


def build_estimator(family: str, params: dict, feature_cols: list[str], seed: int):
    numeric = [c for c in feature_cols if c not in CATEGORICAL]
    if family == "random_forest":
        return _pipeline(RandomForestRegressor(random_state=seed, n_jobs=-1, **params), numeric)
    if family == "gradient_boosting":
        return _pipeline(HistGradientBoostingRegressor(random_state=seed, **params), numeric)
    if family == "hybrid_gradient_boosting":
        return ResidualRegressor(_pipeline(HistGradientBoostingRegressor(random_state=seed, **params), numeric))
    raise ValueError(f"Unknown model family: {family}")


def param_combinations(family: str) -> list[dict]:
    grid = PARAM_GRIDS["gradient_boosting" if family.endswith("gradient_boosting") else family]
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]
