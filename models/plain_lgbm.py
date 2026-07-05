"""Fixed first-run LightGBM model for Stage 3."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PlainLightGBMConfig:
    objective: str = "regression_l1"
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_data_in_leaf: int = 100
    num_boost_round: int = 300
    seed: int = 42
    num_threads: int = 6

    def parameters(self) -> dict[str, object]:
        return {
            "objective": self.objective,
            "metric": "l1",
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": self.min_data_in_leaf,
            "feature_fraction": 1.0,
            "bagging_fraction": 1.0,
            "bagging_freq": 0,
            "seed": self.seed,
            "feature_fraction_seed": self.seed,
            "bagging_seed": self.seed,
            "data_random_seed": self.seed,
            "deterministic": True,
            "force_col_wise": True,
            "verbosity": -1,
            "num_threads": self.num_threads,
        }

    def audit_dict(self) -> dict[str, object]:
        return asdict(self)


def train_plain_lightgbm(
    features: pd.DataFrame,
    target: pd.Series | np.ndarray,
    categorical_features: tuple[str, ...],
    config: PlainLightGBMConfig,
) -> lgb.Booster:
    """Fit one unweighted global L1 model with no validation-driven tuning."""

    target_array = np.asarray(target, dtype=np.float64)
    if len(features) != len(target_array):
        raise ValueError("Feature and target row counts differ.")
    if not np.isfinite(target_array).all():
        raise ValueError("Training target contains missing or non-finite values.")
    missing_categories = set(categorical_features) - set(features.columns)
    if missing_categories:
        raise KeyError(f"Missing categorical features: {sorted(missing_categories)}.")

    dataset = lgb.Dataset(
        features,
        label=target_array,
        categorical_feature=list(categorical_features),
        free_raw_data=False,
    )
    return lgb.train(
        config.parameters(),
        dataset,
        num_boost_round=config.num_boost_round,
    )


def predict_plain_lightgbm(model: lgb.Booster, features: pd.DataFrame) -> np.ndarray:
    """Return unrounded and unclipped continuous predictions."""

    prediction = np.asarray(model.predict(features), dtype=np.float64)
    if prediction.ndim != 1 or len(prediction) != len(features):
        raise ValueError("Model returned an invalid prediction shape.")
    if not np.isfinite(prediction).all():
        raise ValueError("Model returned missing or non-finite predictions.")
    return prediction
