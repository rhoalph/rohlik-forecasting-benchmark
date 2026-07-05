from __future__ import annotations

import numpy as np
import pandas as pd

from features.stage3_minimal import FORBIDDEN_FEATURE_FIELDS
from scripts.run_stage5_horizon_target_experiments import (
    FOLD_SPECS,
    HORIZON_DAYS,
    S5B_FEATURE_SET,
    _clip_predictions,
    _inverse_sqrt_predictions,
    _partition_by_horizon,
    _predict_horizon_models,
)


class _DummyModel:
    def __init__(self, horizon: int) -> None:
        self.horizon = horizon

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.full(len(features), float(self.horizon), dtype=np.float64)


def test_stage5_horizon_feature_contract_includes_price_discount_batch_only() -> None:
    assert len(S5B_FEATURE_SET) == 46
    assert "price_relative_to_item_history_median" in S5B_FEATURE_SET
    assert "discount_depth_relative_to_item_history" in S5B_FEATURE_SET
    assert not FORBIDDEN_FEATURE_FIELDS.intersection(S5B_FEATURE_SET)


def test_stage5_horizon_fold_specification_is_the_approved_f1_through_f4_schedule() -> None:
    assert [spec.name for spec in FOLD_SPECS] == ["F1", "F2", "F3", "F4"]
    assert [str(spec.cutoff.date()) for spec in FOLD_SPECS] == [
        "2024-05-19",
        "2024-05-05",
        "2024-04-21",
        "2024-04-07",
    ]
    assert len(HORIZON_DAYS) == 14


def test_stage5_horizon_partition_and_routing_use_the_matching_horizon_model() -> None:
    frame = pd.DataFrame(
        {
            "horizon_day": [1, 2, 1, 3],
            "x": [10.0, 20.0, 30.0, 40.0],
        }
    )
    partitions = _partition_by_horizon(frame, horizon_column="horizon_day")
    assert len(partitions[1]) == 2
    assert len(partitions[2]) == 1
    assert len(partitions[3]) == 1

    models = {horizon: _DummyModel(horizon) for horizon in (1, 2, 3)}
    prediction = _predict_horizon_models(
        models,
        frame,
        feature_columns=("horizon_day", "x"),
        horizon_column="horizon_day",
    )
    assert np.array_equal(prediction, np.array([1.0, 2.0, 1.0, 3.0]))


def test_stage5_sqrt_inverse_and_clipping_helpers_are_monotone() -> None:
    clipped, changed = _clip_predictions(np.array([-1.5, 0.0, 2.0], dtype=np.float64))
    assert np.array_equal(clipped, np.array([0.0, 0.0, 2.0]))
    assert changed == 1

    inverse_clipped, negatives = _inverse_sqrt_predictions(
        np.array([-2.0, 0.0, 3.0], dtype=np.float64),
        clip_negative=True,
    )
    inverse_unclipped, negatives_unclipped = _inverse_sqrt_predictions(
        np.array([-2.0, 0.0, 3.0], dtype=np.float64),
        clip_negative=False,
    )
    assert np.array_equal(inverse_clipped, np.array([0.0, 0.0, 9.0]))
    assert np.array_equal(inverse_unclipped, np.array([4.0, 0.0, 9.0]))
    assert negatives == 1
    assert negatives_unclipped == 1

