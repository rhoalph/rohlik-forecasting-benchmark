from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.stage5e_stronger_features import (
    APPROVED_STAGE5E_FEATURES,
    STAGE5E_EXTRA_FEATURES,
    build_stage5e_feature_batch,
)
from scripts.run_stage5f_objective_blend_experiments import (
    _blend_predictions,
    _inverse_prediction,
    _transform_target,
)


def _inventory() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unique_id": [1, 2],
            "product_unique_id": [100, 100],
            "warehouse": ["W1", "W2"],
            "L1_category_name_en": ["L1", "L1"],
            "L2_category_name_en": ["L2", "L2"],
            "L3_category_name_en": ["L3", "L3"],
            "L4_category_name_en": ["L4", "L4"],
        }
    )


def _history() -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2024-01-01", "2024-02-04")
    for uid, warehouse, price_base in [(1, "W1", 10.0), (2, "W2", 20.0)]:
        for idx, date in enumerate(dates, start=1):
            rows.append(
                {
                    "unique_id": uid,
                    "date": date,
                    "warehouse": warehouse,
                    "sales": float(idx if uid == 1 else 100 + idx),
                    "sell_price_main": price_base,
                    **{f"type_{discount}_discount": 0.0 for discount in range(7)},
                }
            )
    return pd.DataFrame(rows)


def _requests() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "unique_id": [1, 1, 2, 2],
            "date": pd.to_datetime(["2024-02-05", "2024-02-12", "2024-02-05", "2024-02-12"]),
            "warehouse": ["W1", "W1", "W2", "W2"],
            "horizon_day": [1, 8, 1, 8],
            "day_of_week": [0, 0, 0, 0],
            "sell_price_main": [15.0, 15.0, 25.0, 25.0],
        }
    )
    for index in range(7):
        frame[f"type_{index}_discount"] = 0.0
    return frame


def _calendar() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "warehouse": ["W1", "W1", "W2", "W2"],
            "date": pd.to_datetime(["2024-02-05", "2024-02-12", "2024-02-05", "2024-02-12"]),
            "holiday": [0, 0, 0, 0],
            "shops_closed": [0, 0, 0, 0],
            "winter_school_holidays": [0, 0, 0, 0],
            "school_holidays": [0, 0, 0, 0],
        }
    )


def test_stage5e_contract_is_unchanged() -> None:
    batch = build_stage5e_feature_batch(_history(), _requests(), _inventory(), _calendar(), "2024-02-04")
    assert tuple(batch.matrix.columns) == APPROVED_STAGE5E_FEATURES
    assert len(APPROVED_STAGE5E_FEATURES) == 76
    assert len(STAGE5E_EXTRA_FEATURES) == 30


def test_target_transforms_round_trip() -> None:
    target = pd.Series([0.0, 1.0, 4.0, 9.0], dtype=float)
    sqrt_target, _ = _transform_target(target, "sqrt")
    log_target, _ = _transform_target(target, "log1p")

    sqrt_unclipped, sqrt_final, sqrt_diag = _inverse_prediction(np.array([-2.0, -1.0, 0.0, 3.0]), "sqrt")
    log_unclipped, log_final, log_diag = _inverse_prediction(np.array([-1.0, 0.0, 1.0]), "log1p")

    assert np.allclose(np.square(np.asarray(sqrt_target)), target)
    assert np.allclose(np.expm1(np.asarray(log_target)), target)
    assert sqrt_diag["negative_transformed_count"] == 2
    assert sqrt_diag["negative_final_before_clip"] == 0
    assert sqrt_diag["clipped_rows"] == 0
    assert log_diag["negative_transformed_count"] == 1
    assert log_diag["negative_final_before_clip"] == 1
    assert log_diag["clipped_rows"] == 1
    assert np.all(sqrt_unclipped >= 0.0)
    assert np.all(log_unclipped > -1.0)
    assert np.all(sqrt_final >= 0.0)
    assert np.all(log_final >= 0.0)


def test_blend_predictions_require_exact_alignment_and_unit_weights() -> None:
    keys = pd.DataFrame({"unique_id": [1, 2], "date": pd.to_datetime(["2024-02-05", "2024-02-05"])})
    other_keys = pd.DataFrame({"unique_id": [2, 1], "date": pd.to_datetime(["2024-02-05", "2024-02-05"])})
    predictions = [
        {"keys": keys, "prediction_final": np.array([1.0, 2.0])},
        {"keys": keys.copy(), "prediction_final": np.array([3.0, 4.0])},
    ]
    blended, clipped, audit = _blend_predictions(predictions, [0.25, 0.75])
    assert np.allclose(blended, np.array([2.5, 3.5]))
    assert np.allclose(clipped, np.array([2.5, 3.5]))
    assert audit["clipped_rows"] == 0

    with pytest.raises(ValueError):
        _blend_predictions(predictions, [0.2, 0.2])

    with pytest.raises(ValueError):
        _blend_predictions([predictions[0], {"keys": other_keys, "prediction_final": np.array([3.0, 4.0])}], [0.5, 0.5])
