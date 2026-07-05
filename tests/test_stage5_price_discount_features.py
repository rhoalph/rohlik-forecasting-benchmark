from __future__ import annotations

import pandas as pd
import pytest

from features.stage5_price_discount import (
    APPROVED_STAGE5_FEATURES,
    FEATURE_AVAILABILITY,
    PRICE_DISCOUNT_FEATURES,
    build_stage5_feature_batch,
)


def _history() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "unique_id": [1] * 14,
            "date": pd.date_range("2024-01-01", "2024-01-14"),
            "sales": [float(value) for value in range(1, 15)],
            "sell_price_main": [10.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0, 14.0, 16.0, 15.0, 17.0, 16.0, 18.0],
        }
    )
    for index in range(7):
        frame[f"type_{index}_discount"] = [0.0] * len(frame)
    return frame


def _requests() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "unique_id": [1, 1],
            "date": pd.to_datetime(["2024-01-15", "2024-01-22"]),
            "warehouse": ["Prague_1", "Prague_1"],
            "horizon_day": [1, 8],
            "sell_price_main": [20.0, 15.0],
        }
    )
    for index in range(7):
        frame[f"type_{index}_discount"] = [0.0, 0.05 if index == 4 else 0.0]
    return frame


def _inventory() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unique_id": [1],
            "product_unique_id": [10],
            "warehouse": ["Prague_1"],
            "L1_category_name_en": ["L1"],
            "L2_category_name_en": ["L2"],
            "L3_category_name_en": ["L3"],
            "L4_category_name_en": ["L4"],
        }
    )


def _calendar() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "warehouse": ["Prague_1", "Prague_1"],
            "date": pd.to_datetime(["2024-01-15", "2024-01-22"]),
            "holiday": [0, 1],
            "shops_closed": [0, 0],
            "winter_school_holidays": [0, 1],
            "school_holidays": [0, 0],
        }
    )


def test_stage5_feature_contract_and_cutoff_safe_price_features() -> None:
    batch = build_stage5_feature_batch(
        _history(),
        _requests(),
        _inventory(),
        _calendar(),
        "2024-01-14",
    )

    assert tuple(batch.matrix.columns) == APPROVED_STAGE5_FEATURES
    assert set(batch.matrix.columns) == set(FEATURE_AVAILABILITY)
    assert len(PRICE_DISCOUNT_FEATURES) == 10
    assert batch.matrix.loc[0, "price_relative_to_item_history_median"] == pytest.approx(20.0 / 14.0)
    assert batch.matrix.loc[0, "price_relative_to_item_history_mean"] == pytest.approx(20.0 / 14.0)
    assert batch.matrix.loc[0, "price_change_vs_last_observed"] == pytest.approx(2.0)
    assert batch.matrix.loc[0, "any_discount_active"] == 0
    assert batch.matrix.loc[0, "max_discount"] == pytest.approx(0.0)
    assert batch.matrix.loc[0, "total_discount"] == pytest.approx(0.0)
    assert batch.matrix.loc[0, "active_discount_count"] == 0
    assert batch.matrix.loc[1, "any_discount_active"] == 1
    assert batch.matrix.loc[1, "active_discount_count"] == 1
    assert batch.matrix.loc[1, "total_discount"] == pytest.approx(0.05)


def test_stage5_features_reject_forbidden_request_fields() -> None:
    requests = _requests()
    requests["total_orders"] = 1.0
    with pytest.raises(ValueError):
        build_stage5_feature_batch(_history(), requests, _inventory(), _calendar(), "2024-01-14")
