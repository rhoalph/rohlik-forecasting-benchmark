from __future__ import annotations

import pandas as pd
import pytest

from features.stage3_minimal import FORBIDDEN_FEATURE_FIELDS
from features.stage5e_stronger_features import (
    APPROVED_STAGE5E_FEATURES,
    FEATURE_AVAILABILITY,
    STAGE5E_EXTRA_FEATURES,
    build_stage5e_feature_batch,
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
            "date": pd.to_datetime(
                ["2024-02-05", "2024-02-12", "2024-02-05", "2024-02-12"]
            ),
            "warehouse": ["W1", "W1", "W2", "W2"],
            "horizon_day": [1, 8, 1, 8],
            "day_of_week": [0, 0, 0, 0],
            "sell_price_main": [15.0, 15.0, 25.0, 25.0],
        }
    )
    for index in range(7):
        frame[f"type_{index}_discount"] = 0.0
    frame.loc[0, "type_2_discount"] = 0.2
    return frame


def _calendar() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "warehouse": ["W1", "W1", "W2", "W2"],
            "date": pd.to_datetime(["2024-02-05", "2024-02-12", "2024-02-05", "2024-02-12"]),
            "holiday": [0, 0, 0, 0],
            "shops_closed": [0, 0, 0, 0],
            "winter_school_holidays": [0, 0, 0, 0],
            "school_holidays": [0, 0, 0, 0],
        }
    )
    return frame


def test_stage5e_feature_contract_is_closed_and_allowlisted() -> None:
    batch = build_stage5e_feature_batch(_history(), _requests(), _inventory(), _calendar(), "2024-02-04")
    assert tuple(batch.matrix.columns) == APPROVED_STAGE5E_FEATURES
    assert set(FEATURE_AVAILABILITY) == set(APPROVED_STAGE5E_FEATURES)
    assert len(STAGE5E_EXTRA_FEATURES) == 30
    assert not FORBIDDEN_FEATURE_FIELDS.intersection(batch.matrix.columns)


def test_stage5e_cutoff_safe_lag_and_fallback_features() -> None:
    batch = build_stage5e_feature_batch(_history(), _requests(), _inventory(), _calendar(), "2024-02-04")

    assert batch.matrix.loc[0, "lag_7_sales"] == pytest.approx(29.0)
    assert batch.matrix.loc[1, "lag_7_sales"] == pytest.approx(35.0)
    assert batch.matrix.loc[0, "lag_14_sales"] == pytest.approx(22.0)
    assert batch.matrix.loc[0, "same_weekday_2w_sales"] == pytest.approx(22.0)
    assert batch.matrix.loc[0, "rolling_28_mean_sales"] == pytest.approx(21.5)
    assert batch.matrix.loc[0, "rolling_7_median_sales"] == pytest.approx(32.0)
    assert batch.matrix.loc[0, "rolling_14_median_sales"] == pytest.approx(28.5)
    assert batch.matrix.loc[0, "recent_trend_7_vs_28"] == pytest.approx(10.5)


def test_stage5e_group_and_interaction_features_remain_cutoff_safe() -> None:
    batch = build_stage5e_feature_batch(_history(), _requests(), _inventory(), _calendar(), "2024-02-04")

    assert batch.matrix.loc[0, "warehouse_category_l2_mean_sales"] == pytest.approx(18.0)
    assert batch.matrix.loc[2, "warehouse_category_l2_mean_sales"] == pytest.approx(118.0)
    assert batch.matrix.loc[0, "product_mean_sales_across_warehouses"] == pytest.approx(68.0)
    assert batch.matrix.loc[1, "product_median_sales_across_warehouses"] == pytest.approx(68.0)
    assert batch.matrix.loc[0, "category_l2_mean_sales"] == pytest.approx(68.0)
    assert batch.matrix.loc[0, "item_share_of_warehouse_category_l2_sales"] == pytest.approx(1.0)
    assert batch.matrix.loc[0, "discount_active_count"] == 1
    assert batch.matrix.loc[0, "price_x_any_discount"] == pytest.approx(15.0)
    assert batch.matrix.loc[0, "price_x_max_discount"] == pytest.approx(3.0)
    assert batch.matrix.loc[0, "horizon_x_any_discount"] == pytest.approx(1.0)
    assert batch.matrix.loc[1, "horizon_x_any_discount"] == pytest.approx(0.0)
    assert batch.matrix.loc[0, "dayofweek_x_any_discount"] == pytest.approx(0.0)


def test_stage5e_features_reject_future_history_and_forbidden_fields() -> None:
    history = _history()
    future_row = history.iloc[[0]].copy()
    future_row["date"] = pd.Timestamp("2024-02-05")
    history = pd.concat([history, future_row], ignore_index=True)
    with pytest.raises((ValueError, AssertionError)):
        build_stage5e_feature_batch(history, _requests(), _inventory(), _calendar(), "2024-02-04")

    requests = _requests()
    requests["total_orders"] = 1.0
    with pytest.raises(ValueError):
        build_stage5e_feature_batch(_history(), requests, _inventory(), _calendar(), "2024-02-04")
