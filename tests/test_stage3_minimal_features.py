import pandas as pd
import pytest

from dataguard.cutoff import LeakageError
from features.stage3_minimal import (
    APPROVED_FEATURES,
    FEATURE_AVAILABILITY,
    FORBIDDEN_FEATURE_FIELDS,
    build_stage3_feature_batch,
)


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unique_id": [1] * 14,
            "date": pd.date_range("2024-01-01", "2024-01-14"),
            "sales": [float(value) for value in range(1, 15)],
        }
    )


def _requests() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "unique_id": [1, 1],
            "date": pd.to_datetime(["2024-01-15", "2024-01-22"]),
            "warehouse": ["Prague_1", "Prague_1"],
            "horizon_day": [1, 8],
            "sell_price_main": [100.0, 110.0],
        }
    )
    for index in range(7):
        frame[f"type_{index}_discount"] = [0.0, -0.1 if index == 4 else 0.0]
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


def test_stage3_feature_contract_and_cutoff_safe_values() -> None:
    batch = build_stage3_feature_batch(
        _history(),
        _requests(),
        _inventory(),
        _calendar(),
        "2024-01-14",
    )

    assert tuple(batch.matrix.columns) == APPROVED_FEATURES
    assert set(batch.matrix.columns) == set(FEATURE_AVAILABILITY)
    assert not FORBIDDEN_FEATURE_FIELDS.intersection(batch.matrix.columns)
    assert batch.maximum_history_date == pd.Timestamp("2024-01-14")
    assert batch.maximum_same_weekday_source_date == pd.Timestamp("2024-01-08")
    assert batch.matrix.loc[0, "trailing_7_mean"] == pytest.approx(11.0)
    assert batch.matrix.loc[0, "trailing_14_mean"] == pytest.approx(7.5)
    assert batch.matrix.loc[0, "same_weekday_sales"] == pytest.approx(8.0)
    assert batch.matrix.loc[0, "same_weekday_direct_available"] == 1
    assert batch.matrix.loc[1, "same_weekday_sales"] == pytest.approx(14.0)
    assert batch.matrix.loc[1, "same_weekday_direct_available"] == 0
    assert batch.matrix.loc[1, "type_4_discount"] == pytest.approx(-0.1)


def test_stage3_features_reject_post_cutoff_history() -> None:
    history = _history()
    history.loc[len(history)] = [1, pd.Timestamp("2024-01-15"), 999.0]
    with pytest.raises(LeakageError, match="after cutoff"):
        build_stage3_feature_batch(
            history,
            _requests(),
            _inventory(),
            _calendar(),
            "2024-01-14",
        )


@pytest.mark.parametrize("forbidden", ["sales", "total_orders", "availability", "weight"])
def test_stage3_features_reject_forbidden_request_fields(forbidden: str) -> None:
    requests = _requests()
    requests[forbidden] = 1.0
    with pytest.raises(ValueError, match="forbidden fields"):
        build_stage3_feature_batch(
            _history(),
            requests,
            _inventory(),
            _calendar(),
            "2024-01-14",
        )
