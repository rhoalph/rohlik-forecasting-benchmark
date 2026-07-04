import pandas as pd
import pytest

from dataguard.availability import (
    Availability,
    UnknownFieldError,
    assert_future_columns_allowed,
    classify_field,
    select_future_covariates,
)


def test_adopted_covariate_policy_is_explicit() -> None:
    assert classify_field("sell_price_main").availability is Availability.KNOWN_FUTURE
    assert classify_field("type_0_discount").availability is Availability.KNOWN_FUTURE
    assert classify_field("type_6_discount").availability is Availability.KNOWN_FUTURE
    assert classify_field("total_orders").availability is Availability.HISTORICAL_ONLY
    assert classify_field("availability").availability is Availability.HISTORICAL_ONLY
    assert classify_field("sales").availability is Availability.TARGET
    assert classify_field("weight").availability is Availability.EVALUATION_ONLY


def test_future_covariate_selection_excludes_unsafe_columns() -> None:
    frame = pd.DataFrame(
        {
            "unique_id": [1],
            "date": ["2024-01-02"],
            "warehouse": ["Prague_1"],
            "sell_price_main": [10.0],
            "type_0_discount": [0.2],
            "total_orders": [1000.0],
            "availability": [1.0],
            "sales": [4.0],
            "weight": [2.0],
        }
    )

    selected = select_future_covariates(frame)

    assert list(selected.columns) == [
        "unique_id",
        "date",
        "warehouse",
        "sell_price_main",
        "type_0_discount",
    ]
    assert_future_columns_allowed(selected.columns)


def test_unknown_fields_fail_closed() -> None:
    with pytest.raises(UnknownFieldError, match="not classified"):
        classify_field("unreviewed_feature")
    with pytest.raises(UnknownFieldError, match="not classified"):
        select_future_covariates(pd.DataFrame({"unreviewed_feature": [1]}))


def test_assert_future_columns_rejects_historical_target_and_weight() -> None:
    with pytest.raises(ValueError, match="total_orders"):
        assert_future_columns_allowed(["unique_id", "total_orders"])
    with pytest.raises(ValueError, match="sales"):
        assert_future_columns_allowed(["sales"])
    with pytest.raises(ValueError, match="weight"):
        assert_future_columns_allowed(["weight"])
