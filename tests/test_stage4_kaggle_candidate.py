from __future__ import annotations

import pandas as pd
import pytest

from eval.grid import build_official_test_grid, validate_solution_template
from scripts.run_stage4_kaggle_candidate import (
    FINAL_CUTOFF,
    OFFICIAL_FORECAST_END,
    OFFICIAL_FORECAST_START,
    _build_official_request_frame,
    _validate_submission_frame,
)
from scripts.run_stage3_all_folds_plain_model import training_origins_for_fold


def test_stage4_candidate_final_horizon_contract() -> None:
    assert FINAL_CUTOFF == pd.Timestamp("2024-06-02")
    assert OFFICIAL_FORECAST_START == pd.Timestamp("2024-06-03")
    assert OFFICIAL_FORECAST_END == pd.Timestamp("2024-06-16")


def test_stage4_candidate_uses_twelve_origins_before_final_cutoff() -> None:
    origins = training_origins_for_fold(FINAL_CUTOFF)
    assert len(origins) == 12
    assert origins[0] == pd.Timestamp("2024-05-19")
    assert origins[-1] == pd.Timestamp("2023-12-17")
    assert all(left - right == pd.Timedelta(days=14) for left, right in zip(origins, origins[1:]))


def test_stage4_candidate_request_frame_drops_total_orders() -> None:
    dates = pd.date_range("2024-06-03", "2024-06-16")
    sales_test = pd.DataFrame(
        {
            "unique_id": [1] * len(dates),
            "date": dates,
            "warehouse": ["Prague_1"] * len(dates),
            "total_orders": [123.0] * len(dates),
            "sell_price_main": [10.5] * len(dates),
            "type_0_discount": [0.0] * len(dates),
            "type_1_discount": [0.0] * len(dates),
            "type_2_discount": [0.0] * len(dates),
            "type_3_discount": [0.0] * len(dates),
            "type_4_discount": [0.0] * len(dates),
            "type_5_discount": [0.0] * len(dates),
            "type_6_discount": [0.0] * len(dates),
        }
    )
    weights = pd.DataFrame({"unique_id": [1], "weight": [1.0]})
    solution = pd.DataFrame({"id": [f"1_{date:%Y-%m-%d}" for date in dates], "sales_hat": [0.0] * len(dates)})
    official_grid = build_official_test_grid(sales_test, weights)
    validate_solution_template(official_grid, solution)

    request_frame = _build_official_request_frame(sales_test, official_grid)
    assert "total_orders" not in request_frame.columns
    assert request_frame.loc[0, "horizon_day"] == 1
    assert request_frame.loc[0, "sell_price_main"] == pytest.approx(10.5)


def test_stage4_candidate_submission_validation_enforces_order_and_shape() -> None:
    official_grid = pd.DataFrame(
        {
            "unique_id": [1, 2],
            "test_date": pd.to_datetime(["2024-06-03", "2024-06-04"]),
        }
    )
    official_grid["grid_order"] = [0, 1]
    official_grid["horizon_day"] = [1, 2]
    official_grid["weight"] = [1.0, 1.0]
    solution = pd.DataFrame({"id": ["1_2024-06-03", "2_2024-06-04"]})
    submission = pd.DataFrame({"id": ["1_2024-06-03", "2_2024-06-04"], "sales_hat": [1.0, 2.0]})
    _validate_submission_frame(official_grid, solution, submission)

    with pytest.raises(AssertionError, match="duplicate ids"):
        _validate_submission_frame(
            official_grid,
            solution,
            pd.DataFrame({"id": ["1_2024-06-03", "1_2024-06-03"], "sales_hat": [1.0, 2.0]}),
        )

    with pytest.raises(AssertionError, match="order does not match"):
        _validate_submission_frame(
            official_grid,
            solution,
            pd.DataFrame({"id": ["2_2024-06-04", "1_2024-06-03"], "sales_hat": [1.0, 2.0]}),
        )
