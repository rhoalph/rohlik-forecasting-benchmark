"""Official Kaggle grid alignment and weight handling."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from eval.metrics import MetricResult, score_metrics


OFFICIAL_TEST_ORIGIN = pd.Timestamp("2024-06-02")
KEY_COLUMNS = ("unique_id", "date")


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, frame_name: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise KeyError(f"{frame_name} is missing columns: {sorted(missing)}.")


def _normalized_key_frame(frame: pd.DataFrame, *, frame_name: str) -> pd.DataFrame:
    _require_columns(frame, KEY_COLUMNS, frame_name=frame_name)
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    if result["date"].isna().any():
        raise ValueError(f"{frame_name}.date contains missing or invalid values.")
    if result.loc[:, KEY_COLUMNS].duplicated().any():
        raise ValueError(f"{frame_name} contains duplicate {KEY_COLUMNS} keys.")
    return result


def build_official_test_grid(
    sales_test: pd.DataFrame,
    test_weights: pd.DataFrame,
    *,
    origin: object = OFFICIAL_TEST_ORIGIN,
    expected_horizon_days: int = 14,
) -> pd.DataFrame:
    """Validate the public test grid and attach official per-ID sample weights."""

    test = _normalized_key_frame(sales_test, frame_name="sales_test")
    _require_columns(test_weights, ("unique_id", "weight"), frame_name="test_weights")
    if test_weights["unique_id"].duplicated().any():
        raise ValueError("test_weights contains duplicate unique_id values.")
    weights = test_weights.loc[:, ["unique_id", "weight"]].copy()
    weights["weight"] = pd.to_numeric(weights["weight"], errors="coerce")
    if weights["weight"].isna().any() or not np.isfinite(weights["weight"]).all():
        raise ValueError("test_weights.weight contains missing or non-finite values.")
    if (weights["weight"] < 0).any():
        raise ValueError("test_weights.weight contains negative values.")

    normalized_origin = pd.Timestamp(origin).normalize()
    horizon_day = (test["date"] - normalized_origin).dt.days
    expected_days = set(range(1, expected_horizon_days + 1))
    actual_days = set(horizon_day.unique().tolist())
    if actual_days != expected_days:
        raise ValueError(
            f"Official test horizon days are {sorted(actual_days)}, expected {sorted(expected_days)}."
        )

    grid = pd.DataFrame(
        {
            "grid_order": np.arange(len(test), dtype=np.int64),
            "unique_id": test["unique_id"].to_numpy(),
            "test_date": test["date"].to_numpy(),
            "horizon_day": horizon_day.to_numpy(dtype=np.int16),
        }
    )
    grid = grid.merge(weights, on="unique_id", how="left", validate="many_to_one")
    if grid["weight"].isna().any():
        missing_ids = sorted(grid.loc[grid["weight"].isna(), "unique_id"].unique().tolist())
        raise ValueError(f"Official grid IDs lack test weights: {missing_ids[:10]}.")
    return grid.sort_values("grid_order", kind="stable").reset_index(drop=True)


def shift_official_grid(official_grid: pd.DataFrame, cutoff: object) -> pd.DataFrame:
    """Shift the official (`unique_id`, horizon-day) mask to a local cutoff."""

    _require_columns(
        official_grid,
        ("grid_order", "unique_id", "test_date", "horizon_day", "weight"),
        frame_name="official_grid",
    )
    normalized_cutoff = pd.Timestamp(cutoff).normalize()
    shifted = official_grid.copy()
    shifted["date"] = normalized_cutoff + pd.to_timedelta(shifted["horizon_day"], unit="D")
    if shifted.loc[:, KEY_COLUMNS].duplicated().any():
        raise ValueError("Shifted official grid contains duplicate item-date keys.")
    return shifted


def validate_solution_template(official_grid: pd.DataFrame, solution: pd.DataFrame) -> None:
    """Assert that a solution template has exactly the official ordered IDs."""

    _require_columns(solution, ("id",), frame_name="solution")
    if solution["id"].duplicated().any():
        raise ValueError("solution contains duplicate id values.")
    expected = (
        official_grid["unique_id"].astype(str)
        + "_"
        + pd.to_datetime(official_grid["test_date"]).dt.strftime("%Y-%m-%d")
    )
    actual = solution["id"].astype(str).reset_index(drop=True)
    if not actual.equals(expected.reset_index(drop=True)):
        raise ValueError("solution IDs or ordering do not exactly match the official test grid.")


def score_kaggle_aligned(
    labels: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    target_col: str = "sales",
    prediction_col: str = "sales_hat",
) -> MetricResult:
    """Score predictions after enforcing an exact one-to-one grid-key match."""

    label_frame = _normalized_key_frame(labels, frame_name="labels")
    prediction_frame = _normalized_key_frame(predictions, frame_name="predictions")
    _require_columns(label_frame, (target_col, "weight"), frame_name="labels")
    _require_columns(prediction_frame, (prediction_col,), frame_name="predictions")

    aligned = label_frame.loc[:, [*KEY_COLUMNS, target_col, "weight"]].merge(
        prediction_frame.loc[:, [*KEY_COLUMNS, prediction_col]],
        on=list(KEY_COLUMNS),
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if not aligned["_merge"].eq("both").all():
        missing_predictions = int(aligned["_merge"].eq("left_only").sum())
        extra_predictions = int(aligned["_merge"].eq("right_only").sum())
        raise ValueError(
            "Prediction grid does not exactly match label grid: "
            f"missing={missing_predictions}, extra={extra_predictions}."
        )
    return score_metrics(
        aligned[target_col],
        aligned[prediction_col],
        aligned["weight"],
    )
