"""Cutoff-safe minimal features for the approved Stage 3 F1 model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from dataguard.cutoff import (
    assert_history_at_or_before_cutoff,
    assert_source_dates_at_or_before_cutoff,
    assert_target_not_present,
    assert_validation_within_window,
)


KEY_COLUMNS: Final[tuple[str, str]] = ("unique_id", "date")
DISCOUNT_COLUMNS: Final[tuple[str, ...]] = tuple(
    f"type_{index}_discount" for index in range(7)
)
STATIC_CATEGORY_COLUMNS: Final[tuple[str, ...]] = (
    "warehouse",
    "L1_category_name_en",
    "L2_category_name_en",
    "L3_category_name_en",
    "L4_category_name_en",
)
CATEGORICAL_FEATURES: Final[tuple[str, ...]] = (
    "unique_id",
    "product_unique_id",
    *STATIC_CATEGORY_COLUMNS,
    "day_of_week",
    "month",
)
APPROVED_FEATURES: Final[tuple[str, ...]] = (
    "unique_id",
    "product_unique_id",
    *STATIC_CATEGORY_COLUMNS,
    "horizon_day",
    "day_of_week",
    "iso_week",
    "month",
    "weekend_flag",
    "holiday",
    "shops_closed",
    "winter_school_holidays",
    "school_holidays",
    "sell_price_main",
    *DISCOUNT_COLUMNS,
    "last_observed_sales",
    "last_observed_available",
    "trailing_7_mean",
    "trailing_7_available",
    "trailing_14_mean",
    "trailing_14_available",
    "same_weekday_sales",
    "same_weekday_direct_available",
    "historical_mean_sales",
    "historical_median_sales",
    "historical_stats_available",
    "observed_history_row_count",
)

FEATURE_AVAILABILITY: Final[dict[str, str]] = {
    "unique_id": "static metadata",
    "product_unique_id": "static metadata",
    **{column: "static metadata" for column in STATIC_CATEGORY_COLUMNS},
    "horizon_day": "known future",
    "day_of_week": "known future",
    "iso_week": "known future",
    "month": "known future",
    "weekend_flag": "known future",
    "holiday": "known future",
    "shops_closed": "known future",
    "winter_school_holidays": "known future",
    "school_holidays": "known future",
    "sell_price_main": "known future",
    **{column: "known future" for column in DISCOUNT_COLUMNS},
    "last_observed_sales": "historical only",
    "last_observed_available": "historical only",
    "trailing_7_mean": "historical only",
    "trailing_7_available": "historical only",
    "trailing_14_mean": "historical only",
    "trailing_14_available": "historical only",
    "same_weekday_sales": "historical only",
    "same_weekday_direct_available": "historical only",
    "historical_mean_sales": "historical only",
    "historical_median_sales": "historical only",
    "historical_stats_available": "historical only",
    "observed_history_row_count": "historical only",
}
FORBIDDEN_FEATURE_FIELDS: Final[frozenset[str]] = frozenset(
    {"sales", "weight", "total_orders", "availability", "id", "sales_hat"}
)


@dataclass(frozen=True)
class FeatureBatch:
    """One ordered feature matrix and its leakage-audit lineage."""

    keys: pd.DataFrame
    matrix: pd.DataFrame
    origin: pd.Timestamp
    maximum_history_date: pd.Timestamp
    maximum_same_weekday_source_date: pd.Timestamp | None


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"{name} is missing required columns: {sorted(missing)}.")


def _assert_unique_keys(frame: pd.DataFrame, name: str) -> None:
    if frame.loc[:, KEY_COLUMNS].duplicated().any():
        raise ValueError(f"{name} contains duplicate {KEY_COLUMNS} keys.")


def _encode_static_categories(
    frame: pd.DataFrame,
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    encoded = frame.copy()
    for column in STATIC_CATEGORY_COLUMNS:
        vocabulary = sorted(inventory[column].astype(str).unique().tolist())
        mapping = {value: code for code, value in enumerate(vocabulary)}
        values = encoded[column].astype(str).map(mapping)
        if values.isna().any():
            raise ValueError(f"Feature {column!r} contains a value absent from inventory.")
        encoded[column] = values.astype(np.int32)
    return encoded


def _fill_historical_statistic(
    primary: pd.Series,
    last_observed: pd.Series,
    global_median: float,
) -> pd.Series:
    return (
        pd.to_numeric(primary, errors="coerce")
        .fillna(last_observed)
        .fillna(global_median)
        .astype(np.float64)
    )


def build_stage3_feature_batch(
    training_history: pd.DataFrame,
    request_covariates: pd.DataFrame,
    inventory: pd.DataFrame,
    calendar: pd.DataFrame,
    origin: object,
) -> FeatureBatch:
    """Build the exact approved Stage 3 matrix from one forecast origin.

    ``training_history`` may contain sales only through ``origin``.
    ``request_covariates`` must already be label-free and may contain only the
    approved request keys, horizon, warehouse, price, and discount columns.
    """

    required_history = {"unique_id", "date", "sales"}
    required_request = {
        "unique_id",
        "date",
        "warehouse",
        "horizon_day",
        "sell_price_main",
        *DISCOUNT_COLUMNS,
    }
    required_inventory = {
        "unique_id",
        "product_unique_id",
        *STATIC_CATEGORY_COLUMNS,
    }
    required_calendar = {
        "warehouse",
        "date",
        "holiday",
        "shops_closed",
        "winter_school_holidays",
        "school_holidays",
    }
    _require_columns(training_history, required_history, "training_history")
    _require_columns(request_covariates, required_request, "request_covariates")
    _require_columns(inventory, required_inventory, "inventory")
    _require_columns(calendar, required_calendar, "calendar")

    forbidden_request = FORBIDDEN_FEATURE_FIELDS.intersection(request_covariates.columns)
    if forbidden_request:
        raise ValueError(
            "request_covariates contains forbidden fields: "
            f"{sorted(forbidden_request)}."
        )

    normalized_origin = pd.Timestamp(origin).normalize()
    history = training_history.loc[:, ["unique_id", "date", "sales"]].copy()
    history["date"] = pd.to_datetime(history["date"], errors="raise").dt.normalize()
    history["sales"] = pd.to_numeric(history["sales"], errors="coerce")
    if history["sales"].isna().any():
        raise ValueError("training_history contains missing sales.")
    _assert_unique_keys(history, "training_history")
    assert_history_at_or_before_cutoff(history, normalized_origin)
    if history.empty:
        raise ValueError("training_history must not be empty.")

    requests = request_covariates.loc[:, sorted(required_request)].copy()
    requests["date"] = pd.to_datetime(requests["date"], errors="raise").dt.normalize()
    _assert_unique_keys(requests, "request_covariates")
    assert_validation_within_window(requests, normalized_origin, horizon_days=14)
    assert_target_not_present(requests)

    inventory_columns = [
        "unique_id",
        "product_unique_id",
        *STATIC_CATEGORY_COLUMNS,
    ]
    inventory_frame = inventory.loc[:, inventory_columns].copy()
    if inventory_frame["unique_id"].duplicated().any():
        raise ValueError("inventory contains duplicate unique_id values.")

    base = requests.rename(columns={"warehouse": "request_warehouse"}).merge(
        inventory_frame,
        on="unique_id",
        how="left",
        validate="many_to_one",
    )
    if base[inventory_columns[1:]].isna().any().any():
        raise ValueError("One or more request IDs lack inventory metadata.")
    if not base["request_warehouse"].astype(str).eq(base["warehouse"].astype(str)).all():
        raise ValueError("Request warehouse does not match inventory warehouse.")
    base = base.drop(columns="request_warehouse")

    calendar_columns = [
        "warehouse",
        "date",
        "holiday",
        "shops_closed",
        "winter_school_holidays",
        "school_holidays",
    ]
    calendar_frame = calendar.loc[:, calendar_columns].copy()
    calendar_frame["date"] = pd.to_datetime(
        calendar_frame["date"], errors="raise"
    ).dt.normalize()
    if calendar_frame.loc[:, ["warehouse", "date"]].duplicated().any():
        raise ValueError("calendar contains duplicate warehouse/date keys.")
    base = base.merge(
        calendar_frame,
        on=["warehouse", "date"],
        how="left",
        validate="many_to_one",
    )
    calendar_value_columns = calendar_columns[2:]
    if base[calendar_value_columns].isna().any().any():
        raise ValueError("One or more requests lack calendar metadata.")

    history = history.sort_values(["unique_id", "date"], kind="stable")
    global_median = float(history["sales"].median())
    if not np.isfinite(global_median):
        raise ValueError("Origin global median is not finite.")

    last_by_id = (
        history.drop_duplicates("unique_id", keep="last")
        .set_index("unique_id")["sales"]
        .astype(np.float64)
    )
    last_primary = base["unique_id"].map(last_by_id).astype(np.float64)
    base["last_observed_available"] = last_primary.notna().astype(np.int8)
    last_filled = last_primary.fillna(global_median)
    base["last_observed_sales"] = last_filled

    for window_days in (7, 14):
        start = normalized_origin - pd.Timedelta(days=window_days - 1)
        recent = history.loc[history["date"].between(start, normalized_origin)]
        means = recent.groupby("unique_id", observed=True)["sales"].mean()
        primary = base["unique_id"].map(means).astype(np.float64)
        base[f"trailing_{window_days}_available"] = primary.notna().astype(np.int8)
        base[f"trailing_{window_days}_mean"] = _fill_historical_statistic(
            primary,
            last_filled,
            global_median,
        )

    same_weekday_source = base["date"] - pd.Timedelta(days=7)
    safe_source = same_weekday_source <= normalized_origin
    safe_dates = same_weekday_source.loc[safe_source]
    maximum_same_weekday_source: pd.Timestamp | None = None
    if not safe_dates.empty:
        assert_source_dates_at_or_before_cutoff(
            safe_dates,
            normalized_origin,
            source_name="Stage 3 same-weekday source",
        )
        maximum_same_weekday_source = safe_dates.max()
    lookup = history.set_index(list(KEY_COLUMNS))["sales"]
    same_primary = pd.Series(np.nan, index=base.index, dtype=np.float64)
    if safe_source.any():
        lookup_keys = pd.MultiIndex.from_arrays(
            [base.loc[safe_source, "unique_id"], safe_dates],
            names=KEY_COLUMNS,
        )
        same_primary.loc[safe_source] = lookup.reindex(lookup_keys).to_numpy(
            dtype=np.float64
        )
    base["same_weekday_direct_available"] = same_primary.notna().astype(np.int8)
    base["same_weekday_sales"] = _fill_historical_statistic(
        same_primary,
        last_filled,
        global_median,
    )

    historical = history.groupby("unique_id", observed=True)["sales"].agg(
        historical_mean_sales="mean",
        historical_median_sales="median",
        observed_history_row_count="count",
    )
    for column in ("historical_mean_sales", "historical_median_sales"):
        primary = base["unique_id"].map(historical[column]).astype(np.float64)
        base[column] = _fill_historical_statistic(primary, last_filled, global_median)
    count = base["unique_id"].map(historical["observed_history_row_count"])
    base["historical_stats_available"] = count.notna().astype(np.int8)
    base["observed_history_row_count"] = count.fillna(0).astype(np.int32)

    base["day_of_week"] = base["date"].dt.dayofweek.astype(np.int8)
    base["iso_week"] = base["date"].dt.isocalendar().week.astype(np.int16)
    base["month"] = base["date"].dt.month.astype(np.int8)
    base["weekend_flag"] = base["day_of_week"].isin([5, 6]).astype(np.int8)

    base = _encode_static_categories(base, inventory_frame)
    base["unique_id"] = base["unique_id"].astype(np.int32)
    base["product_unique_id"] = base["product_unique_id"].astype(np.int32)
    base["horizon_day"] = base["horizon_day"].astype(np.int8)

    binary_columns = (
        "weekend_flag",
        "holiday",
        "shops_closed",
        "winter_school_holidays",
        "school_holidays",
        "last_observed_available",
        "trailing_7_available",
        "trailing_14_available",
        "same_weekday_direct_available",
        "historical_stats_available",
    )
    for column in binary_columns:
        base[column] = pd.to_numeric(base[column], errors="raise").astype(np.int8)

    float_columns = (
        "sell_price_main",
        *DISCOUNT_COLUMNS,
        "last_observed_sales",
        "trailing_7_mean",
        "trailing_14_mean",
        "same_weekday_sales",
        "historical_mean_sales",
        "historical_median_sales",
    )
    for column in float_columns:
        base[column] = pd.to_numeric(base[column], errors="raise").astype(np.float32)

    matrix = base.loc[:, APPROVED_FEATURES].copy()
    if tuple(matrix.columns) != APPROVED_FEATURES:
        raise AssertionError("Final feature columns do not match the approved contract.")
    if set(matrix.columns) != set(FEATURE_AVAILABILITY):
        raise AssertionError("Every approved feature must have one availability class.")
    forbidden_final = FORBIDDEN_FEATURE_FIELDS.intersection(matrix.columns)
    if forbidden_final:
        raise AssertionError(f"Final matrix contains forbidden fields: {forbidden_final}.")
    numeric = matrix.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(numeric).all():
        raise ValueError("Final feature matrix contains missing or non-finite values.")

    keys = base.loc[:, KEY_COLUMNS].copy()
    _assert_unique_keys(keys, "final feature keys")
    return FeatureBatch(
        keys=keys,
        matrix=matrix,
        origin=normalized_origin,
        maximum_history_date=history["date"].max(),
        maximum_same_weekday_source_date=maximum_same_weekday_source,
    )
