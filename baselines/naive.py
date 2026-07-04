"""Leakage-safe naive forecasts evaluated from a fixed forecast origin."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from dataguard.cutoff import (
    assert_history_at_or_before_cutoff,
    assert_source_dates_at_or_before_cutoff,
    assert_target_not_present,
    assert_validation_within_window,
)


KEY_COLUMNS = ("unique_id", "date")


@dataclass
class BaselineContext:
    """Shared cutoff-safe state for a collection of naive baselines."""

    training: pd.DataFrame
    validation_keys: pd.DataFrame
    cutoff: pd.Timestamp
    horizon_days: int
    global_median: float
    last_sales_by_id: pd.Series


@dataclass(frozen=True)
class BaselineOutput:
    """Predictions plus transparent fallback counts."""

    predictions: pd.DataFrame
    primary_fallback_rows: int
    global_fallback_rows: int


BaselineFunction = Callable[[BaselineContext], BaselineOutput]


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"{name} is missing required columns: {sorted(missing)}.")


def prepare_context(
    training: pd.DataFrame,
    validation_keys: pd.DataFrame,
    cutoff: object,
    *,
    horizon_days: int = 14,
) -> BaselineContext:
    """Validate inputs and compute shared fallbacks from pre-cutoff sales only."""

    _require_columns(training, {"unique_id", "date", "sales"}, "training")
    _require_columns(validation_keys, set(KEY_COLUMNS), "validation_keys")

    history = training.loc[:, ["unique_id", "date", "sales"]].copy()
    history["date"] = pd.to_datetime(history["date"], errors="raise").dt.normalize()
    if history["sales"].isna().any():
        raise ValueError("training contains missing sales; exclude missing targets first.")
    if history.loc[:, KEY_COLUMNS].duplicated().any():
        raise ValueError("training contains duplicate unique_id/date keys.")

    keys = validation_keys.loc[:, KEY_COLUMNS].copy()
    keys["date"] = pd.to_datetime(keys["date"], errors="raise").dt.normalize()
    if keys.loc[:, KEY_COLUMNS].duplicated().any():
        raise ValueError("validation_keys contains duplicate unique_id/date keys.")

    normalized_cutoff = pd.Timestamp(cutoff).normalize()
    assert_history_at_or_before_cutoff(history, normalized_cutoff)
    assert_validation_within_window(
        keys,
        normalized_cutoff,
        horizon_days=horizon_days,
    )
    assert_target_not_present(keys)

    global_median = float(history["sales"].median())
    if not np.isfinite(global_median):
        raise ValueError("Global median is not finite.")

    last_rows = (
        history.sort_values(["unique_id", "date"], kind="stable")
        .drop_duplicates("unique_id", keep="last")
        .set_index("unique_id")["sales"]
        .astype(np.float64)
    )
    return BaselineContext(
        training=history,
        validation_keys=keys,
        cutoff=normalized_cutoff,
        horizon_days=horizon_days,
        global_median=global_median,
        last_sales_by_id=last_rows,
    )


def _prediction_frame(context: BaselineContext, values: pd.Series | np.ndarray) -> pd.DataFrame:
    prediction = np.asarray(values, dtype=np.float64)
    if prediction.ndim != 1 or len(prediction) != len(context.validation_keys):
        raise ValueError("Prediction vector must have one value per validation key.")
    if not np.isfinite(prediction).all():
        raise ValueError("Prediction vector contains missing or non-finite values.")
    frame = context.validation_keys.copy()
    frame["sales_hat"] = prediction
    assert_target_not_present(frame)
    return frame


def _apply_last_global_fallback(
    context: BaselineContext,
    primary: pd.Series,
) -> tuple[pd.Series, int, int]:
    values = pd.Series(primary, index=context.validation_keys.index, dtype=np.float64)
    primary_missing = int(values.isna().sum())
    last = context.validation_keys["unique_id"].map(context.last_sales_by_id).astype(np.float64)
    values = values.fillna(last)
    global_missing = int(values.isna().sum())
    values = values.fillna(context.global_median)
    return values, primary_missing, global_missing


def zero_forecast(context: BaselineContext) -> BaselineOutput:
    values = np.zeros(len(context.validation_keys), dtype=np.float64)
    return BaselineOutput(_prediction_frame(context, values), 0, 0)


def global_median_forecast(context: BaselineContext) -> BaselineOutput:
    values = np.full(len(context.validation_keys), context.global_median, dtype=np.float64)
    return BaselineOutput(_prediction_frame(context, values), 0, 0)


def last_observed_forecast(context: BaselineContext) -> BaselineOutput:
    primary = context.validation_keys["unique_id"].map(context.last_sales_by_id)
    values, missing, global_missing = _apply_last_global_fallback(context, primary)
    return BaselineOutput(
        _prediction_frame(context, values),
        primary_fallback_rows=missing,
        global_fallback_rows=global_missing,
    )


def same_weekday_last_week_forecast(context: BaselineContext) -> BaselineOutput:
    source_dates = context.validation_keys["date"] - pd.Timedelta(days=7)
    safe_source = source_dates <= context.cutoff
    assert_source_dates_at_or_before_cutoff(
        source_dates.loc[safe_source],
        context.cutoff,
        source_name="same-weekday source",
    )

    lookup = context.training.set_index(list(KEY_COLUMNS))["sales"]
    primary = pd.Series(np.nan, index=context.validation_keys.index, dtype=np.float64)
    if safe_source.any():
        safe_keys = pd.MultiIndex.from_arrays(
            [
                context.validation_keys.loc[safe_source, "unique_id"],
                source_dates.loc[safe_source],
            ],
            names=KEY_COLUMNS,
        )
        primary.loc[safe_source] = lookup.reindex(safe_keys).to_numpy(dtype=np.float64)

    values, missing, global_missing = _apply_last_global_fallback(context, primary)
    return BaselineOutput(
        _prediction_frame(context, values),
        primary_fallback_rows=missing,
        global_fallback_rows=global_missing,
    )


def trailing_mean_forecast(context: BaselineContext, window_days: int) -> BaselineOutput:
    if window_days <= 0:
        raise ValueError("window_days must be positive.")
    start = context.cutoff - pd.Timedelta(days=window_days - 1)
    recent = context.training.loc[
        context.training["date"].between(start, context.cutoff, inclusive="both")
    ]
    means = recent.groupby("unique_id", observed=True)["sales"].mean()
    primary = context.validation_keys["unique_id"].map(means)
    values, missing, global_missing = _apply_last_global_fallback(context, primary)
    return BaselineOutput(
        _prediction_frame(context, values),
        primary_fallback_rows=missing,
        global_fallback_rows=global_missing,
    )


def trailing_7_day_mean_forecast(context: BaselineContext) -> BaselineOutput:
    return trailing_mean_forecast(context, 7)


def trailing_14_day_mean_forecast(context: BaselineContext) -> BaselineOutput:
    return trailing_mean_forecast(context, 14)


def item_weekday_median_forecast(context: BaselineContext) -> BaselineOutput:
    history = context.training.assign(day_of_week=context.training["date"].dt.dayofweek)
    medians = history.groupby(["unique_id", "day_of_week"], observed=True)["sales"].median()
    validation_weekday = context.validation_keys["date"].dt.dayofweek
    lookup_keys = pd.MultiIndex.from_arrays(
        [context.validation_keys["unique_id"], validation_weekday],
        names=["unique_id", "day_of_week"],
    )
    primary = pd.Series(
        medians.reindex(lookup_keys).to_numpy(dtype=np.float64),
        index=context.validation_keys.index,
    )
    values, missing, global_missing = _apply_last_global_fallback(context, primary)
    return BaselineOutput(
        _prediction_frame(context, values),
        primary_fallback_rows=missing,
        global_fallback_rows=global_missing,
    )


BASELINES: tuple[tuple[str, BaselineFunction], ...] = (
    ("zero_forecast", zero_forecast),
    ("global_median", global_median_forecast),
    ("last_observed_by_id", last_observed_forecast),
    ("same_weekday_last_week", same_weekday_last_week_forecast),
    ("trailing_7_day_mean", trailing_7_day_mean_forecast),
    ("trailing_14_day_mean", trailing_14_day_mean_forecast),
    ("median_by_id_day_of_week", item_weekday_median_forecast),
)
