"""Stage 5 price and discount feature experiment."""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

from dataguard.cutoff import assert_history_at_or_before_cutoff, assert_target_not_present
from features.stage3_minimal import (
    APPROVED_FEATURES,
    CATEGORICAL_FEATURES,
    DISCOUNT_COLUMNS,
    FEATURE_AVAILABILITY as STAGE3_FEATURE_AVAILABILITY,
    FORBIDDEN_FEATURE_FIELDS,
    FeatureBatch,
    build_stage3_feature_batch,
)


PRICE_DISCOUNT_FEATURES: Final[tuple[str, ...]] = (
    "price_relative_to_item_history_median",
    "price_relative_to_item_history_mean",
    "price_change_vs_last_observed",
    "price_rank_within_warehouse_date",
    "price_zscore_vs_item_history",
    "any_discount_active",
    "max_discount",
    "total_discount",
    "active_discount_count",
    "discount_depth_relative_to_item_history",
)

APPROVED_STAGE5_FEATURES: Final[tuple[str, ...]] = (
    *APPROVED_FEATURES,
    *PRICE_DISCOUNT_FEATURES,
)

FEATURE_AVAILABILITY: Final[dict[str, str]] = {
    **STAGE3_FEATURE_AVAILABILITY,
    "price_relative_to_item_history_median": "historical only",
    "price_relative_to_item_history_mean": "historical only",
    "price_change_vs_last_observed": "historical only",
    "price_rank_within_warehouse_date": "known future",
    "price_zscore_vs_item_history": "historical only",
    "any_discount_active": "known future",
    "max_discount": "known future",
    "total_discount": "known future",
    "active_discount_count": "known future",
    "discount_depth_relative_to_item_history": "historical only",
}

STAGE5_FORBIDDEN_FEATURE_FIELDS: Final[frozenset[str]] = FORBIDDEN_FEATURE_FIELDS


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"{name} is missing required columns: {sorted(missing)}.")


def _safe_ratio(numerator: pd.Series, denominator: pd.Series, *, fallback: float = 1.0) -> pd.Series:
    denominator = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    ratio = pd.to_numeric(numerator, errors="coerce") / denominator
    return ratio.replace([np.inf, -np.inf], np.nan).fillna(fallback).astype(np.float32)


def _safe_zscore(
    value: pd.Series,
    mean: pd.Series,
    std: pd.Series,
    *,
    fallback: float = 0.0,
) -> pd.Series:
    safe_std = pd.to_numeric(std, errors="coerce").replace(0, np.nan)
    zscore = (pd.to_numeric(value, errors="coerce") - pd.to_numeric(mean, errors="coerce")) / safe_std
    return zscore.replace([np.inf, -np.inf], np.nan).fillna(fallback).astype(np.float32)


def _historical_price_discount_stats(history: pd.DataFrame, origin: object) -> pd.DataFrame:
    required = {"unique_id", "date", "sell_price_main", *DISCOUNT_COLUMNS}
    _require_columns(history, required, "training_history")
    frame = history.loc[:, ["unique_id", "date", "sell_price_main", *DISCOUNT_COLUMNS]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    frame["sell_price_main"] = pd.to_numeric(frame["sell_price_main"], errors="coerce")
    if frame["sell_price_main"].isna().any():
        raise ValueError("training_history contains missing sell_price_main values.")
    for column in DISCOUNT_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[column].isna().any():
            raise ValueError(f"training_history contains missing {column} values.")
    frame["discount_total"] = frame.loc[:, DISCOUNT_COLUMNS].sum(axis=1)
    frame = frame.sort_values(["unique_id", "date"], kind="stable")
    assert_history_at_or_before_cutoff(frame.loc[:, ["unique_id", "date"]], origin)
    return frame


def _price_discount_additions(
    training_history: pd.DataFrame,
    request_covariates: pd.DataFrame,
    origin: object,
) -> pd.DataFrame:
    required_request = {
        "unique_id",
        "date",
        "warehouse",
        "sell_price_main",
        *DISCOUNT_COLUMNS,
    }
    _require_columns(request_covariates, required_request, "request_covariates")
    assert_target_not_present(request_covariates)

    requests = request_covariates.loc[:, ["unique_id", "date", "warehouse", "sell_price_main", *DISCOUNT_COLUMNS]].copy()
    requests["date"] = pd.to_datetime(requests["date"], errors="raise").dt.normalize()
    requests["sell_price_main"] = pd.to_numeric(requests["sell_price_main"], errors="coerce")
    if requests["sell_price_main"].isna().any():
        raise ValueError("request_covariates contains missing sell_price_main values.")
    for column in DISCOUNT_COLUMNS:
        requests[column] = pd.to_numeric(requests[column], errors="coerce")
        if requests[column].isna().any():
            raise ValueError(f"request_covariates contains missing {column} values.")

    requests["total_discount"] = requests.loc[:, DISCOUNT_COLUMNS].sum(axis=1)
    requests["any_discount_active"] = (requests.loc[:, DISCOUNT_COLUMNS] > 0).any(axis=1).astype(np.int8)
    requests["max_discount"] = requests.loc[:, DISCOUNT_COLUMNS].max(axis=1)
    requests["active_discount_count"] = (requests.loc[:, DISCOUNT_COLUMNS] > 0).sum(axis=1).astype(np.int8)
    requests["price_rank_within_warehouse_date"] = (
        requests.groupby(["warehouse", "date"], observed=True)["sell_price_main"].rank(
            method="average",
            pct=True,
        )
    ).astype(np.float32)

    history = _historical_price_discount_stats(training_history, origin=origin)
    history_stats = history.groupby("unique_id", observed=True).agg(
        price_mean=("sell_price_main", "mean"),
        price_median=("sell_price_main", "median"),
        price_std=("sell_price_main", "std"),
        price_last=("sell_price_main", "last"),
        discount_mean=("discount_total", "mean"),
        discount_median=("discount_total", "median"),
        discount_last=("discount_total", "last"),
    )
    global_price_mean = float(history["sell_price_main"].mean())
    global_price_median = float(history["sell_price_main"].median())
    global_price_std = float(history["sell_price_main"].std(ddof=0))
    global_discount_mean = float(history["discount_total"].mean())
    global_discount_median = float(history["discount_total"].median())
    global_discount_last = float(history.sort_values("date", kind="stable")["discount_total"].iloc[-1])

    index = requests["unique_id"]
    price_mean = index.map(history_stats["price_mean"]).astype(np.float64)
    price_median = index.map(history_stats["price_median"]).astype(np.float64)
    price_std = index.map(history_stats["price_std"]).astype(np.float64)
    price_last = index.map(history_stats["price_last"]).astype(np.float64)
    discount_mean = index.map(history_stats["discount_mean"]).astype(np.float64)
    discount_median = index.map(history_stats["discount_median"]).astype(np.float64)
    discount_last = index.map(history_stats["discount_last"]).astype(np.float64)

    price_mean = price_mean.fillna(global_price_mean)
    price_median = price_median.fillna(global_price_median)
    price_std = price_std.fillna(global_price_std if np.isfinite(global_price_std) and global_price_std > 0 else 1.0)
    price_last = price_last.fillna(global_price_median)
    discount_mean = discount_mean.fillna(global_discount_mean)
    discount_median = discount_median.fillna(global_discount_median)
    discount_last = discount_last.fillna(global_discount_last)

    frame = pd.DataFrame(index=requests.index)
    frame["price_relative_to_item_history_median"] = _safe_ratio(
        requests["sell_price_main"], price_median
    )
    frame["price_relative_to_item_history_mean"] = _safe_ratio(
        requests["sell_price_main"], price_mean
    )
    frame["price_change_vs_last_observed"] = (
        requests["sell_price_main"] - price_last
    ).astype(np.float32)
    frame["price_rank_within_warehouse_date"] = requests["price_rank_within_warehouse_date"].astype(np.float32)
    frame["price_zscore_vs_item_history"] = _safe_zscore(
        requests["sell_price_main"], price_mean, price_std
    )
    frame["any_discount_active"] = requests["any_discount_active"].astype(np.int8)
    frame["max_discount"] = requests["max_discount"].astype(np.float32)
    frame["total_discount"] = requests["total_discount"].astype(np.float32)
    frame["active_discount_count"] = requests["active_discount_count"].astype(np.int8)
    frame["discount_depth_relative_to_item_history"] = _safe_ratio(
        requests["total_discount"], discount_mean
    )
    # A tiny extra stability check: historical totals must exist for the reference calculations.
    if not np.isfinite(frame.to_numpy(dtype=np.float64, copy=False)).all():
        raise ValueError("Stage 5 price/discount additions contain non-finite values.")
    return frame


def build_stage5_feature_batch(
    training_history: pd.DataFrame,
    request_covariates: pd.DataFrame,
    inventory: pd.DataFrame,
    calendar: pd.DataFrame,
    origin: object,
) -> FeatureBatch:
    """Build the approved Stage 3 matrix plus controlled price/discount additions."""

    base = build_stage3_feature_batch(
        training_history,
        request_covariates,
        inventory,
        calendar,
        origin,
    )
    additions = _price_discount_additions(training_history, request_covariates, origin)
    matrix = pd.concat([base.matrix.reset_index(drop=True), additions.reset_index(drop=True)], axis=1)
    if tuple(matrix.columns) != APPROVED_STAGE5_FEATURES:
        raise AssertionError("Stage 5 feature columns do not match the approved contract.")
    if STAGE5_FORBIDDEN_FEATURE_FIELDS.intersection(matrix.columns):
        raise AssertionError("Stage 5 matrix contains forbidden fields.")
    if not np.isfinite(matrix.to_numpy(dtype=np.float64, copy=False)).all():
        raise ValueError("Stage 5 feature matrix contains missing or non-finite values.")
    return FeatureBatch(
        keys=base.keys.copy(),
        matrix=matrix,
        origin=base.origin,
        maximum_history_date=base.maximum_history_date,
        maximum_same_weekday_source_date=base.maximum_same_weekday_source_date,
    )
