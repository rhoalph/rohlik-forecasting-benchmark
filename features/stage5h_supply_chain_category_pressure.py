"""Stage 5-H supply-chain category-pressure features.

The feature batch extends the approved Stage 5-E contract with cutoff-safe
warehouse/category pressure and item-share signals. The additional features are
computed only from sales history available on or before the supplied origin.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

from features.stage3_minimal import (
    DISCOUNT_COLUMNS,
    FEATURE_AVAILABILITY as STAGE3_FEATURE_AVAILABILITY,
    FORBIDDEN_FEATURE_FIELDS,
    FeatureBatch,
)
from features.stage5e_stronger_features import (
    APPROVED_STAGE5E_FEATURES,
    FEATURE_AVAILABILITY as STAGE5E_FEATURE_AVAILABILITY,
    build_stage5e_feature_batch,
)


STAGE5H_EXTRA_FEATURES: Final[tuple[str, ...]] = (
    "wh_cat_l2_sales_7d_sum",
    "wh_cat_l2_sales_14d_sum",
    "wh_cat_l2_sales_28d_sum",
    "wh_cat_l2_sales_7d_mean",
    "wh_cat_l2_sales_28d_mean",
    "wh_cat_l2_7d_vs_28d_ratio",
    "wh_cat_l2_7d_minus_28d_mean",
    "wh_cat_l2_reversion_pressure",
    "item_share_of_wh_cat_l2_7d",
    "item_share_of_wh_cat_l2_28d",
    "item_share_7d_minus_28d",
    "item_share_7d_to_28d_ratio",
    "horizon_x_wh_cat_l2_reversion_pressure",
    "discount_x_wh_cat_l2_reversion_pressure",
    "relative_price_x_wh_cat_l2_reversion_pressure",
)

APPROVED_STAGE5H_FEATURES: Final[tuple[str, ...]] = (
    *APPROVED_STAGE5E_FEATURES,
    *STAGE5H_EXTRA_FEATURES,
)

FEATURE_AVAILABILITY: Final[dict[str, str]] = {
    **STAGE3_FEATURE_AVAILABILITY,
    **STAGE5E_FEATURE_AVAILABILITY,
    "wh_cat_l2_sales_7d_sum": "historical only",
    "wh_cat_l2_sales_14d_sum": "historical only",
    "wh_cat_l2_sales_28d_sum": "historical only",
    "wh_cat_l2_sales_7d_mean": "historical only",
    "wh_cat_l2_sales_28d_mean": "historical only",
    "wh_cat_l2_7d_vs_28d_ratio": "historical only",
    "wh_cat_l2_7d_minus_28d_mean": "historical only",
    "wh_cat_l2_reversion_pressure": "historical only",
    "item_share_of_wh_cat_l2_7d": "historical only",
    "item_share_of_wh_cat_l2_28d": "historical only",
    "item_share_7d_minus_28d": "historical only",
    "item_share_7d_to_28d_ratio": "historical only",
    "horizon_x_wh_cat_l2_reversion_pressure": "known future",
    "discount_x_wh_cat_l2_reversion_pressure": "known future",
    "relative_price_x_wh_cat_l2_reversion_pressure": "known future",
}

STAGE5H_FORBIDDEN_FEATURE_FIELDS: Final[frozenset[str]] = FORBIDDEN_FEATURE_FIELDS


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"{name} is missing required columns: {sorted(missing)}.")


def _safe_ratio(numerator: pd.Series, denominator: pd.Series, *, fallback: float) -> pd.Series:
    denominator = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    ratio = pd.to_numeric(numerator, errors="coerce") / denominator
    return ratio.replace([np.inf, -np.inf], np.nan).fillna(fallback).astype(np.float32)


def _history_cutoff(history: pd.DataFrame, origin: object) -> pd.DataFrame:
    frame = history.loc[:, ["unique_id", "date", "sales"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    frame["sales"] = pd.to_numeric(frame["sales"], errors="coerce")
    frame = frame.loc[frame["date"] <= pd.Timestamp(origin).normalize()].copy()
    if frame.empty:
        raise ValueError("training_history must contain data on or before the origin.")
    if frame["sales"].isna().any():
        raise ValueError("training_history contains missing sales.")
    if frame.loc[:, ["unique_id", "date"]].duplicated().any():
        raise ValueError("training_history contains duplicate unique_id/date keys.")
    return frame.sort_values(["unique_id", "date"], kind="stable")


def _request_metadata(request_covariates: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        request_covariates,
        {"unique_id", "date", "warehouse", "horizon_day", "sell_price_main", *DISCOUNT_COLUMNS},
        "request_covariates",
    )
    _require_columns(
        inventory,
        {"unique_id", "product_unique_id", "warehouse", "L2_category_name_en"},
        "inventory",
    )

    meta = request_covariates.loc[
        :,
        ["unique_id", "date", "warehouse", "horizon_day", "sell_price_main", *DISCOUNT_COLUMNS],
    ].copy()
    meta["date"] = pd.to_datetime(meta["date"], errors="raise").dt.normalize()
    meta = meta.merge(
        inventory.loc[:, ["unique_id", "product_unique_id", "warehouse", "L2_category_name_en"]],
        on="unique_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_inventory"),
    )
    if meta[["product_unique_id", "L2_category_name_en"]].isna().any().any():
        raise ValueError("One or more request IDs lack inventory metadata for Stage 5-H features.")
    if not meta["warehouse"].astype(str).eq(meta["warehouse_inventory"].astype(str)).all():
        raise ValueError("Request warehouse does not match inventory warehouse for Stage 5-H features.")
    return meta.drop(columns=["warehouse_inventory"])


def _window_group_stats(
    history_meta: pd.DataFrame,
    origin: pd.Timestamp,
    window_days: int,
) -> pd.DataFrame:
    start = origin - pd.Timedelta(days=window_days - 1)
    window = history_meta.loc[history_meta["date"].between(start, origin)].copy()
    if window.empty:
        return pd.DataFrame(columns=["warehouse", "L2_category_name_en", "sales_sum", "sales_mean"])
    daily = (
        window.groupby(["warehouse", "L2_category_name_en", "date"], observed=True)["sales"]
        .sum()
        .reset_index(name="daily_sales")
    )
    stats = daily.groupby(["warehouse", "L2_category_name_en"], observed=True)["daily_sales"].agg(
        sales_sum="sum",
        sales_mean="mean",
    )
    return stats.reset_index()


def _window_item_share_stats(
    history_meta: pd.DataFrame,
    origin: pd.Timestamp,
    window_days: int,
) -> pd.DataFrame:
    start = origin - pd.Timedelta(days=window_days - 1)
    window = history_meta.loc[history_meta["date"].between(start, origin)].copy()
    if window.empty:
        return pd.DataFrame(
            columns=["warehouse", "L2_category_name_en", "unique_id", "item_sum", "group_sum", "item_share"]
        )
    item_sum = (
        window.groupby(["warehouse", "L2_category_name_en", "unique_id"], observed=True)["sales"]
        .sum()
        .reset_index(name="item_sum")
    )
    group_sum = (
        window.groupby(["warehouse", "L2_category_name_en"], observed=True)["sales"]
        .sum()
        .reset_index(name="group_sum")
    )
    stats = item_sum.merge(
        group_sum,
        on=["warehouse", "L2_category_name_en"],
        how="left",
        validate="many_to_one",
    )
    stats["item_share"] = _safe_ratio(stats["item_sum"], stats["group_sum"], fallback=0.0)
    return stats


def build_stage5h_feature_batch(
    training_history: pd.DataFrame,
    request_covariates: pd.DataFrame,
    inventory: pd.DataFrame,
    calendar: pd.DataFrame,
    origin: object,
) -> FeatureBatch:
    """Build the approved Stage 5-E matrix plus supply-chain category-pressure features."""

    origin_ts = pd.Timestamp(origin).normalize()
    base = build_stage5e_feature_batch(
        training_history,
        request_covariates,
        inventory,
        calendar,
        origin_ts,
    )
    request_meta = _request_metadata(request_covariates, inventory)
    history = _history_cutoff(training_history, origin_ts)
    history_meta = history.merge(
        inventory.loc[:, ["unique_id", "warehouse", "L2_category_name_en"]],
        on="unique_id",
        how="left",
        validate="many_to_one",
    )
    if history_meta[["warehouse", "L2_category_name_en"]].isna().any().any():
        raise ValueError("History rows lack inventory metadata for Stage 5-H features.")

    fallback_mean = base.matrix["historical_mean_sales"].astype(np.float64)
    fallback_sum_7 = (fallback_mean * 7.0).astype(np.float64)
    fallback_sum_14 = (fallback_mean * 14.0).astype(np.float64)
    fallback_sum_28 = (fallback_mean * 28.0).astype(np.float64)

    stats_7 = _window_group_stats(history_meta, origin_ts, 7)
    stats_14 = _window_group_stats(history_meta, origin_ts, 14)
    stats_28 = _window_group_stats(history_meta, origin_ts, 28)
    shares_7 = _window_item_share_stats(history_meta, origin_ts, 7)
    shares_28 = _window_item_share_stats(history_meta, origin_ts, 28)

    group_key = ["warehouse", "L2_category_name_en"]
    item_key = ["warehouse", "L2_category_name_en", "unique_id"]
    request_group = request_meta.loc[:, ["warehouse", "L2_category_name_en", "unique_id"]].copy()

    group_7 = request_group.merge(stats_7, on=group_key, how="left", validate="many_to_one")
    group_14 = request_group.merge(stats_14, on=group_key, how="left", validate="many_to_one")
    group_28 = request_group.merge(stats_28, on=group_key, how="left", validate="many_to_one")
    share_7 = request_group.merge(shares_7, on=item_key, how="left", validate="many_to_one")
    share_28 = request_group.merge(shares_28, on=item_key, how="left", validate="many_to_one")

    extra = pd.DataFrame(index=base.matrix.index)
    extra["wh_cat_l2_sales_7d_sum"] = group_7["sales_sum"].astype(np.float64).fillna(fallback_sum_7).astype(np.float32)
    extra["wh_cat_l2_sales_14d_sum"] = group_14["sales_sum"].astype(np.float64).fillna(fallback_sum_14).astype(np.float32)
    extra["wh_cat_l2_sales_28d_sum"] = group_28["sales_sum"].astype(np.float64).fillna(fallback_sum_28).astype(np.float32)
    extra["wh_cat_l2_sales_7d_mean"] = group_7["sales_mean"].astype(np.float64).fillna(fallback_mean).astype(np.float32)
    extra["wh_cat_l2_sales_28d_mean"] = group_28["sales_mean"].astype(np.float64).fillna(fallback_mean).astype(np.float32)
    extra["wh_cat_l2_7d_vs_28d_ratio"] = _safe_ratio(
        extra["wh_cat_l2_sales_7d_mean"],
        extra["wh_cat_l2_sales_28d_mean"],
        fallback=1.0,
    )
    extra["wh_cat_l2_7d_minus_28d_mean"] = (
        extra["wh_cat_l2_sales_7d_mean"].astype(np.float64) - extra["wh_cat_l2_sales_28d_mean"].astype(np.float64)
    ).astype(np.float32)
    extra["wh_cat_l2_reversion_pressure"] = (
        extra["wh_cat_l2_sales_7d_mean"].astype(np.float64) / extra["wh_cat_l2_sales_28d_mean"].replace(0, np.nan).astype(np.float64)
        - 1.0
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)

    extra["item_share_of_wh_cat_l2_7d"] = share_7["item_share"].astype(np.float64).fillna(0.0).astype(np.float32)
    extra["item_share_of_wh_cat_l2_28d"] = share_28["item_share"].astype(np.float64).fillna(0.0).astype(np.float32)
    extra["item_share_7d_minus_28d"] = (
        extra["item_share_of_wh_cat_l2_7d"].astype(np.float64)
        - extra["item_share_of_wh_cat_l2_28d"].astype(np.float64)
    ).astype(np.float32)
    extra["item_share_7d_to_28d_ratio"] = _safe_ratio(
        extra["item_share_of_wh_cat_l2_7d"],
        extra["item_share_of_wh_cat_l2_28d"],
        fallback=1.0,
    )

    extra["horizon_x_wh_cat_l2_reversion_pressure"] = (
        base.matrix["horizon_day"].astype(np.float64) * extra["wh_cat_l2_reversion_pressure"].astype(np.float64)
    ).astype(np.float32)
    extra["discount_x_wh_cat_l2_reversion_pressure"] = (
        base.matrix["any_discount_active"].astype(np.float64) * extra["wh_cat_l2_reversion_pressure"].astype(np.float64)
    ).astype(np.float32)
    extra["relative_price_x_wh_cat_l2_reversion_pressure"] = (
        base.matrix["price_relative_to_28d_item_median"].astype(np.float64)
        * extra["wh_cat_l2_reversion_pressure"].astype(np.float64)
    ).astype(np.float32)

    matrix = pd.concat([base.matrix.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)
    if tuple(matrix.columns) != APPROVED_STAGE5H_FEATURES:
        raise AssertionError("Stage 5-H feature columns do not match the approved contract.")
    if tuple(matrix.columns[: len(APPROVED_STAGE5E_FEATURES)]) != APPROVED_STAGE5E_FEATURES:
        raise AssertionError("Stage 5-E base feature contract changed.")
    if STAGE5H_FORBIDDEN_FEATURE_FIELDS.intersection(matrix.columns):
        raise AssertionError("Stage 5-H matrix contains forbidden fields.")
    if not np.isfinite(matrix.to_numpy(dtype=np.float64, copy=False)).all():
        raise ValueError("Stage 5-H feature matrix contains missing or non-finite values.")
    return FeatureBatch(
        keys=base.keys.copy(),
        matrix=matrix,
        origin=base.origin,
        maximum_history_date=base.maximum_history_date,
        maximum_same_weekday_source_date=base.maximum_same_weekday_source_date,
    )

