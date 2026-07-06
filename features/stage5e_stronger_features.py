"""Stronger Stage 5-E cutoff-safe feature batch."""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

from features.stage3_minimal import (
    APPROVED_FEATURES,
    DISCOUNT_COLUMNS,
    FEATURE_AVAILABILITY as STAGE3_FEATURE_AVAILABILITY,
    FORBIDDEN_FEATURE_FIELDS,
    FeatureBatch,
    STATIC_CATEGORY_COLUMNS,
)
from features.stage5_price_discount import (
    APPROVED_STAGE5_FEATURES,
    FEATURE_AVAILABILITY as STAGE5_FEATURE_AVAILABILITY,
    build_stage5_feature_batch,
)


STAGE5E_EXTRA_FEATURES: Final[tuple[str, ...]] = (
    "lag_7_sales",
    "lag_14_sales",
    "lag_21_sales",
    "lag_28_sales",
    "same_weekday_2w_sales",
    "same_weekday_3w_sales",
    "same_weekday_4w_sales",
    "rolling_7_median_sales",
    "rolling_14_median_sales",
    "rolling_28_mean_sales",
    "rolling_28_median_sales",
    "rolling_7_to_28_mean_ratio",
    "recent_trend_7_vs_28",
    "price_relative_to_28d_item_median",
    "price_relative_to_28d_item_mean",
    "price_change_vs_last_observed_price",
    "discount_active_count",
    "discount_change_vs_last_observed_total_discount",
    "price_x_any_discount",
    "price_x_max_discount",
    "warehouse_category_l2_mean_sales",
    "warehouse_category_l2_median_sales",
    "product_mean_sales_across_warehouses",
    "product_median_sales_across_warehouses",
    "category_l2_mean_sales",
    "item_share_of_warehouse_category_l2_sales",
    "horizon_x_any_discount",
    "horizon_x_max_discount",
    "horizon_x_recent_trend_7_vs_28",
    "dayofweek_x_any_discount",
)

APPROVED_STAGE5E_FEATURES: Final[tuple[str, ...]] = (
    *APPROVED_STAGE5_FEATURES,
    *STAGE5E_EXTRA_FEATURES,
)

FEATURE_AVAILABILITY: Final[dict[str, str]] = {
    **STAGE5_FEATURE_AVAILABILITY,
    "lag_7_sales": "historical only",
    "lag_14_sales": "historical only",
    "lag_21_sales": "historical only",
    "lag_28_sales": "historical only",
    "same_weekday_2w_sales": "historical only",
    "same_weekday_3w_sales": "historical only",
    "same_weekday_4w_sales": "historical only",
    "rolling_7_median_sales": "historical only",
    "rolling_14_median_sales": "historical only",
    "rolling_28_mean_sales": "historical only",
    "rolling_28_median_sales": "historical only",
    "rolling_7_to_28_mean_ratio": "historical only",
    "recent_trend_7_vs_28": "historical only",
    "price_relative_to_28d_item_median": "historical only",
    "price_relative_to_28d_item_mean": "historical only",
    "price_change_vs_last_observed_price": "historical only",
    "discount_active_count": "known future",
    "discount_change_vs_last_observed_total_discount": "historical only",
    "price_x_any_discount": "known future",
    "price_x_max_discount": "known future",
    "warehouse_category_l2_mean_sales": "historical only",
    "warehouse_category_l2_median_sales": "historical only",
    "product_mean_sales_across_warehouses": "historical only",
    "product_median_sales_across_warehouses": "historical only",
    "category_l2_mean_sales": "historical only",
    "item_share_of_warehouse_category_l2_sales": "historical only",
    "horizon_x_any_discount": "known future",
    "horizon_x_max_discount": "known future",
    "horizon_x_recent_trend_7_vs_28": "known future",
    "dayofweek_x_any_discount": "known future",
}

STAGE5E_FORBIDDEN_FEATURE_FIELDS: Final[frozenset[str]] = FORBIDDEN_FEATURE_FIELDS


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"{name} is missing required columns: {sorted(missing)}.")


def _safe_ratio(numerator: pd.Series, denominator: pd.Series, *, fallback: float = 1.0) -> pd.Series:
    denominator = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    ratio = pd.to_numeric(numerator, errors="coerce") / denominator
    return ratio.replace([np.inf, -np.inf], np.nan).fillna(fallback).astype(np.float32)


def _history_cutoff(history: pd.DataFrame, origin: object) -> pd.DataFrame:
    frame = history.loc[
        :,
        [
            "unique_id",
            "date",
            "sales",
            "sell_price_main",
            *DISCOUNT_COLUMNS,
        ],
    ].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    frame["sales"] = pd.to_numeric(frame["sales"], errors="coerce")
    frame["sell_price_main"] = pd.to_numeric(frame["sell_price_main"], errors="coerce")
    for column in DISCOUNT_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[frame["date"] <= pd.Timestamp(origin).normalize()].copy()
    if frame.empty:
        raise ValueError("training_history must contain data on or before the origin.")
    if frame["sales"].isna().any():
        raise ValueError("training_history contains missing sales.")
    if frame["sell_price_main"].isna().any():
        raise ValueError("training_history contains missing sell_price_main values.")
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
        raise ValueError("One or more request IDs lack inventory metadata for Stage 5-E features.")
    if not meta["warehouse"].astype(str).eq(meta["warehouse_inventory"].astype(str)).all():
        raise ValueError("Request warehouse does not match inventory warehouse for Stage 5-E features.")
    return meta.drop(columns=["warehouse_inventory"])


def _map_lookup(
    lookup: pd.Series,
    keys: pd.MultiIndex,
    *,
    fallback: pd.Series,
) -> pd.Series:
    values = lookup.reindex(keys).to_numpy(dtype=np.float64)
    series = pd.Series(values, index=fallback.index, dtype=np.float64)
    return series.fillna(fallback)


def _lag_feature(
    history_lookup: pd.Series,
    request_meta: pd.DataFrame,
    origin: pd.Timestamp,
    days: int,
    *,
    fallback: pd.Series,
    same_weekday: bool = False,
) -> pd.Series:
    source_dates = pd.to_datetime(request_meta["date"], errors="raise").dt.normalize() - pd.Timedelta(days=days)
    safe_source = source_dates <= origin
    result = pd.Series(np.nan, index=request_meta.index, dtype=np.float64)
    if safe_source.any():
        lookup_keys = pd.MultiIndex.from_arrays(
            [request_meta.loc[safe_source, "unique_id"], source_dates.loc[safe_source]],
            names=["unique_id", "date"],
        )
        result.loc[safe_source] = history_lookup.reindex(lookup_keys).to_numpy(dtype=np.float64)
    if same_weekday:
        return result.fillna(fallback)
    return result.fillna(fallback)


def _window_stats(
    history: pd.DataFrame,
    origin: pd.Timestamp,
    window_days: int,
) -> pd.DataFrame:
    start = origin - pd.Timedelta(days=window_days - 1)
    window = history.loc[history["date"].between(start, origin)]
    if window.empty:
        return pd.DataFrame(index=pd.Index([], name="unique_id"))
    stats = window.groupby("unique_id", observed=True)["sales"].agg(["mean", "median"])
    stats.columns = [f"{window_days}_{column}" for column in stats.columns]
    return stats


def _price_discount_window_stats(history: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    window = history.loc[history["date"].between(origin - pd.Timedelta(days=27), origin)].copy()
    if window.empty:
        return pd.DataFrame(index=pd.Index([], name="unique_id"))
    window["discount_total"] = window.loc[:, DISCOUNT_COLUMNS].sum(axis=1)
    stats = window.groupby("unique_id", observed=True).agg(
        price_28_mean=("sell_price_main", "mean"),
        price_28_median=("sell_price_main", "median"),
        price_last=("sell_price_main", "last"),
        discount_last=("discount_total", "last"),
    )
    return stats


def _group_stats(history: pd.DataFrame, inventory: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged = history.merge(
        inventory.loc[:, ["unique_id", "product_unique_id", "warehouse", "L2_category_name_en"]],
        on="unique_id",
        how="left",
        validate="many_to_one",
    )
    if merged[["product_unique_id", "warehouse", "L2_category_name_en"]].isna().any().any():
        raise ValueError("History rows lack inventory metadata for Stage 5-E group features.")
    wh_l2 = merged.groupby(["warehouse", "L2_category_name_en"], observed=True)["sales"].agg(["mean", "median"]).reset_index()
    product = merged.groupby("product_unique_id", observed=True)["sales"].agg(["mean", "median"]).reset_index()
    category = merged.groupby("L2_category_name_en", observed=True)["sales"].mean().reset_index(name="mean")
    item_group = (
        merged.groupby(["warehouse", "L2_category_name_en", "unique_id"], observed=True)["sales"].sum().reset_index(name="item_sum")
    )
    group_sum = (
        merged.groupby(["warehouse", "L2_category_name_en"], observed=True)["sales"].sum().reset_index(name="group_sum")
    )
    item_group = item_group.merge(group_sum, on=["warehouse", "L2_category_name_en"], how="left", validate="many_to_one")
    return wh_l2, product, category, item_group


def build_stage5e_feature_batch(
    training_history: pd.DataFrame,
    request_covariates: pd.DataFrame,
    inventory: pd.DataFrame,
    calendar: pd.DataFrame,
    origin: object,
) -> FeatureBatch:
    """Build the approved Stage 5-B matrix plus stronger cutoff-safe features."""

    origin_ts = pd.Timestamp(origin).normalize()
    base = build_stage5_feature_batch(
        training_history,
        request_covariates,
        inventory,
        calendar,
        origin_ts,
    )
    request_meta = _request_metadata(request_covariates, inventory)
    history = _history_cutoff(training_history, origin_ts)
    history_lookup_sales = history.set_index(["unique_id", "date"])["sales"]
    history["discount_total"] = history.loc[:, DISCOUNT_COLUMNS].sum(axis=1)
    history_sorted = history.sort_values("date", kind="stable")
    fallback_last_sales = base.matrix["last_observed_sales"].astype(np.float64)
    fallback_same_weekday = base.matrix["same_weekday_sales"].astype(np.float64)
    fallback_hist_mean = base.matrix["historical_mean_sales"].astype(np.float64)
    fallback_hist_median = base.matrix["historical_median_sales"].astype(np.float64)

    extra = pd.DataFrame(index=base.matrix.index)

    for days in (7, 14, 21, 28):
        fallback = fallback_last_sales if days == 7 else fallback_same_weekday
        extra[f"lag_{days}_sales"] = _lag_feature(
            history_lookup_sales,
            request_meta,
            origin_ts,
            days,
            fallback=fallback,
        ).astype(np.float32)

    for days, column in ((14, "same_weekday_2w_sales"), (21, "same_weekday_3w_sales"), (28, "same_weekday_4w_sales")):
        extra[column] = _lag_feature(
            history_lookup_sales,
            request_meta,
            origin_ts,
            days,
            fallback=fallback_same_weekday,
            same_weekday=True,
        ).astype(np.float32)

    rolling_7 = _window_stats(history, origin_ts, 7)
    rolling_14 = _window_stats(history, origin_ts, 14)
    rolling_28 = _window_stats(history, origin_ts, 28)
    rolling_7_mean = request_meta["unique_id"].map(rolling_7["7_mean"] if "7_mean" in rolling_7.columns else pd.Series(dtype=float)).astype(np.float64)
    rolling_14_mean = request_meta["unique_id"].map(rolling_14["14_mean"] if "14_mean" in rolling_14.columns else pd.Series(dtype=float)).astype(np.float64)
    rolling_28_mean = request_meta["unique_id"].map(rolling_28["28_mean"] if "28_mean" in rolling_28.columns else pd.Series(dtype=float)).astype(np.float64)
    rolling_7_median = request_meta["unique_id"].map(rolling_7["7_median"] if "7_median" in rolling_7.columns else pd.Series(dtype=float)).astype(np.float64)
    rolling_14_median = request_meta["unique_id"].map(rolling_14["14_median"] if "14_median" in rolling_14.columns else pd.Series(dtype=float)).astype(np.float64)
    rolling_28_median = request_meta["unique_id"].map(rolling_28["28_median"] if "28_median" in rolling_28.columns else pd.Series(dtype=float)).astype(np.float64)

    extra["rolling_7_median_sales"] = rolling_7_median.fillna(fallback_hist_median).astype(np.float32)
    extra["rolling_14_median_sales"] = rolling_14_median.fillna(fallback_hist_median).astype(np.float32)
    extra["rolling_28_mean_sales"] = rolling_28_mean.fillna(fallback_hist_mean).astype(np.float32)
    extra["rolling_28_median_sales"] = rolling_28_median.fillna(fallback_hist_median).astype(np.float32)
    extra["rolling_7_to_28_mean_ratio"] = _safe_ratio(
        base.matrix["trailing_7_mean"],
        extra["rolling_28_mean_sales"],
        fallback=1.0,
    )
    extra["recent_trend_7_vs_28"] = (
        base.matrix["trailing_7_mean"].astype(np.float64) - extra["rolling_28_mean_sales"].astype(np.float64)
    ).astype(np.float32)

    price_stats = _price_discount_window_stats(history, origin_ts)
    price_stats_index = price_stats.index
    price_median_28 = request_meta["unique_id"].map(price_stats["price_28_median"] if "price_28_median" in price_stats else pd.Series(dtype=float)).astype(np.float64)
    price_mean_28 = request_meta["unique_id"].map(price_stats["price_28_mean"] if "price_28_mean" in price_stats else pd.Series(dtype=float)).astype(np.float64)
    price_last = request_meta["unique_id"].map(price_stats["price_last"] if "price_last" in price_stats else pd.Series(dtype=float)).astype(np.float64)
    discount_last = request_meta["unique_id"].map(price_stats["discount_last"] if "discount_last" in price_stats else pd.Series(dtype=float)).astype(np.float64)
    global_price_median = float(history_sorted["sell_price_main"].median())
    global_price_mean = float(history_sorted["sell_price_main"].mean())
    global_discount_last = float(history_sorted["discount_total"].iloc[-1])
    extra["price_relative_to_28d_item_median"] = _safe_ratio(
        base.matrix["sell_price_main"],
        price_median_28.fillna(global_price_median),
        fallback=1.0,
    )
    extra["price_relative_to_28d_item_mean"] = _safe_ratio(
        base.matrix["sell_price_main"],
        price_mean_28.fillna(global_price_mean),
        fallback=1.0,
    )
    extra["price_change_vs_last_observed_price"] = (
        base.matrix["sell_price_main"].astype(np.float64) - price_last.fillna(global_price_median)
    ).astype(np.float32)
    extra["discount_active_count"] = base.matrix["active_discount_count"].astype(np.int8)
    extra["discount_change_vs_last_observed_total_discount"] = (
        base.matrix["total_discount"].astype(np.float64) - discount_last.fillna(global_discount_last)
    ).astype(np.float32)
    extra["price_x_any_discount"] = (
        base.matrix["sell_price_main"].astype(np.float64) * base.matrix["any_discount_active"].astype(np.float64)
    ).astype(np.float32)
    extra["price_x_max_discount"] = (
        base.matrix["sell_price_main"].astype(np.float64) * base.matrix["max_discount"].astype(np.float64)
    ).astype(np.float32)

    wh_l2, product, category, item_group = _group_stats(history, inventory)
    request_group = request_meta.loc[:, ["unique_id", "warehouse", "L2_category_name_en", "product_unique_id"]].copy()
    warehouse_category = request_group.merge(
        wh_l2.loc[:, ["warehouse", "L2_category_name_en", "mean", "median"]],
        on=["warehouse", "L2_category_name_en"],
        how="left",
        validate="many_to_one",
    )
    product_stats = request_group.merge(
        product.loc[:, ["product_unique_id", "mean", "median"]],
        on="product_unique_id",
        how="left",
        validate="many_to_one",
    )
    category_stats = request_group.merge(
        category.loc[:, ["L2_category_name_en", "mean"]],
        on="L2_category_name_en",
        how="left",
        validate="many_to_one",
    )
    item_share_stats = request_group.merge(
        item_group.loc[:, ["warehouse", "L2_category_name_en", "unique_id", "item_sum", "group_sum"]],
        on=["warehouse", "L2_category_name_en", "unique_id"],
        how="left",
        validate="many_to_one",
    )
    warehouse_category_mean = warehouse_category["mean"].astype(np.float64)
    warehouse_category_median = warehouse_category["median"].astype(np.float64)
    product_mean_mapped = product_stats["mean"].astype(np.float64)
    product_median_mapped = product_stats["median"].astype(np.float64)
    category_mean_mapped = category_stats["mean"].astype(np.float64)
    item_sum_mapped = item_share_stats["item_sum"].astype(np.float64)
    group_sum_mapped = item_share_stats["group_sum"].astype(np.float64)
    extra["warehouse_category_l2_mean_sales"] = warehouse_category_mean.fillna(fallback_hist_mean).astype(np.float32)
    extra["warehouse_category_l2_median_sales"] = warehouse_category_median.fillna(fallback_hist_median).astype(np.float32)
    extra["product_mean_sales_across_warehouses"] = product_mean_mapped.fillna(fallback_hist_mean).astype(np.float32)
    extra["product_median_sales_across_warehouses"] = product_median_mapped.fillna(fallback_hist_median).astype(np.float32)
    extra["category_l2_mean_sales"] = category_mean_mapped.fillna(fallback_hist_mean).astype(np.float32)
    extra["item_share_of_warehouse_category_l2_sales"] = _safe_ratio(
        item_sum_mapped.fillna(0.0),
        group_sum_mapped.fillna(0.0),
        fallback=0.0,
    )

    extra["horizon_x_any_discount"] = (
        base.matrix["horizon_day"].astype(np.float64) * base.matrix["any_discount_active"].astype(np.float64)
    ).astype(np.float32)
    extra["horizon_x_max_discount"] = (
        base.matrix["horizon_day"].astype(np.float64) * base.matrix["max_discount"].astype(np.float64)
    ).astype(np.float32)
    extra["horizon_x_recent_trend_7_vs_28"] = (
        base.matrix["horizon_day"].astype(np.float64) * extra["recent_trend_7_vs_28"].astype(np.float64)
    ).astype(np.float32)
    extra["dayofweek_x_any_discount"] = (
        base.matrix["day_of_week"].astype(np.float64) * base.matrix["any_discount_active"].astype(np.float64)
    ).astype(np.float32)

    matrix = pd.concat([base.matrix.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)
    if tuple(matrix.columns) != APPROVED_STAGE5E_FEATURES:
        raise AssertionError("Stage 5-E feature columns do not match the approved contract.")
    if STAGE5E_FORBIDDEN_FEATURE_FIELDS.intersection(matrix.columns):
        raise AssertionError("Stage 5-E matrix contains forbidden fields.")
    if not np.isfinite(matrix.to_numpy(dtype=np.float64, copy=False)).all():
        raise ValueError("Stage 5-E feature matrix contains missing or non-finite values.")
    return FeatureBatch(
        keys=base.keys.copy(),
        matrix=matrix,
        origin=base.origin,
        maximum_history_date=base.maximum_history_date,
        maximum_same_weekday_source_date=base.maximum_same_weekday_source_date,
    )
