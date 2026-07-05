#!/usr/bin/env python3
"""Run the approved first batch of Stage 3 F1 diagnostic ablations only."""

from __future__ import annotations

import gc
import json
import resource
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from baselines.naive import (
    item_weekday_median_forecast,
    prepare_context,
    trailing_14_day_mean_forecast,
)
from eval.backtest import make_backtest_folds, materialize_backtest_split
from eval.grid import score_kaggle_aligned
from features.stage3_minimal import (
    APPROVED_FEATURES,
    CATEGORICAL_FEATURES,
    DISCOUNT_COLUMNS,
    FORBIDDEN_FEATURE_FIELDS,
    build_stage3_feature_batch,
)
from models.plain_lgbm import (
    PlainLightGBMConfig,
    predict_plain_lightgbm,
    train_plain_lightgbm,
)
from scripts.run_stage3_f1_plain_model import (
    EXPECTED_F1_COVERAGE,
    EXPECTED_F1_ROWS,
    EXPECTED_TOTAL_TRAINING_ROWS,
    EXPECTED_TRAINING_ROWS,
    F1_BASELINE,
    F1_CUTOFF,
    _aligned_target,
    _load_raw_data,
)


ROOT = Path(__file__).resolve().parents[1]
FULL_STAGE3_F1 = {
    "wmae": 20.58275093511847,
    "wape": 0.21869379320058285,
    "bias": -0.015387375961508831,
}
STAGE2_ID_WEEKDAY_REFERENCE = {
    "wmae": 30.715020322087646,
    "wape": 0.34110600973260435,
    "bias": -0.19291461525334522,
}
HISTORICAL_FEATURES = (
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
STATIC_FEATURES = (
    "unique_id",
    "product_unique_id",
    "warehouse",
    "L1_category_name_en",
    "L2_category_name_en",
    "L3_category_name_en",
    "L4_category_name_en",
)
KNOWN_FUTURE_FEATURES = (
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
)
ABLATION_FEATURES = {
    "A1": tuple(
        feature
        for feature in APPROVED_FEATURES
        if feature not in {"sell_price_main", *DISCOUNT_COLUMNS}
    ),
    "A2": HISTORICAL_FEATURES,
    "A3": STATIC_FEATURES,
    "A4": (*STATIC_FEATURES, *KNOWN_FUTURE_FEATURES),
}
ABLATION_DESCRIPTIONS = {
    "A1": "LightGBM without price-discount features",
    "A2": "LightGBM historical-demand-only",
    "A3": "LightGBM static-metadata-only",
    "A4": "LightGBM known-future-plus-static",
}


def peak_rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def select_ablation_matrix(matrix: pd.DataFrame, ablation_id: str) -> pd.DataFrame:
    """Select one exact approved subset and fail closed on schema drift."""

    try:
        features = ABLATION_FEATURES[ablation_id]
    except KeyError as exc:
        raise ValueError(f"Unknown approved ablation: {ablation_id!r}.") from exc
    if len(features) != len(set(features)):
        raise AssertionError(f"{ablation_id} contains duplicate feature names.")
    missing = set(features) - set(matrix.columns)
    if missing:
        raise KeyError(f"{ablation_id} matrix is missing features: {sorted(missing)}.")
    selected = matrix.loc[:, features].copy()
    if tuple(selected.columns) != features:
        raise AssertionError(f"{ablation_id} feature order differs from its contract.")
    forbidden = FORBIDDEN_FEATURE_FIELDS.intersection(selected.columns)
    if forbidden:
        raise AssertionError(f"{ablation_id} contains forbidden fields: {forbidden}.")
    return selected


def categorical_features_for(ablation_id: str) -> tuple[str, ...]:
    """Return only reviewed categorical fields retained by an ablation."""

    selected = set(ABLATION_FEATURES[ablation_id])
    return tuple(feature for feature in CATEGORICAL_FEATURES if feature in selected)


def _comparison(metrics: object) -> dict[str, float]:
    return {
        "wmae_change_vs_full_stage3": float(metrics.wmae - FULL_STAGE3_F1["wmae"]),
        "wmae_improvement_vs_stage2_trailing14": float(
            F1_BASELINE["wmae"] - metrics.wmae
        ),
        "wape_change_vs_full_stage3": float(metrics.wape - FULL_STAGE3_F1["wape"]),
        "bias_change_vs_full_stage3": float(metrics.bias - FULL_STAGE3_F1["bias"]),
    }


def _baseline_result(
    ablation_id: str,
    description: str,
    feature_name: str,
    context: object,
    labels: pd.DataFrame,
    baseline_function: object,
) -> dict[str, object]:
    start = perf_counter()
    output = baseline_function(context)
    metrics = score_kaggle_aligned(labels, output.predictions)
    runtime_seconds = perf_counter() - start
    prediction = output.predictions["sales_hat"].to_numpy(dtype=np.float64)
    return {
        "ablation_id": ablation_id,
        "description": description,
        "kind": "baseline_reproduction",
        "features": [feature_name],
        "feature_count": 1,
        "training_rows": len(context.training),
        "validation_rows": metrics.rows,
        "metrics": {
            "wmae": metrics.wmae,
            "wape": metrics.wape,
            "bias": metrics.bias,
        },
        "runtime_seconds": runtime_seconds,
        "runtime_minutes": runtime_seconds / 60,
        "peak_rss_mib": peak_rss_mib(),
        "negative_predictions": int((prediction < 0).sum()),
        "prediction_minimum": float(prediction.min()),
        "prediction_maximum": float(prediction.max()),
        "comparison": _comparison(metrics),
    }


def main() -> None:
    total_start = perf_counter()
    history, inventory, calendar, official_grid = _load_raw_data()
    raw_load_seconds = perf_counter() - total_start

    if {"total_orders", "availability"}.intersection(history.columns):
        raise AssertionError("Forbidden future fields were loaded.")

    f1_fold = make_backtest_folds([F1_CUTOFF])[0]
    baseline_prep_start = perf_counter()
    f1_split = materialize_backtest_split(history, official_grid, f1_fold)
    if f1_split.scored_rows != EXPECTED_F1_ROWS:
        raise AssertionError("F1 scored-row count differs from the frozen contract.")
    coverage = f1_split.scored_rows / f1_split.requested_rows
    if not np.isclose(coverage, EXPECTED_F1_COVERAGE, rtol=0, atol=1e-15):
        raise AssertionError("F1 coverage differs from the frozen contract.")
    context = prepare_context(
        f1_split.training_history,
        f1_split.validation_features.loc[:, ["unique_id", "date"]],
        F1_CUTOFF,
    )
    labels = f1_split.validation_labels.loc[
        :, ["unique_id", "date", "sales", "weight"]
    ].copy()
    baseline_preparation_seconds = perf_counter() - baseline_prep_start

    results: list[dict[str, object]] = []
    a9 = _baseline_result(
        "A9",
        "reproduce Stage 2 F1 trailing 14-day baseline",
        "trailing_14_day_mean_by_unique_id",
        context,
        labels,
        trailing_14_day_mean_forecast,
    )
    a10 = _baseline_result(
        "A10",
        "reproduce Stage 2 F1 ID-day-of-week median baseline",
        "median_by_unique_id_and_day_of_week",
        context,
        labels,
        item_weekday_median_forecast,
    )
    for actual, expected, name in (
        (a9["metrics"], F1_BASELINE, "A9"),
        (a10["metrics"], STAGE2_ID_WEEKDAY_REFERENCE, "A10"),
    ):
        for metric_name in ("wmae", "wape", "bias"):
            if not np.isclose(
                actual[metric_name], expected[metric_name], rtol=0, atol=5e-10
            ):
                raise AssertionError(
                    f"{name} {metric_name} differs from its Stage 2 reference."
                )
    results.extend([a9, a10])
    del context, labels, f1_split
    gc.collect()

    feature_generation_start = perf_counter()
    origins = [F1_CUTOFF - pd.Timedelta(days=14 * index) for index in range(1, 13)]
    training_matrices: list[pd.DataFrame] = []
    training_targets: list[pd.Series] = []
    origin_rows: list[dict[str, object]] = []
    for origin in origins:
        fold = make_backtest_folds([origin])[0]
        split = materialize_backtest_split(history, official_grid, fold)
        expected_rows = EXPECTED_TRAINING_ROWS[str(origin.date())]
        if split.scored_rows != expected_rows:
            raise AssertionError(f"Origin {origin.date()} row count changed.")
        if split.validation_labels["date"].max() > F1_CUTOFF:
            raise AssertionError("A training label crosses the F1 cutoff.")
        batch = build_stage3_feature_batch(
            split.training_history,
            split.validation_features,
            inventory,
            calendar,
            origin,
        )
        if batch.maximum_history_date > origin:
            raise AssertionError("Historical feature lineage crosses its origin.")
        training_matrices.append(batch.matrix)
        training_targets.append(_aligned_target(batch.keys, split.validation_labels))
        origin_rows.append({"origin": str(origin.date()), "rows": len(batch.matrix)})
        del split, batch
        gc.collect()

    training_features = pd.concat(training_matrices, ignore_index=True)
    training_target = pd.concat(training_targets, ignore_index=True)
    del training_matrices, training_targets
    gc.collect()
    if len(training_features) != EXPECTED_TOTAL_TRAINING_ROWS:
        raise AssertionError("Combined training row count differs from the reference.")
    if tuple(training_features.columns) != APPROVED_FEATURES:
        raise AssertionError("Shared training matrix differs from the approved contract.")

    validation_split = materialize_backtest_split(history, official_grid, f1_fold)
    validation_batch = build_stage3_feature_batch(
        validation_split.training_history,
        validation_split.validation_features,
        inventory,
        calendar,
        F1_CUTOFF,
    )
    if len(validation_batch.matrix) != EXPECTED_F1_ROWS:
        raise AssertionError("Shared validation row count differs from the reference.")
    shared_feature_generation_seconds = perf_counter() - feature_generation_start

    config = PlainLightGBMConfig()
    for ablation_id in ("A1", "A2", "A3", "A4"):
        ablation_start = perf_counter()
        train_subset = select_ablation_matrix(training_features, ablation_id)
        validation_subset = select_ablation_matrix(validation_batch.matrix, ablation_id)
        categorical = categorical_features_for(ablation_id)
        model = train_plain_lightgbm(
            train_subset,
            training_target,
            categorical,
            config,
        )
        prediction = predict_plain_lightgbm(model, validation_subset)
        prediction_frame = validation_batch.keys.copy()
        prediction_frame["sales_hat"] = prediction
        if prediction_frame.loc[:, ["unique_id", "date"]].duplicated().any():
            raise AssertionError(f"{ablation_id} predictions contain duplicate keys.")
        metrics = score_kaggle_aligned(
            validation_split.validation_labels,
            prediction_frame,
        )
        runtime_seconds = perf_counter() - ablation_start
        results.append(
            {
                "ablation_id": ablation_id,
                "description": ABLATION_DESCRIPTIONS[ablation_id],
                "kind": "fixed_model_feature_ablation",
                "features": list(ABLATION_FEATURES[ablation_id]),
                "feature_count": len(ABLATION_FEATURES[ablation_id]),
                "categorical_features": list(categorical),
                "training_rows": len(train_subset),
                "validation_rows": metrics.rows,
                "metrics": {
                    "wmae": metrics.wmae,
                    "wape": metrics.wape,
                    "bias": metrics.bias,
                },
                "runtime_seconds": runtime_seconds,
                "runtime_minutes": runtime_seconds / 60,
                "peak_rss_mib": peak_rss_mib(),
                "negative_predictions": int((prediction < 0).sum()),
                "prediction_minimum": float(prediction.min()),
                "prediction_maximum": float(prediction.max()),
                "comparison": _comparison(metrics),
            }
        )
        del train_subset, validation_subset, model, prediction, prediction_frame, metrics
        gc.collect()

    output = {
        "stage": "stage_3_f1_diagnostic_ablations",
        "folds_run": ["F1"],
        "ablations_run": [row["ablation_id"] for row in results],
        "model_configuration": config.audit_dict(),
        "full_stage3_f1_reference": FULL_STAGE3_F1,
        "stage2_trailing14_reference": F1_BASELINE,
        "stage2_id_weekday_reference": STAGE2_ID_WEEKDAY_REFERENCE,
        "training_origins": origin_rows,
        "shared": {
            "raw_load_seconds": raw_load_seconds,
            "baseline_preparation_seconds": baseline_preparation_seconds,
            "model_feature_generation_seconds": shared_feature_generation_seconds,
            "training_rows": len(training_features),
            "validation_rows": len(validation_batch.matrix),
            "coverage": coverage,
            "total_runtime_seconds": perf_counter() - total_start,
            "peak_rss_mib": peak_rss_mib(),
        },
        "results": results,
        "diagnostic_only": True,
        "hyperparameter_tuning": False,
        "new_features_added": False,
        "predictions_clipped": False,
        "predictions_rounded": False,
        "kaggle_submission_made": False,
        "processed_data_written": False,
        "model_artifact_written": False,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
