#!/usr/bin/env python3
"""Run an explicitly selected approved Stage 3 F1 diagnostic ablation batch."""

from __future__ import annotations

import argparse
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
PRIORITY2_DESCRIPTIONS = {
    "A5": "LightGBM single-origin full-feature model",
    "A6": "LightGBM full-feature model without categorical handling",
}
PRIORITY2_FEATURES = {
    "A5": APPROVED_FEATURES,
    "A6": APPROVED_FEATURES,
}
PRIORITY2_CATEGORICAL_FEATURES = {
    "A5": CATEGORICAL_FEATURES,
    "A6": (),
}
MOST_RECENT_ORIGIN = pd.Timestamp("2024-05-05")


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


def _build_shared_model_frames(
    history: pd.DataFrame,
    inventory: pd.DataFrame,
    calendar: pd.DataFrame,
    official_grid: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    object,
    pd.DataFrame,
    list[dict[str, object]],
    float,
    float,
]:
    """Build the reviewed twelve-origin and F1 matrices once per batch."""

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
    if origin_rows[0] != {
        "origin": str(MOST_RECENT_ORIGIN.date()),
        "rows": EXPECTED_TRAINING_ROWS[str(MOST_RECENT_ORIGIN.date())],
    }:
        raise AssertionError("The most recent origin is not the first training block.")

    f1_fold = make_backtest_folds([F1_CUTOFF])[0]
    validation_split = materialize_backtest_split(history, official_grid, f1_fold)
    if validation_split.scored_rows != EXPECTED_F1_ROWS:
        raise AssertionError("F1 scored-row count differs from the frozen contract.")
    coverage = validation_split.scored_rows / validation_split.requested_rows
    if not np.isclose(coverage, EXPECTED_F1_COVERAGE, rtol=0, atol=1e-15):
        raise AssertionError("F1 coverage differs from the frozen contract.")
    validation_batch = build_stage3_feature_batch(
        validation_split.training_history,
        validation_split.validation_features,
        inventory,
        calendar,
        F1_CUTOFF,
    )
    if len(validation_batch.matrix) != EXPECTED_F1_ROWS:
        raise AssertionError("Shared validation row count differs from the reference.")
    validation_labels = validation_split.validation_labels.copy()
    del validation_split
    gc.collect()
    return (
        training_features,
        training_target,
        validation_batch,
        validation_labels,
        origin_rows,
        coverage,
        perf_counter() - feature_generation_start,
    )


def _fixed_model_result(
    ablation_id: str,
    description: str,
    training_features: pd.DataFrame,
    training_target: pd.Series,
    validation_features: pd.DataFrame,
    validation_keys: pd.DataFrame,
    validation_labels: pd.DataFrame,
    categorical_features: tuple[str, ...],
    config: PlainLightGBMConfig,
    *,
    kind: str,
) -> dict[str, object]:
    """Train and score one fixed diagnostic model without shared preparation."""

    start = perf_counter()
    if tuple(training_features.columns) != tuple(validation_features.columns):
        raise AssertionError(f"{ablation_id} train and validation schemas differ.")
    if FORBIDDEN_FEATURE_FIELDS.intersection(training_features.columns):
        raise AssertionError(f"{ablation_id} contains a forbidden feature.")
    model = train_plain_lightgbm(
        training_features,
        training_target,
        categorical_features,
        config,
    )
    prediction = predict_plain_lightgbm(model, validation_features)
    prediction_frame = validation_keys.copy()
    prediction_frame["sales_hat"] = prediction
    if prediction_frame.loc[:, ["unique_id", "date"]].duplicated().any():
        raise AssertionError(f"{ablation_id} predictions contain duplicate keys.")
    metrics = score_kaggle_aligned(validation_labels, prediction_frame)
    runtime_seconds = perf_counter() - start
    return {
        "ablation_id": ablation_id,
        "description": description,
        "kind": kind,
        "features": list(training_features.columns),
        "feature_count": len(training_features.columns),
        "categorical_features": list(categorical_features),
        "training_rows": len(training_features),
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


def _load_checked_raw_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    history, inventory, calendar, official_grid = _load_raw_data()
    if {"total_orders", "availability"}.intersection(history.columns):
        raise AssertionError("Forbidden future fields were loaded.")
    return history, inventory, calendar, official_grid


def run_priority1() -> dict[str, object]:
    """Reproduce committed A9/A10/A1–A4 when explicitly requested."""

    total_start = perf_counter()
    history, inventory, calendar, official_grid = _load_checked_raw_data()
    raw_load_seconds = perf_counter() - total_start

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

    (
        training_features,
        training_target,
        validation_batch,
        validation_labels,
        origin_rows,
        coverage,
        shared_feature_generation_seconds,
    ) = _build_shared_model_frames(
        history,
        inventory,
        calendar,
        official_grid,
    )

    config = PlainLightGBMConfig()
    for ablation_id in ("A1", "A2", "A3", "A4"):
        train_subset = select_ablation_matrix(training_features, ablation_id)
        validation_subset = select_ablation_matrix(validation_batch.matrix, ablation_id)
        categorical = categorical_features_for(ablation_id)
        results.append(
            _fixed_model_result(
                ablation_id,
                ABLATION_DESCRIPTIONS[ablation_id],
                train_subset,
                training_target,
                validation_subset,
                validation_batch.keys,
                validation_labels,
                categorical,
                config,
                kind="fixed_model_feature_ablation",
            )
        )
        del train_subset, validation_subset
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
    return output


def run_priority2() -> dict[str, object]:
    """Run only approved A5 and meaningful categorical-treatment A6 diagnostics."""

    total_start = perf_counter()
    history, inventory, calendar, official_grid = _load_checked_raw_data()
    raw_load_seconds = perf_counter() - total_start
    (
        training_features,
        training_target,
        validation_batch,
        validation_labels,
        origin_rows,
        coverage,
        shared_feature_generation_seconds,
    ) = _build_shared_model_frames(
        history,
        inventory,
        calendar,
        official_grid,
    )

    config = PlainLightGBMConfig()
    recent_rows = EXPECTED_TRAINING_ROWS[str(MOST_RECENT_ORIGIN.date())]
    a5_train = training_features.iloc[:recent_rows].copy()
    a5_target = training_target.iloc[:recent_rows].copy()
    if len(a5_train) != 43433:
        raise AssertionError("A5 must use exactly 43,433 most-recent-origin rows.")

    results = [
        _fixed_model_result(
            "A5",
            PRIORITY2_DESCRIPTIONS["A5"],
            a5_train,
            a5_target,
            validation_batch.matrix,
            validation_batch.keys,
            validation_labels,
            PRIORITY2_CATEGORICAL_FEATURES["A5"],
            config,
            kind="single_origin_design_ablation",
        )
    ]
    del a5_train, a5_target
    gc.collect()

    # The reference passes deterministic inventory-derived integer codes as
    # categorical fields. An empty list changes only their LightGBM treatment.
    results.append(
        _fixed_model_result(
            "A6",
            PRIORITY2_DESCRIPTIONS["A6"],
            training_features,
            training_target,
            validation_batch.matrix,
            validation_batch.keys,
            validation_labels,
            PRIORITY2_CATEGORICAL_FEATURES["A6"],
            config,
            kind="categorical_treatment_ablation",
        )
    )

    return {
        "stage": "stage_3_f1_priority2_ablations",
        "folds_run": ["F1"],
        "ablations_run": ["A5", "A6"],
        "a6_feasibility": {
            "meaningful": True,
            "reason": (
                "The reference uses deterministic inventory-derived integer codes "
                "and explicitly passes nine fields as categorical to LightGBM. "
                "A6 keeps identical codes and columns but passes no categorical fields."
            ),
            "caveat": (
                "Numeric treatment imposes arbitrary ordinal relationships and is "
                "diagnostic only; it cannot replace the reviewed reference model."
            ),
        },
        "model_configuration": config.audit_dict(),
        "full_stage3_f1_reference": FULL_STAGE3_F1,
        "stage2_trailing14_reference": F1_BASELINE,
        "training_origins": origin_rows,
        "shared": {
            "raw_load_seconds": raw_load_seconds,
            "model_feature_generation_seconds": shared_feature_generation_seconds,
            "full_training_rows": len(training_features),
            "a5_training_rows": recent_rows,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch",
        choices=("priority1", "priority2"),
        default="priority1",
        help="Approved ablation batch to run; defaults to committed Priority 1.",
    )
    args = parser.parse_args()
    output = run_priority1() if args.batch == "priority1" else run_priority2()
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
