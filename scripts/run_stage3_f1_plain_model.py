#!/usr/bin/env python3
"""Run the approved Stage 3 plain LightGBM model on F1 only."""

from __future__ import annotations

import gc
import json
import resource
from pathlib import Path
from time import perf_counter

import lightgbm
import numpy as np
import pandas as pd

from eval.backtest import make_backtest_folds, materialize_backtest_split
from eval.grid import build_official_test_grid, score_kaggle_aligned
from features.stage3_minimal import (
    APPROVED_FEATURES,
    CATEGORICAL_FEATURES,
    DISCOUNT_COLUMNS,
    FEATURE_AVAILABILITY,
    FORBIDDEN_FEATURE_FIELDS,
    build_stage3_feature_batch,
)
from models.plain_lgbm import (
    PlainLightGBMConfig,
    predict_plain_lightgbm,
    train_plain_lightgbm,
)


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
F1_CUTOFF = pd.Timestamp("2024-05-19")
F1_BASELINE = {
    "wmae": 31.19858033084802,
    "wape": 0.30211101284954356,
    "bias": 0.030395766081574085,
}
EXPECTED_TRAINING_ROWS = {
    "2024-05-05": 43433,
    "2024-04-21": 42794,
    "2024-04-07": 42035,
    "2024-03-24": 40993,
    "2024-03-10": 41138,
    "2024-02-25": 41190,
    "2024-02-11": 41077,
    "2024-01-28": 41208,
    "2024-01-14": 40749,
    "2023-12-31": 38945,
    "2023-12-17": 38434,
    "2023-12-03": 40830,
}
EXPECTED_TOTAL_TRAINING_ROWS = 492826
EXPECTED_F1_ROWS = 44212
EXPECTED_F1_COVERAGE = 0.9402607345654069


def peak_rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _aligned_target(keys: pd.DataFrame, labels: pd.DataFrame) -> pd.Series:
    if labels.loc[:, ["unique_id", "date"]].duplicated().any():
        raise ValueError("Labels contain duplicate prediction keys.")
    aligned = keys.merge(
        labels.loc[:, ["unique_id", "date", "sales"]],
        on=["unique_id", "date"],
        how="left",
        validate="one_to_one",
    )
    if len(aligned) != len(keys) or aligned["sales"].isna().any():
        raise ValueError("Feature keys do not align exactly with available labels.")
    return aligned["sales"].astype(np.float64)


def _load_raw_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    sales_dtype: dict[str, str] = {
        "unique_id": "uint16",
        "warehouse": "category",
        "sales": "float64",
        "sell_price_main": "float32",
        **{column: "float32" for column in DISCOUNT_COLUMNS},
    }
    sales_columns = [
        "unique_id",
        "date",
        "warehouse",
        "sales",
        "sell_price_main",
        *DISCOUNT_COLUMNS,
    ]
    history = pd.read_csv(
        RAW / "sales_train.csv",
        usecols=sales_columns,
        dtype=sales_dtype,
        parse_dates=["date"],
        date_format="%Y-%m-%d",
    )
    inventory = pd.read_csv(
        RAW / "inventory.csv",
        usecols=[
            "unique_id",
            "product_unique_id",
            "warehouse",
            "L1_category_name_en",
            "L2_category_name_en",
            "L3_category_name_en",
            "L4_category_name_en",
        ],
        dtype={"unique_id": "uint16", "product_unique_id": "uint16"},
    )
    calendar = pd.read_csv(
        RAW / "calendar.csv",
        usecols=[
            "date",
            "warehouse",
            "holiday",
            "shops_closed",
            "winter_school_holidays",
            "school_holidays",
        ],
        dtype={
            "holiday": "int8",
            "shops_closed": "int8",
            "winter_school_holidays": "int8",
            "school_holidays": "int8",
        },
        parse_dates=["date"],
        date_format="%Y-%m-%d",
    )
    sales_test = pd.read_csv(
        RAW / "sales_test.csv",
        usecols=["unique_id", "date"],
        dtype={"unique_id": "uint16"},
    )
    weights = pd.read_csv(
        RAW / "test_weights.csv",
        dtype={"unique_id": "uint16", "weight": "float64"},
    )
    official_grid = build_official_test_grid(sales_test, weights)
    del sales_test, weights
    return history, inventory, calendar, official_grid


def main() -> None:
    run_start = perf_counter()
    history, inventory, calendar, official_grid = _load_raw_data()
    raw_load_seconds = perf_counter() - run_start

    forbidden_loaded = {"total_orders", "availability"}.intersection(history.columns)
    if forbidden_loaded:
        raise AssertionError(f"Forbidden future fields were loaded: {forbidden_loaded}.")

    origins = [F1_CUTOFF - pd.Timedelta(days=14 * index) for index in range(1, 13)]
    train_matrices: list[pd.DataFrame] = []
    train_targets: list[pd.Series] = []
    origin_audit: list[dict[str, object]] = []

    for origin in origins:
        origin_start = perf_counter()
        fold = make_backtest_folds([origin])[0]
        split = materialize_backtest_split(history, official_grid, fold)
        origin_key = str(origin.date())
        expected_rows = EXPECTED_TRAINING_ROWS[origin_key]
        if split.scored_rows != expected_rows:
            raise AssertionError(
                f"Origin {origin_key} has {split.scored_rows} rows; expected {expected_rows}."
            )
        if split.validation_labels["date"].max() > F1_CUTOFF:
            raise AssertionError("A historical training label crosses the F1 cutoff.")

        batch = build_stage3_feature_batch(
            split.training_history,
            split.validation_features,
            inventory,
            calendar,
            origin,
        )
        target = _aligned_target(batch.keys, split.validation_labels)
        if len(batch.matrix) != expected_rows:
            raise AssertionError("Feature row count differs from the approved origin count.")
        if batch.maximum_history_date > origin:
            raise AssertionError("Historical feature lineage crosses its origin.")
        if (
            batch.maximum_same_weekday_source_date is not None
            and batch.maximum_same_weekday_source_date > origin
        ):
            raise AssertionError("Same-weekday lineage crosses its origin.")

        train_matrices.append(batch.matrix)
        train_targets.append(target)
        origin_audit.append(
            {
                "origin": origin_key,
                "target_start": str(fold.validation_start.date()),
                "target_end": str(fold.validation_end.date()),
                "training_examples": len(batch.matrix),
                "maximum_history_feature_date": str(batch.maximum_history_date.date()),
                "maximum_same_weekday_source_date": (
                    str(batch.maximum_same_weekday_source_date.date())
                    if batch.maximum_same_weekday_source_date is not None
                    else None
                ),
                "runtime_seconds": perf_counter() - origin_start,
            }
        )
        del split, batch, target
        gc.collect()

    training_features = pd.concat(train_matrices, ignore_index=True)
    training_target = pd.concat(train_targets, ignore_index=True)
    del train_matrices, train_targets
    gc.collect()
    if len(training_features) != EXPECTED_TOTAL_TRAINING_ROWS:
        raise AssertionError(
            f"Training frame has {len(training_features)} rows; "
            f"expected {EXPECTED_TOTAL_TRAINING_ROWS}."
        )
    if tuple(training_features.columns) != APPROVED_FEATURES:
        raise AssertionError("Training columns differ from the approved feature contract.")
    if FORBIDDEN_FEATURE_FIELDS.intersection(training_features.columns):
        raise AssertionError("Training matrix contains a forbidden field.")
    feature_generation_seconds = perf_counter() - run_start - raw_load_seconds

    config = PlainLightGBMConfig()
    fit_start = perf_counter()
    model = train_plain_lightgbm(
        training_features,
        training_target,
        CATEGORICAL_FEATURES,
        config,
    )
    fit_seconds = perf_counter() - fit_start

    validation_start = perf_counter()
    f1_fold = make_backtest_folds([F1_CUTOFF])[0]
    f1_split = materialize_backtest_split(history, official_grid, f1_fold)
    if f1_split.scored_rows != EXPECTED_F1_ROWS:
        raise AssertionError("F1 scored-row count differs from the frozen contract.")
    f1_coverage = f1_split.scored_rows / f1_split.requested_rows
    if not np.isclose(f1_coverage, EXPECTED_F1_COVERAGE, rtol=0, atol=1e-15):
        raise AssertionError("F1 coverage differs from the frozen contract.")
    validation_batch = build_stage3_feature_batch(
        f1_split.training_history,
        f1_split.validation_features,
        inventory,
        calendar,
        F1_CUTOFF,
    )
    if tuple(validation_batch.matrix.columns) != APPROVED_FEATURES:
        raise AssertionError("Validation columns differ from the approved contract.")
    if validation_batch.maximum_history_date > F1_CUTOFF:
        raise AssertionError("F1 feature lineage crosses the cutoff.")
    validation_feature_seconds = perf_counter() - validation_start

    prediction_start = perf_counter()
    prediction = predict_plain_lightgbm(model, validation_batch.matrix)
    prediction_frame = validation_batch.keys.copy()
    prediction_frame["sales_hat"] = prediction
    if prediction_frame.loc[:, ["unique_id", "date"]].duplicated().any():
        raise AssertionError("Predictions contain duplicate keys.")
    metrics = score_kaggle_aligned(f1_split.validation_labels, prediction_frame)
    prediction_scoring_seconds = perf_counter() - prediction_start
    total_runtime_seconds = perf_counter() - run_start
    peak_memory_mib = peak_rss_mib()

    gates = {
        "wmae_below_31_1985803308": metrics.wmae < 31.1985803308,
        "wape_at_most_0_3081532331": metrics.wape <= 0.3081532331,
        "absolute_bias_at_most_0_10": abs(metrics.bias) <= 0.10,
        "runtime_below_30_minutes": total_runtime_seconds < 30 * 60,
        "peak_rss_below_12_gib": peak_memory_mib < 12 * 1024,
        "f1_rows_and_coverage_match": bool(
            metrics.rows == EXPECTED_F1_ROWS
            and np.isclose(f1_coverage, EXPECTED_F1_COVERAGE, rtol=0, atol=1e-15)
        ),
    }
    result = {
        "stage": "stage_3_f1_plain_lightgbm",
        "lightgbm_version": lightgbm.__version__,
        "model_configuration": config.audit_dict(),
        "feature_availability": FEATURE_AVAILABILITY,
        "feature_count": len(APPROVED_FEATURES),
        "features": list(APPROVED_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "training_origins": origin_audit,
        "training_rows": len(training_features),
        "validation_rows": len(validation_batch.matrix),
        "requested_validation_rows": f1_split.requested_rows,
        "coverage": f1_coverage,
        "metrics": {
            "wmae": metrics.wmae,
            "wape": metrics.wape,
            "bias": metrics.bias,
        },
        "stage2_f1_trailing_14_baseline": F1_BASELINE,
        "comparison": {
            "wmae_absolute_improvement": F1_BASELINE["wmae"] - metrics.wmae,
            "wmae_relative_improvement": (
                F1_BASELINE["wmae"] - metrics.wmae
            )
            / F1_BASELINE["wmae"],
            "wape_change": metrics.wape - F1_BASELINE["wape"],
            "bias_change": metrics.bias - F1_BASELINE["bias"],
        },
        "negative_predictions": int((prediction < 0).sum()),
        "prediction_minimum": float(prediction.min()),
        "prediction_maximum": float(prediction.max()),
        "runtime": {
            "raw_load_seconds": raw_load_seconds,
            "training_feature_generation_seconds": feature_generation_seconds,
            "model_fit_seconds": fit_seconds,
            "validation_feature_generation_seconds": validation_feature_seconds,
            "prediction_and_scoring_seconds": prediction_scoring_seconds,
            "total_seconds": total_runtime_seconds,
            "total_minutes": total_runtime_seconds / 60,
        },
        "peak_rss_mib": peak_memory_mib,
        "success_gates_before_post_run_tests": gates,
        "all_numeric_gates_passed": all(gates.values()),
        "predictions_clipped": False,
        "predictions_rounded": False,
        "future_total_orders_used": False,
        "future_availability_used": False,
        "outer_validation_used_for_fitting": False,
        "processed_data_written": False,
        "model_artifact_written": False,
        "kaggle_submission_made": False,
        "folds_run": ["F1"],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
