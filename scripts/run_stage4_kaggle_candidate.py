#!/usr/bin/env python3
"""Build the final Stage 4 Kaggle candidate without submitting it."""

from __future__ import annotations

import gc
import json
import resource
from pathlib import Path
from time import perf_counter

import lightgbm
import numpy as np
import pandas as pd

from dataguard.cutoff import filter_history_at_cutoff
from eval.backtest import make_backtest_folds, materialize_backtest_split
from eval.grid import build_official_test_grid, validate_solution_template
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
from scripts.run_stage3_all_folds_plain_model import training_origins_for_fold
from scripts.run_stage3_f1_plain_model import _aligned_target


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SUBMISSIONS = ROOT / "submissions"
REPORTS = ROOT / "reports"
FINAL_CUTOFF = pd.Timestamp("2024-06-02")
OFFICIAL_FORECAST_START = pd.Timestamp("2024-06-03")
OFFICIAL_FORECAST_END = pd.Timestamp("2024-06-16")
SUBMISSION_PATH = SUBMISSIONS / "stage4_plain_lgbm_candidate.csv"
REPORT_PATH = REPORTS / "stage4_kaggle_candidate_report.md"


def peak_rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    history = pd.read_csv(
        RAW / "sales_train.csv",
        usecols=[
            "unique_id",
            "date",
            "warehouse",
            "sales",
            "sell_price_main",
            *DISCOUNT_COLUMNS,
        ],
        dtype={
            "unique_id": "uint16",
            "warehouse": "category",
            "sales": "float64",
            "sell_price_main": "float32",
            **{column: "float32" for column in DISCOUNT_COLUMNS},
        },
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
        dtype={
            "unique_id": "uint16",
            "warehouse": "category",
            "total_orders": "float64",
            "sell_price_main": "float32",
            **{column: "float32" for column in DISCOUNT_COLUMNS},
        },
        parse_dates=["date"],
        date_format="%Y-%m-%d",
    )
    weights = pd.read_csv(
        RAW / "test_weights.csv",
        dtype={"unique_id": "uint16", "weight": "float64"},
    )
    solution = pd.read_csv(RAW / "solution.csv")
    official_grid = build_official_test_grid(sales_test, weights)
    validate_solution_template(official_grid, solution)
    return history, inventory, calendar, sales_test, official_grid, solution


def _build_training_frame(
    history: pd.DataFrame,
    inventory: pd.DataFrame,
    calendar: pd.DataFrame,
    official_grid: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[dict[str, object]]]:
    training_matrices: list[pd.DataFrame] = []
    training_targets: list[pd.Series] = []
    audit_rows: list[dict[str, object]] = []

    for origin in training_origins_for_fold(FINAL_CUTOFF):
        fold = make_backtest_folds([origin])[0]
        split = materialize_backtest_split(history, official_grid, fold)
        if split.validation_labels["date"].max() > FINAL_CUTOFF:
            raise AssertionError("Training labels cross the final cutoff.")

        batch = build_stage3_feature_batch(
            split.training_history,
            split.validation_features,
            inventory,
            calendar,
            origin,
        )
        if tuple(batch.matrix.columns) != APPROVED_FEATURES:
            raise AssertionError("Training features differ from the approved contract.")
        if FORBIDDEN_FEATURE_FIELDS.intersection(batch.matrix.columns):
            raise AssertionError("Training features contain a forbidden field.")
        target = _aligned_target(batch.keys, split.validation_labels)

        training_matrices.append(batch.matrix)
        training_targets.append(target)
        audit_rows.append(
            {
                "origin": str(pd.Timestamp(origin).date()),
                "target_start": str(fold.validation_start.date()),
                "target_end": str(fold.validation_end.date()),
                "training_examples": len(batch.matrix),
                "maximum_history_feature_date": str(batch.maximum_history_date.date()),
                "maximum_same_weekday_source_date": (
                    str(batch.maximum_same_weekday_source_date.date())
                    if batch.maximum_same_weekday_source_date is not None
                    else None
                ),
            }
        )
        del split, batch, target
        gc.collect()

    training_features = pd.concat(training_matrices, ignore_index=True)
    training_target = pd.concat(training_targets, ignore_index=True)
    if tuple(training_features.columns) != APPROVED_FEATURES:
        raise AssertionError("Combined training features differ from the approved contract.")
    return training_features, training_target, audit_rows


def _build_official_request_frame(sales_test: pd.DataFrame, official_grid: pd.DataFrame) -> pd.DataFrame:
    requests = sales_test.drop(columns=["total_orders"]).copy()
    requests = requests.merge(
        official_grid.loc[:, ["unique_id", "test_date", "horizon_day"]],
        left_on=["unique_id", "date"],
        right_on=["unique_id", "test_date"],
        how="left",
        validate="one_to_one",
    ).drop(columns=["test_date"])
    if "total_orders" in requests.columns:
        raise AssertionError("Future total_orders leaked into the request frame.")
    return requests


def _validate_submission_frame(
    official_grid: pd.DataFrame,
    solution: pd.DataFrame,
    submission: pd.DataFrame,
) -> None:
    if list(submission.columns) != ["id", "sales_hat"]:
        raise AssertionError("Submission columns must be exactly id and sales_hat.")
    if len(submission) != len(solution):
        raise AssertionError("Submission row count does not match solution.csv.")
    if submission["id"].duplicated().any():
        raise AssertionError("Submission contains duplicate ids.")
    if submission["sales_hat"].isna().any():
        raise AssertionError("Submission contains null predictions.")
    if not np.issubdtype(submission["sales_hat"].dtype, np.number):
        raise AssertionError("Submission predictions must be numeric.")
    expected = solution["id"].astype(str).reset_index(drop=True)
    actual = submission["id"].astype(str).reset_index(drop=True)
    if not actual.equals(expected):
        raise AssertionError("Submission order does not match solution.csv.")
    expected_ids = (
        official_grid["unique_id"].astype(str)
        + "_"
        + pd.to_datetime(official_grid["test_date"]).dt.strftime("%Y-%m-%d")
    )
    if not actual.equals(expected_ids.reset_index(drop=True)):
        raise AssertionError("Submission ids do not match the official grid.")


def run_candidate() -> dict[str, object]:
    start = perf_counter()
    history, inventory, calendar, sales_test, official_grid, solution = _load_raw_data()
    raw_load_seconds = perf_counter() - start

    history_at_cutoff = filter_history_at_cutoff(history, FINAL_CUTOFF)
    if history_at_cutoff["date"].max() > FINAL_CUTOFF:
        raise AssertionError("History at cutoff contains future rows.")
    history_for_official_features = history_at_cutoff.dropna(subset=["sales"]).copy()

    training_start = perf_counter()
    training_features, training_target, origin_audit = _build_training_frame(
        history_at_cutoff,
        inventory,
        calendar,
        official_grid,
    )
    training_seconds = perf_counter() - training_start

    request_frame = _build_official_request_frame(sales_test, official_grid)
    if FORBIDDEN_FEATURE_FIELDS.intersection(request_frame.columns):
        raise AssertionError("Official request frame contains forbidden fields.")

    feature_start = perf_counter()
    batch = build_stage3_feature_batch(
        history_for_official_features,
        request_frame,
        inventory,
        calendar,
        FINAL_CUTOFF,
    )
    feature_seconds = perf_counter() - feature_start
    if tuple(batch.matrix.columns) != APPROVED_FEATURES:
        raise AssertionError("Official candidate features differ from the approved contract.")
    if FORBIDDEN_FEATURE_FIELDS.intersection(batch.matrix.columns):
        raise AssertionError("Official candidate features contain a forbidden field.")

    config = PlainLightGBMConfig()
    fit_start = perf_counter()
    model = train_plain_lightgbm(training_features, training_target, CATEGORICAL_FEATURES, config)
    fit_seconds = perf_counter() - fit_start

    predict_start = perf_counter()
    prediction = predict_plain_lightgbm(model, batch.matrix)
    prediction_seconds = perf_counter() - predict_start
    submission = solution.loc[:, ["id"]].copy()
    submission["sales_hat"] = prediction
    _validate_submission_frame(official_grid, solution, submission)

    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)

    total_seconds = perf_counter() - start
    result = {
        "stage": "stage_4_kaggle_candidate",
        "lightgbm_version": lightgbm.__version__,
        "model_configuration": config.audit_dict(),
        "feature_count": len(APPROVED_FEATURES),
        "features": list(APPROVED_FEATURES),
        "feature_availability": dict(FEATURE_AVAILABILITY),
        "training_origins": origin_audit,
        "training_rows": len(training_features),
        "submission_rows": len(submission),
        "submission_path": str(SUBMISSION_PATH),
        "official_grid_rows": len(official_grid),
        "official_forecast_window": {
            "start": str(OFFICIAL_FORECAST_START.date()),
            "end": str(OFFICIAL_FORECAST_END.date()),
        },
        "prediction_summary": {
            "minimum": float(prediction.min()),
            "maximum": float(prediction.max()),
            "mean": float(prediction.mean()),
            "median": float(np.median(prediction)),
            "negative_count": int((prediction < 0).sum()),
            "null_count": int(np.isnan(prediction).sum()),
        },
        "runtime": {
            "raw_load_seconds": raw_load_seconds,
            "training_seconds": training_seconds,
            "feature_seconds": feature_seconds,
            "fit_seconds": fit_seconds,
            "prediction_seconds": prediction_seconds,
            "total_seconds": total_seconds,
            "total_minutes": total_seconds / 60,
        },
        "peak_rss_mib": peak_rss_mib(),
        "submission_validation": {
            "row_count_matches_solution": len(submission) == len(solution),
            "order_matches_solution": bool(
                submission["id"].astype(str).reset_index(drop=True).equals(
                    solution["id"].astype(str).reset_index(drop=True)
                )
            ),
            "official_grid_rows": len(official_grid),
            "solution_rows": len(solution),
            "no_forbidden_fields_in_request": not FORBIDDEN_FEATURE_FIELDS.intersection(
                request_frame.columns
            ),
            "no_forbidden_fields_in_features": not FORBIDDEN_FEATURE_FIELDS.intersection(
                batch.matrix.columns
            ),
        },
        "caveats": [
            "Price and discount features are benchmark-specific Kaggle-known-future covariates.",
            "Local folds already triggered the suspicious-improvement threshold.",
            "F2 under-forecasting bias remains a caveat from the all-fold review.",
            "Official Kaggle score remains unknown until submission.",
        ],
        "kaggle_submission_made": False,
        "processed_data_written": False,
        "model_artifact_written": False,
        "predictions_clipped": False,
        "predictions_rounded": False,
        "future_total_orders_used": False,
        "future_availability_used": False,
        "target_transform_used": False,
        "recursive_prediction_used": False,
    }
    report_lines = [
        "# Stage 4 Kaggle Candidate Report",
        "",
        f"Date: {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d')}",
        "Status: Candidate prepared; pending human review before Kaggle submission",
        "",
        "## Model configuration",
        "",
        "| Setting | Value |",
        "|---|---|",
        f"| Library | LightGBM {lightgbm.__version__} |",
        f"| Objective | {config.objective} |",
        "| Target | Raw `sales` |",
        f"| Boosting rounds | {config.num_boost_round} |",
        f"| Learning rate | {config.learning_rate} |",
        f"| Leaves | {config.num_leaves} |",
        f"| Minimum rows per leaf | {config.min_data_in_leaf} |",
        "| Clipping / rounding | None |",
        "| Recursive prediction | None |",
        "| Target transform | None |",
        "",
        "## Final training design",
        "",
        f"- Final cutoff: {FINAL_CUTOFF.date()}",
        f"- Official forecast window: {OFFICIAL_FORECAST_START.date()} through {OFFICIAL_FORECAST_END.date()}",
        "- Twelve cutoff-safe historical origins are used, shifted forward to the official cutoff.",
        "- Training labels remain historical only and end on or before the final cutoff.",
        "",
        "## Feature contract",
        "",
        f"- Approved features: {len(APPROVED_FEATURES)}",
        "- Static metadata, known-future price/discount/calendar features, and historical-only demand summaries are retained exactly as approved.",
        "- Future `total_orders` and future availability are excluded.",
        "",
        "## Submission validation",
        "",
        f"- Submission path: `{SUBMISSION_PATH.relative_to(ROOT)}`",
        f"- Row count: {len(submission)}",
        "- Order matches `solution.csv` and the official grid.",
        "- No missing or duplicate IDs.",
        "- Predictions are numeric and non-null.",
        "",
        "## Prediction summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Minimum | {result['prediction_summary']['minimum']:.6f} |",
        f"| Maximum | {result['prediction_summary']['maximum']:.6f} |",
        f"| Mean | {result['prediction_summary']['mean']:.6f} |",
        f"| Median | {result['prediction_summary']['median']:.6f} |",
        f"| Negative count | {result['prediction_summary']['negative_count']} |",
        f"| Null count | {result['prediction_summary']['null_count']} |",
        "",
        "## Runtime and memory",
        "",
        "| Phase | Seconds |",
        "|---|---:|",
        f"| Raw load | {raw_load_seconds:.3f} |",
        f"| Training-origin feature build | {training_seconds:.3f} |",
        f"| Official-grid feature build | {feature_seconds:.3f} |",
        f"| Model fit | {fit_seconds:.3f} |",
        f"| Prediction | {prediction_seconds:.3f} |",
        f"| Total | {total_seconds:.3f} |",
        f"| Peak RSS (MiB) | {result['peak_rss_mib']:.2f} |",
        "",
        "## Caveats",
        "",
        "- Price/discount covariates are benchmark-specific known-future inputs.",
        "- The local all-fold run triggered the suspicious-improvement threshold on every fold.",
        "- F2 carried elevated negative bias in the diagnostic review.",
        "- The Kaggle score is still unknown until a human approves submission.",
        "",
        "## Recommendation",
        "",
        "Ready for human review before Kaggle submission.",
    ]
    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run_candidate()
