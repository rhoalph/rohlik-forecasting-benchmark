#!/usr/bin/env python3
"""Run the approved Stage 3 plain LightGBM model on F2, F3, and F4 only."""

from __future__ import annotations

import gc
import json
import resource
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd

from eval.backtest import make_backtest_folds, materialize_backtest_split
from eval.grid import score_kaggle_aligned
from features.stage3_minimal import (
    APPROVED_FEATURES,
    CATEGORICAL_FEATURES,
    FORBIDDEN_FEATURE_FIELDS,
    build_stage3_feature_batch,
)
from models.plain_lgbm import (
    PlainLightGBMConfig,
    predict_plain_lightgbm,
    train_plain_lightgbm,
)
from scripts.run_stage3_f1_plain_model import _aligned_target, _load_raw_data


@dataclass(frozen=True)
class FoldSpec:
    name: str
    cutoff: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    expected_scored_rows: int
    expected_coverage: float
    baseline_wmae: float
    baseline_wape: float
    baseline_bias: float


FOLD_SPECS = (
    FoldSpec(
        "F2",
        pd.Timestamp("2024-05-05"),
        pd.Timestamp("2024-05-06"),
        pd.Timestamp("2024-05-19"),
        43433,
        0.923693668786287,
        32.94127959626082,
        0.29535323859147833,
        -0.04835594182977103,
    ),
    FoldSpec(
        "F3",
        pd.Timestamp("2024-04-21"),
        pd.Timestamp("2024-04-22"),
        pd.Timestamp("2024-05-05"),
        42794,
        0.9101039960868548,
        30.171353527521557,
        0.2969883500221636,
        0.04890629970833528,
    ),
    FoldSpec(
        "F4",
        pd.Timestamp("2024-04-07"),
        pd.Timestamp("2024-04-08"),
        pd.Timestamp("2024-04-21"),
        42035,
        0.8939622721762617,
        27.946453182718322,
        0.2952420975141513,
        -0.06185725891559534,
    ),
)
COMMITTED_F1 = {
    "fold": "F1",
    "cutoff": "2024-05-19",
    "validation_start": "2024-05-20",
    "validation_end": "2024-06-02",
    "training_rows": 492826,
    "validation_rows": 44212,
    "coverage": 0.9402607345654069,
    "wmae": 20.58275093511847,
    "wape": 0.21869379320058285,
    "bias": -0.015387375961508831,
    "runtime_seconds": 209.82658585699392,
    "peak_rss_mib": 1517.69921875,
    "negative_predictions": 13,
    "baseline_wmae": 31.19858033084802,
    "baseline_wape": 0.30211101284954356,
    "baseline_bias": 0.030395766081574085,
    "source": "committed; not rerun",
}
REQUESTED_GRID_ROWS = 47021
SUSPICIOUS_RELATIVE_IMPROVEMENT = 0.20


def peak_rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def training_origins_for_fold(cutoff: object) -> tuple[pd.Timestamp, ...]:
    """Return the twelve approved non-overlapping origins before a cutoff."""

    normalized = pd.Timestamp(cutoff).normalize()
    return tuple(normalized - pd.Timedelta(days=14 * index) for index in range(1, 13))


def suspicious_improvement(baseline_wmae: float, model_wmae: float) -> bool:
    """Apply the predeclared greater-than-20-percent review trigger."""

    return (baseline_wmae - model_wmae) / baseline_wmae > SUSPICIOUS_RELATIVE_IMPROVEMENT


def _build_fold_frames(
    spec: FoldSpec,
    history: pd.DataFrame,
    inventory: pd.DataFrame,
    calendar: pd.DataFrame,
    official_grid: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, object, pd.DataFrame, list[dict[str, object]]]:
    training_matrices: list[pd.DataFrame] = []
    training_targets: list[pd.Series] = []
    origins_audit: list[dict[str, object]] = []

    for origin in training_origins_for_fold(spec.cutoff):
        fold = make_backtest_folds([origin])[0]
        split = materialize_backtest_split(history, official_grid, fold)
        if split.validation_labels["date"].min() <= origin:
            raise AssertionError(f"{spec.name} training labels do not follow their origin.")
        if split.validation_labels["date"].max() > spec.cutoff:
            raise AssertionError(f"{spec.name} training labels cross the outer cutoff.")
        batch = build_stage3_feature_batch(
            split.training_history,
            split.validation_features,
            inventory,
            calendar,
            origin,
        )
        if batch.maximum_history_date > origin:
            raise AssertionError(f"{spec.name} historical feature lineage crosses origin.")
        if (
            batch.maximum_same_weekday_source_date is not None
            and batch.maximum_same_weekday_source_date > origin
        ):
            raise AssertionError(f"{spec.name} same-weekday lineage crosses origin.")
        if tuple(batch.matrix.columns) != APPROVED_FEATURES:
            raise AssertionError(f"{spec.name} training features differ from the contract.")
        if FORBIDDEN_FEATURE_FIELDS.intersection(batch.matrix.columns):
            raise AssertionError(f"{spec.name} training matrix has forbidden fields.")

        target = _aligned_target(batch.keys, split.validation_labels)
        training_matrices.append(batch.matrix)
        training_targets.append(target)
        origins_audit.append(
            {
                "origin": str(origin.date()),
                "target_start": str(fold.validation_start.date()),
                "target_end": str(fold.validation_end.date()),
                "rows": len(batch.matrix),
                "maximum_history_feature_date": str(batch.maximum_history_date.date()),
            }
        )
        del split, batch, target
        gc.collect()

    training_features = pd.concat(training_matrices, ignore_index=True)
    training_target = pd.concat(training_targets, ignore_index=True)
    del training_matrices, training_targets
    gc.collect()
    if tuple(training_features.columns) != APPROVED_FEATURES:
        raise AssertionError(f"{spec.name} combined features differ from the contract.")

    outer_fold = make_backtest_folds([spec.cutoff])[0]
    if outer_fold.validation_start != spec.validation_start:
        raise AssertionError(f"{spec.name} validation start changed.")
    if outer_fold.validation_end != spec.validation_end:
        raise AssertionError(f"{spec.name} validation end changed.")
    validation_split = materialize_backtest_split(history, official_grid, outer_fold)
    if validation_split.scored_rows != spec.expected_scored_rows:
        raise AssertionError(f"{spec.name} scored-row count changed.")
    coverage = validation_split.scored_rows / validation_split.requested_rows
    if not np.isclose(coverage, spec.expected_coverage, rtol=0, atol=1e-15):
        raise AssertionError(f"{spec.name} coverage changed.")
    validation_batch = build_stage3_feature_batch(
        validation_split.training_history,
        validation_split.validation_features,
        inventory,
        calendar,
        spec.cutoff,
    )
    if tuple(validation_batch.matrix.columns) != APPROVED_FEATURES:
        raise AssertionError(f"{spec.name} validation features differ from the contract.")
    validation_labels = validation_split.validation_labels.copy()
    del validation_split
    gc.collect()
    return (
        training_features,
        training_target,
        validation_batch,
        validation_labels,
        origins_audit,
    )


def run_fold(
    spec: FoldSpec,
    history: pd.DataFrame,
    inventory: pd.DataFrame,
    calendar: pd.DataFrame,
    official_grid: pd.DataFrame,
    config: PlainLightGBMConfig,
) -> dict[str, object]:
    """Build, fit, predict, and score one approved outer fold."""

    fold_start = perf_counter()
    feature_start = perf_counter()
    (
        training_features,
        training_target,
        validation_batch,
        validation_labels,
        origins_audit,
    ) = _build_fold_frames(spec, history, inventory, calendar, official_grid)
    feature_seconds = perf_counter() - feature_start

    fit_start = perf_counter()
    model = train_plain_lightgbm(
        training_features,
        training_target,
        CATEGORICAL_FEATURES,
        config,
    )
    fit_seconds = perf_counter() - fit_start

    score_start = perf_counter()
    prediction = predict_plain_lightgbm(model, validation_batch.matrix)
    prediction_frame = validation_batch.keys.copy()
    prediction_frame["sales_hat"] = prediction
    if prediction_frame.loc[:, ["unique_id", "date"]].duplicated().any():
        raise AssertionError(f"{spec.name} predictions contain duplicate keys.")
    metrics = score_kaggle_aligned(validation_labels, prediction_frame)
    score_seconds = perf_counter() - score_start

    absolute_improvement = spec.baseline_wmae - metrics.wmae
    relative_improvement = absolute_improvement / spec.baseline_wmae
    return {
        "fold": spec.name,
        "cutoff": str(spec.cutoff.date()),
        "validation_start": str(spec.validation_start.date()),
        "validation_end": str(spec.validation_end.date()),
        "training_origins": origins_audit,
        "training_rows": len(training_features),
        "validation_rows": metrics.rows,
        "requested_rows": REQUESTED_GRID_ROWS,
        "coverage": metrics.rows / REQUESTED_GRID_ROWS,
        "metrics": {
            "wmae": metrics.wmae,
            "wape": metrics.wape,
            "bias": metrics.bias,
        },
        "stage2_trailing14_baseline": {
            "wmae": spec.baseline_wmae,
            "wape": spec.baseline_wape,
            "bias": spec.baseline_bias,
        },
        "comparison": {
            "wmae_absolute_improvement": absolute_improvement,
            "wmae_relative_improvement": relative_improvement,
            "beats_baseline_wmae": bool(metrics.wmae < spec.baseline_wmae),
            "suspicious_improvement": suspicious_improvement(
                spec.baseline_wmae, metrics.wmae
            ),
        },
        "runtime": {
            "feature_generation_seconds": feature_seconds,
            "model_fit_seconds": fit_seconds,
            "prediction_and_scoring_seconds": score_seconds,
            "total_seconds": perf_counter() - fold_start,
        },
        "peak_rss_mib": peak_rss_mib(),
        "negative_predictions": int((prediction < 0).sum()),
        "prediction_minimum": float(prediction.min()),
        "prediction_maximum": float(prediction.max()),
    }


def _aggregate(fold_results: list[dict[str, object]]) -> dict[str, float]:
    rows = [COMMITTED_F1, *fold_results]

    def metric(row: dict[str, object], name: str) -> float:
        if name in row:
            return float(row[name])
        return float(row["metrics"][name])

    return {
        "mean_wmae": float(np.mean([metric(row, "wmae") for row in rows])),
        "mean_wape": float(np.mean([metric(row, "wape") for row in rows])),
        "mean_bias": float(np.mean([metric(row, "bias") for row in rows])),
        "mean_stage2_baseline_wmae": float(
            np.mean(
                [
                    COMMITTED_F1["baseline_wmae"],
                    *[row["stage2_trailing14_baseline"]["wmae"] for row in fold_results],
                ]
            )
        ),
    }


def main() -> None:
    total_start = perf_counter()
    history, inventory, calendar, official_grid = _load_raw_data()
    raw_load_seconds = perf_counter() - total_start
    if {"total_orders", "availability"}.intersection(history.columns):
        raise AssertionError("Forbidden future fields were loaded.")

    config = PlainLightGBMConfig()
    results: list[dict[str, object]] = []
    for spec in FOLD_SPECS:
        result = run_fold(spec, history, inventory, calendar, official_grid, config)
        results.append(result)
        gc.collect()

    output = {
        "stage": "stage_3_plain_model_all_folds",
        "folds_run": [result["fold"] for result in results],
        "f1_reference": COMMITTED_F1,
        "model_configuration": config.audit_dict(),
        "feature_count": len(APPROVED_FEATURES),
        "features": list(APPROVED_FEATURES),
        "results": results,
        "aggregate_f1_f4": _aggregate(results),
        "shared": {
            "raw_load_seconds": raw_load_seconds,
            "f2_f4_total_runtime_seconds": perf_counter() - total_start,
            "peak_rss_mib": peak_rss_mib(),
        },
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
