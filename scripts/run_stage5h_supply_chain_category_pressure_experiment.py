#!/usr/bin/env python3
"""Run the Stage 5-H supply-chain category-pressure experiment on F1-F4."""

from __future__ import annotations

import csv
import gc
import json
import resource
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import lightgbm as lgb
import numpy as np
import pandas as pd

from eval.backtest import make_backtest_folds, materialize_backtest_split
from eval.grid import score_kaggle_aligned
from features.stage3_minimal import CATEGORICAL_FEATURES, FORBIDDEN_FEATURE_FIELDS
from features.stage5h_supply_chain_category_pressure import (
    APPROVED_STAGE5H_FEATURES,
    FEATURE_AVAILABILITY as STAGE5H_FEATURE_AVAILABILITY,
    STAGE5H_EXTRA_FEATURES,
    STAGE5H_FORBIDDEN_FEATURE_FIELDS,
    build_stage5h_feature_batch,
)
from models.plain_lgbm import PlainLightGBMConfig, predict_plain_lightgbm, train_plain_lightgbm
from scripts.run_stage3_all_folds_plain_model import training_origins_for_fold
from scripts.run_stage3_f1_plain_model import _aligned_target
from scripts.run_stage5e_kaggle_candidate import _load_raw_data
from scripts.run_stage5e_kaggle_candidate import _apply_zero_clipping
from scripts.run_stage5f_objective_blend_experiments import FOLD_SPECS


ROOT = Path(__file__).resolve().parents[1]
RESULTS_CSV = ROOT / "results.csv"
REPORT_PATH = ROOT / "reports" / "stage5h_supply_chain_category_pressure_results.md"
STAGE5E_BASE_FEATURE_COUNT = 76
STAGE5G_FIXED_BLEND_LOCAL_MEAN_WMAE = 19.0089170845
STAGE5E_RAW_LOCAL_MEAN_WMAE = 19.5424424315
RAW_MODEL_CONFIG = PlainLightGBMConfig()
TWEEDIE_MODEL_CONFIG = PlainLightGBMConfig(objective="tweedie")
TWEEDIE_VARIANCE_POWER = 1.1
FIXED_BLEND_WEIGHTS = (0.5, 0.5)


@dataclass(frozen=True)
class VariantSpec:
    name: str
    stage: str
    change_description: str
    objective: str


VARIANT_SPECS = (
    VariantSpec(
        name="raw_l1",
        stage="stage_5h_supply_chain_category_pressure_raw_l1_local",
        change_description="S5-H raw L1 with warehouse-category pressure features",
        objective="regression_l1",
    ),
    VariantSpec(
        name="tweedie_1_1",
        stage="stage_5h_supply_chain_category_pressure_tweedie_1_1_local",
        change_description="S5-H Tweedie 1.1 with warehouse-category pressure features",
        objective="tweedie",
    ),
)


def peak_rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _git_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _append_results_rows(rows: list[dict[str, object]]) -> None:
    with RESULTS_CSV.open("a", newline="") as handle:
        writer = csv.writer(handle)
        for row in rows:
            writer.writerow(
                [
                    row["timestamp"],
                    row["git_hash"],
                    row["stage"],
                    row["change_description"],
                    f"{row['local_wmae']:.10f}",
                    f"{row['local_wape']:.10f}",
                    f"{row['local_bias']:.10f}",
                    f"{row['runtime_minutes']:.10f}",
                    row["kept"],
                    row["notes"],
                ]
            )


def _score_frame(labels: pd.DataFrame, keys: pd.DataFrame, prediction: np.ndarray) -> dict[str, float]:
    frame = keys.copy()
    frame["sales_hat"] = prediction
    metrics = score_kaggle_aligned(labels, frame)
    return {"wmae": metrics.wmae, "wape": metrics.wape, "bias": metrics.bias}


def _summary(prediction: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(prediction, dtype=np.float64)
    return {
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "negative_count": int((array < 0).sum()),
        "null_count": int(np.isnan(array).sum()),
    }


def _train_tweedie_lightgbm(
    features: pd.DataFrame,
    target: pd.Series,
    categorical_features: tuple[str, ...],
    config: PlainLightGBMConfig,
) -> lgb.Booster:
    target_array = np.asarray(target, dtype=np.float64)
    if len(features) != len(target_array):
        raise ValueError("Feature and target row counts differ.")
    if not np.isfinite(target_array).all():
        raise ValueError("Training target contains missing or non-finite values.")
    missing_categories = set(categorical_features) - set(features.columns)
    if missing_categories:
        raise KeyError(f"Missing categorical features: {sorted(missing_categories)}.")

    params = dict(config.parameters())
    params["objective"] = config.objective
    params["tweedie_variance_power"] = TWEEDIE_VARIANCE_POWER
    dataset = lgb.Dataset(
        features,
        label=target_array,
        categorical_feature=list(categorical_features),
        free_raw_data=False,
    )
    return lgb.train(params, dataset, num_boost_round=config.num_boost_round)


def _blend_predictions(
    raw_prediction_frame: pd.DataFrame,
    tweedie_prediction_frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    left_keys = raw_prediction_frame.loc[:, ["unique_id", "date"]].reset_index(drop=True)
    right_keys = tweedie_prediction_frame.loc[:, ["unique_id", "date"]].reset_index(drop=True)
    if not left_keys.equals(right_keys):
        raise AssertionError("Fixed blend predictions must align exactly by id and order.")
    if tuple(FIXED_BLEND_WEIGHTS) != (0.5, 0.5):
        raise AssertionError("Fixed blend weights changed.")
    if not np.isclose(sum(FIXED_BLEND_WEIGHTS), 1.0, atol=1e-12):
        raise AssertionError("Fixed blend weights must sum to 1.")

    raw_values = raw_prediction_frame["sales_hat"].to_numpy(dtype=np.float64)
    tweedie_values = tweedie_prediction_frame["sales_hat"].to_numpy(dtype=np.float64)
    blended = FIXED_BLEND_WEIGHTS[0] * raw_values + FIXED_BLEND_WEIGHTS[1] * tweedie_values
    clipped, clipped_rows = _apply_zero_clipping(blended)
    audit = {
        "weights": list(FIXED_BLEND_WEIGHTS),
        "negative_before_clip": int((blended < 0).sum()),
        "clipped_rows": int(clipped_rows),
    }
    return blended, clipped, audit


def _variant_fold_frame(
    spec: dict[str, object],
    history: pd.DataFrame,
    inventory: pd.DataFrame,
    calendar: pd.DataFrame,
    official_grid: pd.DataFrame,
) -> dict[str, object]:
    training_matrices: list[pd.DataFrame] = []
    training_targets: list[pd.Series] = []
    origin_audit: list[dict[str, object]] = []

    for origin in training_origins_for_fold(spec["cutoff"]):
        fold = make_backtest_folds([origin])[0]
        split = materialize_backtest_split(history, official_grid, fold)
        if split.validation_labels["date"].max() > spec["cutoff"]:
            raise AssertionError(f"{spec['name']} training labels cross the fold cutoff.")

        batch = build_stage5h_feature_batch(
            split.training_history,
            split.validation_features,
            inventory,
            calendar,
            origin,
        )
        if tuple(batch.matrix.columns) != APPROVED_STAGE5H_FEATURES:
            raise AssertionError("Stage 5-H feature contract changed.")
        if FORBIDDEN_FEATURE_FIELDS.intersection(batch.matrix.columns):
            raise AssertionError("Stage 5-H training features contain a forbidden field.")
        if STAGE5H_FORBIDDEN_FEATURE_FIELDS.intersection(batch.matrix.columns):
            raise AssertionError("Stage 5-H training features contain a Stage 5 forbidden field.")
        target = _aligned_target(batch.keys, split.validation_labels)
        training_matrices.append(batch.matrix)
        training_targets.append(target)
        origin_audit.append(
            {
                "origin": str(pd.Timestamp(origin).date()),
                "target_start": str(fold.validation_start.date()),
                "target_end": str(fold.validation_end.date()),
                "training_rows": len(batch.matrix),
                "maximum_history_feature_date": str(batch.maximum_history_date.date()),
            }
        )
        del split, batch, target
        gc.collect()

    training_features = pd.concat(training_matrices, ignore_index=True)
    training_target = pd.concat(training_targets, ignore_index=True)
    if tuple(training_features.columns) != APPROVED_STAGE5H_FEATURES:
        raise AssertionError("Combined Stage 5-H training features differ from the contract.")

    outer_fold = make_backtest_folds([spec["cutoff"]])[0]
    validation_split = materialize_backtest_split(history, official_grid, outer_fold)
    if validation_split.scored_rows != spec["expected_scored_rows"]:
        raise AssertionError(f"{spec['name']} scored-row count changed.")
    coverage = validation_split.scored_rows / validation_split.requested_rows
    if not np.isclose(coverage, spec["expected_coverage"], rtol=0, atol=1e-15):
        raise AssertionError(f"{spec['name']} coverage changed.")

    validation_batch = build_stage5h_feature_batch(
        validation_split.training_history,
        validation_split.validation_features,
        inventory,
        calendar,
        spec["cutoff"],
    )
    if tuple(validation_batch.matrix.columns) != APPROVED_STAGE5H_FEATURES:
        raise AssertionError("Validation Stage 5-H feature contract changed.")
    if FORBIDDEN_FEATURE_FIELDS.intersection(validation_batch.matrix.columns):
        raise AssertionError("Stage 5-H validation features contain a forbidden field.")
    if STAGE5H_FORBIDDEN_FEATURE_FIELDS.intersection(validation_batch.matrix.columns):
        raise AssertionError("Stage 5-H validation features contain a Stage 5 forbidden field.")

    return {
        "fold": spec["name"],
        "cutoff": str(spec["cutoff"].date()),
        "validation_start": str(spec["validation_start"].date()),
        "validation_end": str(spec["validation_end"].date()),
        "training_origins": origin_audit,
        "training_features": training_features,
        "training_target": training_target,
        "validation_batch": validation_batch,
        "validation_labels": validation_split.validation_labels.copy(),
        "validation_rows": len(validation_batch.matrix),
        "validation_scored_rows": validation_split.scored_rows,
        "coverage": coverage,
        "reference": {"wmae": STAGE5G_FIXED_BLEND_LOCAL_MEAN_WMAE},
    }


def _run_single_variant(
    fold_ctx: dict[str, object],
    variant: VariantSpec,
    config: PlainLightGBMConfig,
) -> dict[str, object]:
    training_features = fold_ctx["training_features"]
    training_target = fold_ctx["training_target"]
    validation_batch = fold_ctx["validation_batch"]
    validation_labels = fold_ctx["validation_labels"]

    run_start = perf_counter()
    fit_start = perf_counter()
    if variant.objective == "tweedie":
        model = _train_tweedie_lightgbm(training_features, training_target, CATEGORICAL_FEATURES, config)
    else:
        model = train_plain_lightgbm(training_features, training_target, CATEGORICAL_FEATURES, config)
    fit_seconds = perf_counter() - fit_start

    predict_start = perf_counter()
    prediction_raw = predict_plain_lightgbm(model, validation_batch.matrix)
    prediction_clipped, clipped_rows = _apply_zero_clipping(prediction_raw)
    prediction_seconds = perf_counter() - predict_start
    raw_metrics = _score_frame(validation_labels, validation_batch.keys, prediction_raw)
    clipped_metrics = _score_frame(validation_labels, validation_batch.keys, prediction_clipped)
    total_seconds = perf_counter() - run_start

    return {
        "variant": variant,
        "fold": fold_ctx["fold"],
        "raw_metrics": raw_metrics,
        "clipped_metrics": clipped_metrics,
        "prediction_raw": prediction_raw,
        "prediction_clipped": prediction_clipped,
        "negative_before_clip": int((prediction_raw < 0).sum()),
        "clipped_rows": clipped_rows,
        "fit_seconds": fit_seconds,
        "prediction_seconds": prediction_seconds,
        "runtime_seconds": total_seconds,
        "peak_rss_mib": peak_rss_mib(),
        "beats_stage5g_fold": raw_metrics["wmae"] < fold_ctx["reference"]["wmae"],
    }


def _aggregate(rows: list[dict[str, object]], *, metric_key: str = "raw_metrics") -> dict[str, float]:
    return {
        "mean_wmae": float(np.mean([row[metric_key]["wmae"] for row in rows])),
        "mean_wape": float(np.mean([row[metric_key]["wape"] for row in rows])),
        "mean_bias": float(np.mean([row[metric_key]["bias"] for row in rows])),
        "mean_runtime_seconds": float(np.mean([row["runtime_seconds"] for row in rows])),
        "total_runtime_seconds": float(np.sum([row["runtime_seconds"] for row in rows])),
        "peak_rss_mib": float(max(row["peak_rss_mib"] for row in rows)),
    }


def run() -> dict[str, object]:
    start = perf_counter()
    history, inventory, calendar, _sales_test, official_grid, _solution = _load_raw_data()
    raw_load_seconds = perf_counter() - start

    fold_contexts = [
        _variant_fold_frame(spec, history, inventory, calendar, official_grid) for spec in FOLD_SPECS
    ]

    results_by_variant: dict[str, list[dict[str, object]]] = {spec.name: [] for spec in VARIANT_SPECS}
    fold_results: list[dict[str, object]] = []

    for fold_ctx in fold_contexts:
        raw_variant = _run_single_variant(fold_ctx, VARIANT_SPECS[0], RAW_MODEL_CONFIG)
        tweedie_variant = _run_single_variant(fold_ctx, VARIANT_SPECS[1], TWEEDIE_MODEL_CONFIG)
        blend_start = perf_counter()
        blended_raw, blended_clipped, blend_audit = _blend_predictions(
            fold_ctx["validation_batch"].keys.assign(sales_hat=raw_variant["prediction_raw"]),
            fold_ctx["validation_batch"].keys.assign(sales_hat=tweedie_variant["prediction_raw"]),
        )
        blend_seconds = perf_counter() - blend_start
        blend_raw_metrics = _score_frame(
            fold_ctx["validation_labels"],
            fold_ctx["validation_batch"].keys,
            blended_raw,
        )
        blend_clipped_metrics = _score_frame(
            fold_ctx["validation_labels"],
            fold_ctx["validation_batch"].keys,
            blended_clipped,
        )
        blend_total_seconds = raw_variant["runtime_seconds"] + tweedie_variant["runtime_seconds"] + blend_seconds
        blend_peak_rss = max(
            raw_variant["peak_rss_mib"],
            tweedie_variant["peak_rss_mib"],
            peak_rss_mib(),
        )

        fold_results.append(
            {
                "fold": fold_ctx["fold"],
                "raw_l1": raw_variant,
                "tweedie_1_1": tweedie_variant,
                "fixed_blend": {
                    "raw_metrics": blend_raw_metrics,
                    "clipped_metrics": blend_clipped_metrics,
                    "negative_before_clip": int((blended_raw < 0).sum()),
                    "clipped_rows": blend_audit["clipped_rows"],
                    "prediction_raw": blended_raw,
                    "prediction_clipped": blended_clipped,
                    "beats_stage5g_fold": blend_raw_metrics["wmae"] < STAGE5G_FIXED_BLEND_LOCAL_MEAN_WMAE,
                    "runtime_seconds": blend_total_seconds,
                    "peak_rss_mib": blend_peak_rss,
                },
                "validation_rows": fold_ctx["validation_rows"],
                "validation_scored_rows": fold_ctx["validation_scored_rows"],
                "coverage": fold_ctx["coverage"],
                "training_origins": fold_ctx["training_origins"],
            }
        )

        results_by_variant["raw_l1"].append(raw_variant)
        results_by_variant["tweedie_1_1"].append(tweedie_variant)

        fold_results[-1]["fixed_blend"]["validation_rows"] = fold_ctx["validation_rows"]
        fold_results[-1]["fixed_blend"]["validation_scored_rows"] = fold_ctx["validation_scored_rows"]

    raw_agg = _aggregate(results_by_variant["raw_l1"])
    tweedie_agg = _aggregate(results_by_variant["tweedie_1_1"])
    blend_rows = [row["fixed_blend"] for row in fold_results]
    blend_agg = _aggregate(blend_rows)

    overall_runtime_seconds = perf_counter() - start

    fold_metrics = {
        "raw_l1": [
            {
                "fold": row["fold"],
                "wmae": row["raw_l1"]["raw_metrics"]["wmae"],
                "wape": row["raw_l1"]["raw_metrics"]["wape"],
                "bias": row["raw_l1"]["raw_metrics"]["bias"],
                "clipped_wmae": row["raw_l1"]["clipped_metrics"]["wmae"],
                "clipped_wape": row["raw_l1"]["clipped_metrics"]["wape"],
                "clipped_bias": row["raw_l1"]["clipped_metrics"]["bias"],
                "negative_before_clip": row["raw_l1"]["negative_before_clip"],
                "clipped_rows": row["raw_l1"]["clipped_rows"],
                "beats_stage5g_fold": row["raw_l1"]["beats_stage5g_fold"],
            }
            for row in fold_results
        ],
        "tweedie_1_1": [
            {
                "fold": row["fold"],
                "wmae": row["tweedie_1_1"]["raw_metrics"]["wmae"],
                "wape": row["tweedie_1_1"]["raw_metrics"]["wape"],
                "bias": row["tweedie_1_1"]["raw_metrics"]["bias"],
                "clipped_wmae": row["tweedie_1_1"]["clipped_metrics"]["wmae"],
                "clipped_wape": row["tweedie_1_1"]["clipped_metrics"]["wape"],
                "clipped_bias": row["tweedie_1_1"]["clipped_metrics"]["bias"],
                "negative_before_clip": row["tweedie_1_1"]["negative_before_clip"],
                "clipped_rows": row["tweedie_1_1"]["clipped_rows"],
                "beats_stage5g_fold": row["tweedie_1_1"]["beats_stage5g_fold"],
            }
            for row in fold_results
        ],
        "fixed_blend": [
            {
                "fold": row["fold"],
                "wmae": row["fixed_blend"]["raw_metrics"]["wmae"],
                "wape": row["fixed_blend"]["raw_metrics"]["wape"],
                "bias": row["fixed_blend"]["raw_metrics"]["bias"],
                "clipped_wmae": row["fixed_blend"]["clipped_metrics"]["wmae"],
                "clipped_wape": row["fixed_blend"]["clipped_metrics"]["wape"],
                "clipped_bias": row["fixed_blend"]["clipped_metrics"]["bias"],
                "negative_before_clip": row["fixed_blend"]["negative_before_clip"],
                "clipped_rows": row["fixed_blend"]["clipped_rows"],
                "beats_stage5g_fold": row["fixed_blend"]["beats_stage5g_fold"],
            }
            for row in fold_results
        ],
    }

    support_status = "hypothesis_rejected_for_current_model"
    if blend_agg["mean_wmae"] < STAGE5G_FIXED_BLEND_LOCAL_MEAN_WMAE and sum(
        row["fixed_blend"]["beats_stage5g_fold"] for row in fold_results
    ) >= 3:
        support_status = "hypothesis_supported_candidate_possible"
    elif blend_agg["mean_wmae"] < STAGE5G_FIXED_BLEND_LOCAL_MEAN_WMAE:
        support_status = "inconclusive"

    stage_rows = [
        {
            "timestamp": _timestamp(),
            "git_hash": _git_hash(),
            "stage": VARIANT_SPECS[0].stage,
            "change_description": VARIANT_SPECS[0].change_description,
            "local_wmae": raw_agg["mean_wmae"],
            "local_wape": raw_agg["mean_wape"],
            "local_bias": raw_agg["mean_bias"],
            "runtime_minutes": raw_agg["total_runtime_seconds"] / 60,
            "kept": "diagnostic_ablation",
            "notes": (
                f"feature_count={len(APPROVED_STAGE5H_FEATURES)}; base_feature_count={STAGE5E_BASE_FEATURE_COUNT}; "
                f"extra_feature_count={len(STAGE5H_EXTRA_FEATURES)}; folds=F1,F2,F3,F4; "
                f"mean_wmae={raw_agg['mean_wmae']:.10f}; mean_wape={raw_agg['mean_wape']:.10f}; "
                f"mean_bias={raw_agg['mean_bias']:.10f}; support_status={support_status}; no Kaggle submission"
            ),
        },
        {
            "timestamp": _timestamp(),
            "git_hash": _git_hash(),
            "stage": VARIANT_SPECS[1].stage,
            "change_description": VARIANT_SPECS[1].change_description,
            "local_wmae": tweedie_agg["mean_wmae"],
            "local_wape": tweedie_agg["mean_wape"],
            "local_bias": tweedie_agg["mean_bias"],
            "runtime_minutes": tweedie_agg["total_runtime_seconds"] / 60,
            "kept": "diagnostic_ablation",
            "notes": (
                f"feature_count={len(APPROVED_STAGE5H_FEATURES)}; base_feature_count={STAGE5E_BASE_FEATURE_COUNT}; "
                f"extra_feature_count={len(STAGE5H_EXTRA_FEATURES)}; tweedie_variance_power={TWEEDIE_VARIANCE_POWER}; "
                f"folds=F1,F2,F3,F4; mean_wmae={tweedie_agg['mean_wmae']:.10f}; mean_wape={tweedie_agg['mean_wape']:.10f}; "
                f"mean_bias={tweedie_agg['mean_bias']:.10f}; support_status={support_status}; no Kaggle submission"
            ),
        },
        {
            "timestamp": _timestamp(),
            "git_hash": _git_hash(),
            "stage": "stage_5h_supply_chain_category_pressure_fixed_blend_local",
            "change_description": "S5-H fixed 50/50 raw L1 + Tweedie blend with warehouse-category pressure features",
            "local_wmae": blend_agg["mean_wmae"],
            "local_wape": blend_agg["mean_wape"],
            "local_bias": blend_agg["mean_bias"],
            "runtime_minutes": blend_agg["total_runtime_seconds"] / 60,
            "kept": "diagnostic_ablation",
            "notes": (
                f"feature_count={len(APPROVED_STAGE5H_FEATURES)}; base_feature_count={STAGE5E_BASE_FEATURE_COUNT}; "
                f"extra_feature_count={len(STAGE5H_EXTRA_FEATURES)}; blend_weights=[0.5, 0.5]; "
                f"fixed_rule_applicable_to_official_test_set=True; "
                f"fold_wmae={[round(row['wmae'], 10) for row in fold_metrics['fixed_blend']]}; "
                f"mean_wmae={blend_agg['mean_wmae']:.10f}; mean_wape={blend_agg['mean_wape']:.10f}; "
                f"mean_bias={blend_agg['mean_bias']:.10f}; clipped_rows_total={int(sum(row['clipped_rows'] for row in blend_rows))}; "
                f"support_status={support_status}; no Kaggle submission"
            ),
        },
    ]
    _append_results_rows(stage_rows)

    report_lines = [
        "# Stage 5-H Supply-Chain Category Pressure Results",
        "",
        "## Purpose",
        "",
        "Test whether a supply-chain hypothesis adds incremental signal beyond the current Stage 5-G benchmark recipe.",
        "",
        "## Business hypothesis",
        "",
        "SKU demand should not be forecast in isolation. It should be informed by warehouse-category demand pressure, category mean reversion, and the SKU’s changing share within its warehouse/category.",
        "",
        "## Implemented features",
        "",
        f"- Base contract: Stage 5-E approved feature count = {STAGE5E_BASE_FEATURE_COUNT}",
        f"- Added Stage 5-H feature count: {len(STAGE5H_EXTRA_FEATURES)}",
        f"- Total feature count: {len(APPROVED_STAGE5H_FEATURES)}",
        "- Warehouse-category demand pressure features at 7, 14, and 28 days",
        "- Item share within warehouse/category at 7 and 28 days",
        "- Simple interactions with horizon, discount, and relative price",
        "",
        "## Skipped features and why",
        "",
        "- No fold-specific category selection was used; membership is fixed by warehouse and L2 category metadata.",
        "- No broader tuning sweep was run beyond the approved feature list.",
        "- No future sales, future `total_orders`, or future availability were used.",
        "- No horizon-specific routing was introduced.",
        "",
        "## Cutoff-safety design",
        "",
        "- All category totals were computed only from history dated on or before each origin.",
        "- Validation labels were not used in any category totals.",
        "- Zero or missing denominators used safe fallbacks.",
        "- The official-test-style diagnostics remain benchmark-specific and use only data available through the benchmark cutoff.",
        "",
        "## F1-F4 fold results",
        "",
        "| Fold | Variant | WMAE | WAPE | Bias | Clipped WMAE | Clipped WAPE | Clipped Bias | Negative before clip | Clipped rows | Beats Stage 5-G? |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in fold_results:
        for variant_name in ("raw_l1", "tweedie_1_1"):
            metrics = row[variant_name]
            report_lines.append(
                f"| {row['fold']} | {variant_name} | {metrics['raw_metrics']['wmae']:.10f} | {metrics['raw_metrics']['wape']:.10f} | {metrics['raw_metrics']['bias']:.10f} | "
                f"{metrics['clipped_metrics']['wmae']:.10f} | {metrics['clipped_metrics']['wape']:.10f} | {metrics['clipped_metrics']['bias']:.10f} | "
                f"{metrics['negative_before_clip']} | {metrics['clipped_rows']} | "
                f"{'Yes' if metrics['raw_metrics']['wmae'] < STAGE5G_FIXED_BLEND_LOCAL_MEAN_WMAE else 'No'} |"
            )
        blend = row["fixed_blend"]
        report_lines.append(
            f"| {row['fold']} | fixed_blend | {blend['raw_metrics']['wmae']:.10f} | {blend['raw_metrics']['wape']:.10f} | {blend['raw_metrics']['bias']:.10f} | "
            f"{blend['clipped_metrics']['wmae']:.10f} | {blend['clipped_metrics']['wape']:.10f} | {blend['clipped_metrics']['bias']:.10f} | "
            f"{blend['negative_before_clip']} | {blend['clipped_rows']} | "
            f"{'Yes' if blend['raw_metrics']['wmae'] < STAGE5G_FIXED_BLEND_LOCAL_MEAN_WMAE else 'No'} |"
        )
    report_lines.extend(
        [
            "",
            "## Aggregate metrics",
            "",
            "| Variant | Mean WMAE | Mean WAPE | Mean bias | Runtime (min) | Peak RSS (MiB) | Beats Stage 5-G mean? |",
            "|---|---:|---:|---:|---:|---:|---|",
            f"| raw_l1 | {raw_agg['mean_wmae']:.10f} | {raw_agg['mean_wape']:.10f} | {raw_agg['mean_bias']:.10f} | {raw_agg['total_runtime_seconds'] / 60:.3f} | {raw_agg['peak_rss_mib']:.2f} | {'Yes' if raw_agg['mean_wmae'] < STAGE5G_FIXED_BLEND_LOCAL_MEAN_WMAE else 'No'} |",
            f"| tweedie_1_1 | {tweedie_agg['mean_wmae']:.10f} | {tweedie_agg['mean_wape']:.10f} | {tweedie_agg['mean_bias']:.10f} | {tweedie_agg['total_runtime_seconds'] / 60:.3f} | {tweedie_agg['peak_rss_mib']:.2f} | {'Yes' if tweedie_agg['mean_wmae'] < STAGE5G_FIXED_BLEND_LOCAL_MEAN_WMAE else 'No'} |",
            f"| fixed_blend | {blend_agg['mean_wmae']:.10f} | {blend_agg['mean_wape']:.10f} | {blend_agg['mean_bias']:.10f} | {blend_agg['total_runtime_seconds'] / 60:.3f} | {blend_agg['peak_rss_mib']:.2f} | {'Yes' if blend_agg['mean_wmae'] < STAGE5G_FIXED_BLEND_LOCAL_MEAN_WMAE else 'No'} |",
            "",
            "## Comparison to prior benchmark references",
            "",
            f"- Stage 5-E raw L1 local mean WMAE: {STAGE5E_RAW_LOCAL_MEAN_WMAE}",
            f"- Stage 5-G fixed 50/50 local mean WMAE: {STAGE5G_FIXED_BLEND_LOCAL_MEAN_WMAE}",
            f"- Stage 5-G official private WMAE: 20.14904",
            "",
            f"## Decision: {support_status}",
            "",
        ]
    )
    if support_status == "hypothesis_supported_candidate_possible":
        report_lines.append(
            "The fixed blend improved mean WMAE over Stage 5-G and improved most folds, so a candidate could be justified later."
        )
    elif support_status == "inconclusive":
        report_lines.append(
            "The fixed blend improved mean WMAE over Stage 5-G, but the fold pattern was not strong enough to recommend promotion yet."
        )
    else:
        report_lines.append(
            "The fixed blend did not materially improve mean WMAE over Stage 5-G, so the hypothesis is rejected for the current model."
        )
    report_lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- This remains a benchmark setting.",
            "- Price and discount still depend on benchmark-specific known-future covariates.",
            "- Category logic may be dataset-specific.",
            "- Local validation may not transfer exactly to Kaggle hidden scoring.",
            "",
        ]
    )

    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    return {
        "raw_load_seconds": raw_load_seconds,
        "fold_results": fold_results,
        "aggregate": {
            "raw_l1": raw_agg,
            "tweedie_1_1": tweedie_agg,
            "fixed_blend": blend_agg,
        },
        "feature_count": len(APPROVED_STAGE5H_FEATURES),
        "extra_feature_count": len(STAGE5H_EXTRA_FEATURES),
        "support_status": support_status,
        "report_path": str(REPORT_PATH),
    }


def main() -> None:
    result = run()
    print(
        json.dumps(
            {
                "support_status": result["support_status"],
                "feature_count": result["feature_count"],
                "extra_feature_count": result["extra_feature_count"],
                "aggregate": {
                    key: {
                        "mean_wmae": value["mean_wmae"],
                        "mean_wape": value["mean_wape"],
                        "mean_bias": value["mean_bias"],
                    }
                    for key, value in result["aggregate"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
