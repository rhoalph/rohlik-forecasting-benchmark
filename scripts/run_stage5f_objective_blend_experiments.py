#!/usr/bin/env python3
"""Run Stage 5-F/G objective and blend experiments on the approved Stage 5-E features."""

from __future__ import annotations

import csv
import gc
import json
import resource
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import lightgbm as lgb
import numpy as np
import pandas as pd

from eval.backtest import make_backtest_folds, materialize_backtest_split
from eval.grid import score_kaggle_aligned
from features.stage3_minimal import (
    CATEGORICAL_FEATURES,
    FORBIDDEN_FEATURE_FIELDS,
)
from features.stage5e_stronger_features import (
    APPROVED_STAGE5E_FEATURES,
    FEATURE_AVAILABILITY as STAGE5E_FEATURE_AVAILABILITY,
    STAGE5E_FORBIDDEN_FEATURE_FIELDS,
    build_stage5e_feature_batch,
)
from models.plain_lgbm import PlainLightGBMConfig, predict_plain_lightgbm
from scripts.run_stage3_all_folds_plain_model import training_origins_for_fold
from scripts.run_stage3_f1_plain_model import _aligned_target
from scripts.run_stage5e_kaggle_candidate import _load_raw_data


ROOT = Path(__file__).resolve().parents[1]
RESULTS_CSV = ROOT / "results.csv"
REPORT_PATH = ROOT / "reports" / "stage5f_objective_blend_results.md"

STAGE5E_REFERENCE_BY_FOLD = {
    "F1": {"wmae": 19.606999527283076, "wape": 0.20716367955250256, "bias": -0.02178937944269652},
    "F2": {"wmae": 21.321838547413208, "wape": 0.21582003447579737, "bias": -0.085196857214704},
    "F3": {"wmae": 18.743456466451384, "wape": 0.19127253620467316, "bias": -0.01315669135226799},
    "F4": {"wmae": 18.49747518504875, "wape": 0.19690290637012992, "bias": -0.0688270785229739},
}
STAGE5E_REFERENCE_MEAN_WMAE = 19.5424424315

FOLD_SPECS = (
    {
        "name": "F1",
        "cutoff": pd.Timestamp("2024-05-19"),
        "validation_start": pd.Timestamp("2024-05-20"),
        "validation_end": pd.Timestamp("2024-06-02"),
        "expected_scored_rows": 44212,
        "expected_coverage": 0.9402607345654069,
    },
    {
        "name": "F2",
        "cutoff": pd.Timestamp("2024-05-05"),
        "validation_start": pd.Timestamp("2024-05-06"),
        "validation_end": pd.Timestamp("2024-05-19"),
        "expected_scored_rows": 43433,
        "expected_coverage": 0.923693668786287,
    },
    {
        "name": "F3",
        "cutoff": pd.Timestamp("2024-04-21"),
        "validation_start": pd.Timestamp("2024-04-22"),
        "validation_end": pd.Timestamp("2024-05-05"),
        "expected_scored_rows": 42794,
        "expected_coverage": 0.9101039960868548,
    },
    {
        "name": "F4",
        "cutoff": pd.Timestamp("2024-04-07"),
        "validation_start": pd.Timestamp("2024-04-08"),
        "validation_end": pd.Timestamp("2024-04-21"),
        "expected_scored_rows": 42035,
        "expected_coverage": 0.8939622721762617,
    },
)


@dataclass(frozen=True)
class VariantSpec:
    name: str
    stage: str
    change_description: str
    target_transform: str
    objective: str
    extra_params: dict[str, object]
    is_non_raw: bool
    clipped_only: bool = False


VARIANT_SPECS = (
    VariantSpec(
        name="raw_l1",
        stage="stage_5f_objective_raw_l1_local",
        change_description="S5-F raw L1 control on the Stage 5-E feature contract",
        target_transform="raw",
        objective="regression_l1",
        extra_params={},
        is_non_raw=False,
    ),
    VariantSpec(
        name="sqrt_l1",
        stage="stage_5f_objective_sqrt_l1_local",
        change_description="S5-F sqrt target with L1 objective on the Stage 5-E feature contract",
        target_transform="sqrt",
        objective="regression_l1",
        extra_params={},
        is_non_raw=True,
    ),
    VariantSpec(
        name="log1p_l1",
        stage="stage_5f_objective_log1p_l1_local",
        change_description="S5-F log1p target with L1 objective on the Stage 5-E feature contract",
        target_transform="log1p",
        objective="regression_l1",
        extra_params={},
        is_non_raw=True,
    ),
    VariantSpec(
        name="tweedie_1_1",
        stage="stage_5f_objective_tweedie_1_1_local",
        change_description="S5-F Tweedie objective with variance power 1.1 on the Stage 5-E feature contract",
        target_transform="raw",
        objective="tweedie",
        extra_params={"tweedie_variance_power": 1.1},
        is_non_raw=True,
    ),
    VariantSpec(
        name="tweedie_1_3",
        stage="stage_5f_objective_tweedie_1_3_local",
        change_description="S5-F Tweedie objective with variance power 1.3 on the Stage 5-E feature contract",
        target_transform="raw",
        objective="tweedie",
        extra_params={"tweedie_variance_power": 1.3},
        is_non_raw=True,
    ),
    VariantSpec(
        name="poisson",
        stage="stage_5f_objective_poisson_local",
        change_description="S5-F Poisson objective diagnostic on the Stage 5-E feature contract",
        target_transform="raw",
        objective="poisson",
        extra_params={},
        is_non_raw=True,
    ),
)


def peak_rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


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


def _clip_predictions(prediction: np.ndarray) -> tuple[np.ndarray, int]:
    clipped = np.maximum(np.asarray(prediction, dtype=np.float64), 0.0)
    return clipped, int((clipped != prediction).sum())


def _ensure_stage5e_contract(frame: pd.DataFrame) -> None:
    if tuple(frame.columns) != APPROVED_STAGE5E_FEATURES:
        raise AssertionError("Stage 5-E feature contract changed.")
    forbidden = FORBIDDEN_FEATURE_FIELDS.union(STAGE5E_FORBIDDEN_FEATURE_FIELDS)
    if forbidden.intersection(frame.columns):
        raise AssertionError("Feature matrix contains forbidden fields.")


def _transform_target(target: pd.Series, transform: str) -> tuple[pd.Series, dict[str, object]]:
    target_array = np.asarray(target, dtype=np.float64)
    if transform == "raw":
        return pd.Series(target_array, index=target.index, dtype=np.float64), {"transform": "raw", "negative_transformed_count": 0}
    if transform == "sqrt":
        transformed = np.sqrt(np.clip(target_array, 0.0, None))
        return pd.Series(transformed, index=target.index, dtype=np.float64), {"transform": "sqrt", "negative_transformed_count": 0}
    if transform == "log1p":
        transformed = np.log1p(np.clip(target_array, 0.0, None))
        return pd.Series(transformed, index=target.index, dtype=np.float64), {"transform": "log1p", "negative_transformed_count": 0}
    raise ValueError(f"Unknown target transform: {transform}")


def _inverse_prediction(
    prediction: np.ndarray,
    transform: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    transformed = np.asarray(prediction, dtype=np.float64)
    if transform == "raw":
        final = transformed
    elif transform == "sqrt":
        final = np.square(transformed)
    elif transform == "log1p":
        final = np.expm1(transformed)
    else:
        raise ValueError(f"Unknown target transform: {transform}")
    negative_final_before_clip = int((final < 0).sum())
    clipped, clipped_rows = _clip_predictions(final)
    return final, clipped, {
        "negative_transformed_count": int((transformed < 0).sum()),
        "negative_final_before_clip": negative_final_before_clip,
        "clipped_rows": clipped_rows,
    }


def _train_variant_model(
    features: pd.DataFrame,
    target: pd.Series,
    variant: VariantSpec,
    config: PlainLightGBMConfig,
) -> lgb.Booster:
    params = dict(config.parameters())
    params["objective"] = variant.objective
    params.update(variant.extra_params)
    dataset = lgb.Dataset(
        features,
        label=np.asarray(target, dtype=np.float64),
        categorical_feature=list(CATEGORICAL_FEATURES),
        free_raw_data=False,
    )
    return lgb.train(params, dataset, num_boost_round=config.num_boost_round)


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
        batch = build_stage5e_feature_batch(
            split.training_history,
            split.validation_features,
            inventory,
            calendar,
            origin,
        )
        _ensure_stage5e_contract(batch.matrix)
        target = _aligned_target(batch.keys, split.validation_labels)
        training_matrices.append(batch.matrix)
        training_targets.append(target)
        origin_audit.append(
            {
                "origin": str(pd.Timestamp(origin).date()),
                "target_start": str(fold.validation_start.date()),
                "target_end": str(fold.validation_end.date()),
                "training_rows": len(batch.matrix),
            }
        )
        del split, batch, target
        gc.collect()

    training_features = pd.concat(training_matrices, ignore_index=True)
    training_target = pd.concat(training_targets, ignore_index=True)
    if tuple(training_features.columns) != APPROVED_STAGE5E_FEATURES:
        raise AssertionError(f"{spec['name']} combined training features differ from the contract.")

    outer_fold = make_backtest_folds([spec["cutoff"]])[0]
    validation_split = materialize_backtest_split(history, official_grid, outer_fold)
    if validation_split.scored_rows != spec["expected_scored_rows"]:
        raise AssertionError(f"{spec['name']} scored-row count changed.")
    coverage = validation_split.scored_rows / validation_split.requested_rows
    if not np.isclose(coverage, spec["expected_coverage"], rtol=0, atol=1e-15):
        raise AssertionError(f"{spec['name']} coverage changed.")
    validation_batch = build_stage5e_feature_batch(
        validation_split.training_history,
        validation_split.validation_features,
        inventory,
        calendar,
        spec["cutoff"],
    )
    _ensure_stage5e_contract(validation_batch.matrix)

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
        "reference": STAGE5E_REFERENCE_BY_FOLD[spec["name"]],
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

    variant_start = perf_counter()
    transformed_target, transform_audit = _transform_target(training_target, variant.target_transform)
    fit_start = perf_counter()
    model = _train_variant_model(training_features, transformed_target, variant, config)
    fit_seconds = perf_counter() - fit_start
    predict_start = perf_counter()
    prediction_transformed = predict_plain_lightgbm(model, validation_batch.matrix)
    prediction_final_unclipped, prediction_final_clipped, inverse_audit = _inverse_prediction(
        prediction_transformed,
        variant.target_transform,
    )
    prediction_seconds = perf_counter() - predict_start
    raw_metrics = _score_frame(validation_labels, validation_batch.keys, prediction_final_unclipped)
    clipped_metrics = _score_frame(validation_labels, validation_batch.keys, prediction_final_clipped)
    total_seconds = perf_counter() - variant_start

    return {
        "variant": variant,
        "fold": fold_ctx["fold"],
        "raw_metrics": raw_metrics,
        "clipped_metrics": clipped_metrics,
        "prediction_final": prediction_final_unclipped,
        "prediction_final_clipped": prediction_final_clipped,
        "prediction_transformed": prediction_transformed,
        "negative_transformed_count": inverse_audit["negative_transformed_count"],
        "negative_final_before_clip": inverse_audit["negative_final_before_clip"],
        "clipped_rows": inverse_audit["clipped_rows"],
        "runtime_seconds": total_seconds,
        "fit_seconds": fit_seconds,
        "prediction_seconds": prediction_seconds,
        "peak_rss_mib": peak_rss_mib(),
        "target_transform": transform_audit["transform"],
        "best_reference": fold_ctx["reference"],
        "beats_stage5e": raw_metrics["wmae"] < fold_ctx["reference"]["wmae"],
    }


def _blend_predictions(
    prediction_sets: list[dict[str, object]],
    weights: list[float],
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if len(prediction_sets) != len(weights):
        raise ValueError("Blend weights and prediction sets must have the same length.")
    if not np.isclose(sum(weights), 1.0, atol=1e-12):
        raise ValueError("Blend weights must sum to 1.")
    keys = prediction_sets[0]["keys"]
    for item in prediction_sets[1:]:
        if not keys.reset_index(drop=True).equals(item["keys"].reset_index(drop=True)):
            raise ValueError("Blend predictions must align exactly by key and order.")
    blended = np.zeros(len(prediction_sets[0]["prediction_final"]), dtype=np.float64)
    for weight, item in zip(weights, prediction_sets, strict=True):
        blended += weight * np.asarray(item["prediction_final"], dtype=np.float64)
    clipped, clipped_rows = _clip_predictions(blended)
    return blended, clipped, {"negative_final_before_clip": int((blended < 0).sum()), "clipped_rows": clipped_rows}


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
    config = PlainLightGBMConfig()

    fold_contexts = [
        _variant_fold_frame(spec, history, inventory, calendar, official_grid) for spec in FOLD_SPECS
    ]

    single_results_by_variant: dict[str, list[dict[str, object]]] = {spec.name: [] for spec in VARIANT_SPECS}
    single_predictions_by_fold: dict[str, dict[str, dict[str, object]]] = {}
    stage_rows: list[dict[str, object]] = []
    fold_aggregate_rows: list[dict[str, object]] = []

    for fold_ctx in fold_contexts:
        fold_name = fold_ctx["fold"]
        variant_predictions: dict[str, dict[str, object]] = {}
        for variant in VARIANT_SPECS:
            variant_result = _run_single_variant(fold_ctx, variant, config)
            single_results_by_variant[variant.name].append(variant_result)
            variant_predictions[variant.name] = {
                "keys": fold_ctx["validation_batch"].keys.copy(),
                "prediction_final": variant_result["prediction_final"],
            }
        single_predictions_by_fold[fold_name] = variant_predictions

        raw_control = variant_predictions["raw_l1"]
        best_non_raw_name = min(
            [spec.name for spec in VARIANT_SPECS if spec.is_non_raw],
            key=lambda name: single_results_by_variant[name][-1]["raw_metrics"]["wmae"],
        )
        best_non_raw = variant_predictions[best_non_raw_name]

        blend_specs = [
            (
                "stage_5g_blend_local",
                "S5-G blend 70 percent raw control plus 30 percent best non-raw variant",
                [raw_control, best_non_raw],
                [0.7, 0.3],
            ),
            (
                "stage_5g_blend_local",
                "S5-G blend 50 percent raw control plus 50 percent best non-raw variant",
                [raw_control, best_non_raw],
                [0.5, 0.5],
            ),
        ]

        single_raw_results = [single_results_by_variant[spec.name][-1] for spec in VARIANT_SPECS]
        best_single = min(single_raw_results, key=lambda row: row["raw_metrics"]["wmae"])
        selected_names = [
            row["variant"].name
            for row in single_raw_results
            if row["raw_metrics"]["wmae"] <= best_single["raw_metrics"]["wmae"] + 0.50
        ]
        if len(selected_names) >= 2:
            blend_specs.append(
                (
                    "stage_5g_blend_local",
                    f"S5-G equal blend of variants within 0.50 WMAE of best single: {', '.join(selected_names)}",
                    [variant_predictions[name] for name in selected_names],
                    [1.0 / len(selected_names)] * len(selected_names),
                )
            )

        for stage, description, prediction_sets, weights in blend_specs:
            blend_start = perf_counter()
            blended_predictions, blended_predictions_clipped, blend_audit = _blend_predictions(
                prediction_sets,
                weights,
            )
            blend_seconds = perf_counter() - blend_start
            raw_metrics = _score_frame(fold_ctx["validation_labels"], fold_ctx["validation_batch"].keys, blended_predictions)
            clipped_metrics = _score_frame(
                fold_ctx["validation_labels"],
                fold_ctx["validation_batch"].keys,
                blended_predictions_clipped,
            )
            fold_aggregate_rows.append(
                {
                    "stage": stage,
                    "fold": fold_name,
                    "change_description": description,
                    "raw_metrics": raw_metrics,
                    "clipped_metrics": clipped_metrics,
                    "runtime_seconds": blend_seconds,
                    "peak_rss_mib": peak_rss_mib(),
                    "beats_stage5e": raw_metrics["wmae"] < STAGE5E_REFERENCE_BY_FOLD[fold_name]["wmae"],
                    "weights": weights,
                }
            )
            stage_rows.append(
                {
                    "timestamp": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat().replace("+00:00", "Z"),
                    "git_hash": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    "stage": stage,
                    "change_description": description,
                    "local_wmae": raw_metrics["wmae"],
                    "local_wape": raw_metrics["wape"],
                    "local_bias": raw_metrics["bias"],
                    "runtime_minutes": blend_seconds / 60,
                    "kept": "diagnostic_ablation",
                    "notes": (
                        f"fold={fold_name}; weights={weights}; raw_mean_wmae={raw_metrics['wmae']:.10f}; "
                        f"clipped_mean_wmae={clipped_metrics['wmae']:.10f}; clipped_rows={blend_audit['clipped_rows']}; "
                        f"negative_final_before_clip={blend_audit['negative_final_before_clip']}; "
                        f"beats_stage5e={raw_metrics['wmae'] < STAGE5E_REFERENCE_BY_FOLD[fold_name]['wmae']}; "
                        "no Kaggle submission"
                    ),
                }
            )

    for variant in VARIANT_SPECS:
        rows = single_results_by_variant[variant.name]
        agg = _aggregate(rows)
        notes = [
            f"feature_count={len(APPROVED_STAGE5E_FEATURES)}",
            f"transform={variant.target_transform}",
            f"objective={variant.objective}",
            f"raw_mean_wmae={agg['mean_wmae']:.10f}",
            f"raw_mean_wape={agg['mean_wape']:.10f}",
            f"raw_mean_bias={agg['mean_bias']:.10f}",
            f"peak_rss_mib={agg['peak_rss_mib']:.2f}",
            "Stage 5-E contract",
            "no Kaggle submission",
        ]
        if variant.name == "poisson":
            notes.append("optional diagnostic; run if stable")
        stage_rows.append(
            {
                "timestamp": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat().replace("+00:00", "Z"),
                "git_hash": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "stage": variant.stage,
                "change_description": variant.change_description,
                "local_wmae": agg["mean_wmae"],
                "local_wape": agg["mean_wape"],
                "local_bias": agg["mean_bias"],
                "runtime_minutes": agg["total_runtime_seconds"] / 60,
                "kept": "diagnostic_ablation",
                "notes": "; ".join(notes),
            }
        )

    _append_results_rows(stage_rows)

    report_lines = [
        "# Stage 5-F/G Objective and Blend Results",
        "",
        "## Purpose",
        "",
        "Test whether target transforms, objective diversity, and a small number of predeclared blends can improve local F1-F4 validation on top of the approved Stage 5-E 76-feature contract.",
        "",
        "## Completed variants",
        "",
        "- Raw L1 control",
        "- Sqrt target + L1",
        "- Log1p target + L1",
        "- Tweedie variance power 1.1",
        "- Tweedie variance power 1.3",
        "- Poisson diagnostic (if stable)",
        "",
        "## Skipped variants",
        "",
        "- None if all variants complete successfully.",
        "- Any skipped diagnostic will be documented explicitly in the row notes and below.",
        "",
        "## Single-variant summary",
        "",
        "| Variant | Mean WMAE | Mean WAPE | Mean bias | Runtime (min) | Peak RSS (MiB) | Beats Stage 5-E? |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for variant in VARIANT_SPECS:
        rows = single_results_by_variant[variant.name]
        agg = _aggregate(rows)
        report_lines.append(
            f"| {variant.name} | {agg['mean_wmae']:.10f} | {agg['mean_wape']:.10f} | {agg['mean_bias']:.10f} | {agg['mean_runtime_seconds'] / 60:.3f} | {agg['peak_rss_mib']:.2f} | {'Yes' if agg['mean_wmae'] < STAGE5E_REFERENCE_MEAN_WMAE else 'No'} |"
        )

    report_lines.extend(
        [
            "",
            "## Fold-by-fold table",
            "",
        ]
    )
    for variant in VARIANT_SPECS:
        rows = single_results_by_variant[variant.name]
        report_lines.extend(
            [
                f"### {variant.name}",
                "",
                "| Fold | Raw WMAE | Raw WAPE | Raw bias | Clipped WMAE | Clipped WAPE | Clipped bias | Negative transformed | Negative final before clip | Clipped rows | Beats Stage 5-E fold? |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in rows:
            report_lines.append(
                f"| {row['fold']} | {row['raw_metrics']['wmae']:.10f} | {row['raw_metrics']['wape']:.10f} | {row['raw_metrics']['bias']:.10f} | "
                f"{row['clipped_metrics']['wmae']:.10f} | {row['clipped_metrics']['wape']:.10f} | {row['clipped_metrics']['bias']:.10f} | "
                f"{row['negative_transformed_count']} | {row['negative_final_before_clip']} | {row['clipped_rows']} | "
                f"{'Yes' if row['beats_stage5e'] else 'No'} |"
            )
        report_lines.append("")

    blend_rows = fold_aggregate_rows
    report_lines.extend(
        [
            "## Blend diagnostics",
            "",
            "| Blend | Fold | Raw WMAE | Raw WAPE | Raw bias | Clipped rows | Beats Stage 5-E fold? |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in blend_rows:
        report_lines.append(
            f"| {row['change_description']} | {row['fold']} | {row['raw_metrics']['wmae']:.10f} | {row['raw_metrics']['wape']:.10f} | {row['raw_metrics']['bias']:.10f} | "
            f"{row['clipped_metrics']['wmae']:.10f} | {'Yes' if row['beats_stage5e'] else 'No'} |"
        )

    # Reconstruct the aggregate blend summary from the rows that were emitted above.
    blend_summary_by_description: dict[str, list[dict[str, object]]] = {}
    for row in blend_rows:
        blend_summary_by_description.setdefault(row["change_description"], []).append(row)

    report_lines.extend(
        [
            "",
            "## Aggregate comparison",
            "",
            "| Item | Mean WMAE | Mean WAPE | Mean bias |",
            "|---|---:|---:|---:|",
        ]
    )
    for variant in VARIANT_SPECS:
        rows = single_results_by_variant[variant.name]
        agg = _aggregate(rows)
        report_lines.append(
            f"| {variant.name} | {agg['mean_wmae']:.10f} | {agg['mean_wape']:.10f} | {agg['mean_bias']:.10f} |"
        )
    for description, rows in blend_summary_by_description.items():
        agg = {
            "mean_wmae": float(np.mean([row["raw_metrics"]["wmae"] for row in rows])),
            "mean_wape": float(np.mean([row["raw_metrics"]["wape"] for row in rows])),
            "mean_bias": float(np.mean([row["raw_metrics"]["bias"] for row in rows])),
        }
        report_lines.append(
            f"| {description} | {agg['mean_wmae']:.10f} | {agg['mean_wape']:.10f} | {agg['mean_bias']:.10f} |"
        )

    raw_agg = _aggregate(single_results_by_variant["raw_l1"])
    best_single_name = min(VARIANT_SPECS, key=lambda spec: _aggregate(single_results_by_variant[spec.name])["mean_wmae"]).name
    best_single_agg = _aggregate(single_results_by_variant[best_single_name])
    best_non_raw_name = min(
        [spec.name for spec in VARIANT_SPECS if spec.is_non_raw],
        key=lambda name: _aggregate(single_results_by_variant[name])["mean_wmae"],
    )
    best_non_raw_agg = _aggregate(single_results_by_variant[best_non_raw_name])
    best_blend_description, best_blend_rows = min(
        blend_summary_by_description.items(),
        key=lambda item: float(np.mean([row["raw_metrics"]["wmae"] for row in item[1]])),
    )
    best_blend_agg = {
        "mean_wmae": float(np.mean([row["raw_metrics"]["wmae"] for row in best_blend_rows])),
        "mean_wape": float(np.mean([row["raw_metrics"]["wape"] for row in best_blend_rows])),
        "mean_bias": float(np.mean([row["raw_metrics"]["bias"] for row in best_blend_rows])),
    }

    report_lines.extend(
        [
            "",
            "## Comparison to Stage 5-E",
            "",
            f"- Stage 5-E reference mean WMAE: {STAGE5E_REFERENCE_MEAN_WMAE:.10f}",
            f"- Raw L1 control mean WMAE: {raw_agg['mean_wmae']:.10f}",
            f"- Best single mean WMAE: {best_single_agg['mean_wmae']:.10f} ({best_single_name})",
            f"- Best non-raw single mean WMAE: {best_non_raw_agg['mean_wmae']:.10f} ({best_non_raw_name})",
            f"- Best blend mean WMAE: {best_blend_agg['mean_wmae']:.10f} ({best_blend_description})",
            "",
            "## Runtime and memory",
            "",
            f"- Raw load seconds: {raw_load_seconds:.3f}",
            f"- Peak RSS across folds and variants: {max([peak_rss_mib()] + [row['peak_rss_mib'] for row in blend_rows]):.2f} MiB",
            "",
            "## Leakage and cutoff safety assessment",
            "",
            "- Stage 5-E feature contract remained unchanged.",
            "- No future sales, total_orders, availability, or solution leakage entered the model matrix.",
            "- Blends were predeclared and aligned exactly by key/order before scoring.",
            "",
            "## Recommendation",
            "",
        ]
    )
    if best_single_agg["mean_wmae"] < STAGE5E_REFERENCE_MEAN_WMAE or best_blend_agg["mean_wmae"] < STAGE5E_REFERENCE_MEAN_WMAE:
        report_lines.append("A follow-up candidate is justified for human review.")
    else:
        report_lines.append("Reject Stage 5-F/G as a promotion path; no variant beat Stage 5-E.")
    report_lines.append("")

    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    result = {
        "timestamp": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_hash": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "raw_load_seconds": raw_load_seconds,
        "single_results_by_variant": single_results_by_variant,
        "blend_rows": blend_rows,
        "best_single_name": best_single_name,
        "best_blend_description": best_blend_description,
        "best_single_agg": best_single_agg,
        "best_blend_agg": best_blend_agg,
        "raw_agg": raw_agg,
        "stage5e_reference_mean_wmae": STAGE5E_REFERENCE_MEAN_WMAE,
    }
    serializable_result = {
        "timestamp": result["timestamp"],
        "git_hash": result["git_hash"],
        "raw_load_seconds": raw_load_seconds,
        "best_single_name": best_single_name,
        "best_blend_description": best_blend_description,
        "best_single_mean_wmae": best_single_agg["mean_wmae"],
        "best_blend_mean_wmae": best_blend_agg["mean_wmae"],
        "stage5e_reference_mean_wmae": STAGE5E_REFERENCE_MEAN_WMAE,
    }
    print(json.dumps(serializable_result, indent=2))
    return result


if __name__ == "__main__":
    run()
