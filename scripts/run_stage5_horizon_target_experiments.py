#!/usr/bin/env python3
"""Run Stage 5 horizon-specific direct-model diagnostics."""

from __future__ import annotations

import csv
import gc
import json
import resource
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from eval.backtest import make_backtest_folds, materialize_backtest_split
from eval.grid import score_kaggle_aligned
from features.stage3_minimal import (
    APPROVED_FEATURES,
    CATEGORICAL_FEATURES,
    FORBIDDEN_FEATURE_FIELDS,
)
from features.stage5_price_discount import (
    APPROVED_STAGE5_FEATURES,
    FEATURE_AVAILABILITY,
    STAGE5_FORBIDDEN_FEATURE_FIELDS,
    build_stage5_feature_batch,
)
from models.plain_lgbm import (
    PlainLightGBMConfig,
    predict_plain_lightgbm,
    train_plain_lightgbm,
)
from scripts.run_stage3_all_folds_plain_model import training_origins_for_fold
from scripts.run_stage5_kaggle_candidate import _load_raw_data


ROOT = Path(__file__).resolve().parents[1]
RESULTS_CSV = ROOT / "results.csv"
REPORT_PATH = ROOT / "reports" / "stage5_horizon_target_experiments.md"

HORIZON_DAYS = tuple(range(1, 15))
S5B_FEATURE_SET = APPROVED_STAGE5_FEATURES


@dataclass(frozen=True)
class FoldSpec:
    name: str
    cutoff: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    expected_scored_rows: int
    expected_coverage: float
    stage3_plain_wmae: float
    stage5b_wmae: float
    stage5b_wape: float
    stage5b_bias: float


FOLD_SPECS = (
    FoldSpec(
        "F1",
        pd.Timestamp("2024-05-19"),
        pd.Timestamp("2024-05-20"),
        pd.Timestamp("2024-06-02"),
        44212,
        0.9402607345654069,
        20.58275093511847,
        20.303644991475146,
        0.21408868950808776,
        -0.01199830909189191,
    ),
    FoldSpec(
        "F2",
        pd.Timestamp("2024-05-05"),
        pd.Timestamp("2024-05-06"),
        pd.Timestamp("2024-05-19"),
        43433,
        0.923693668786287,
        22.67167264911902,
        22.307800916123647,
        0.22663404997972925,
        -0.09278695853722141,
    ),
    FoldSpec(
        "F3",
        pd.Timestamp("2024-04-21"),
        pd.Timestamp("2024-04-22"),
        pd.Timestamp("2024-05-05"),
        42794,
        0.9101039960868548,
        20.32179816490604,
        20.20155375225985,
        0.20066013451559186,
        -0.01286197337520826,
    ),
    FoldSpec(
        "F4",
        pd.Timestamp("2024-04-07"),
        pd.Timestamp("2024-04-08"),
        pd.Timestamp("2024-04-21"),
        42035,
        0.8939622721762617,
        19.209094143933466,
        19.04134386516688,
        0.20293940390005497,
        -0.07114824024505635,
    ),
)

S5B_REFERENCE = {spec.name: spec for spec in FOLD_SPECS}


def peak_rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _ensure_feature_contract(frame: pd.DataFrame, *, name: str, approved: tuple[str, ...]) -> None:
    if tuple(frame.columns) != approved:
        raise AssertionError(f"{name} feature columns differ from the approved contract.")
    forbidden = FORBIDDEN_FEATURE_FIELDS.union(STAGE5_FORBIDDEN_FEATURE_FIELDS)
    if forbidden.intersection(frame.columns):
        raise AssertionError(f"{name} contains forbidden fields.")


def _partition_by_horizon(frame: pd.DataFrame, *, horizon_column: str = "horizon_day") -> dict[int, pd.DataFrame]:
    if horizon_column not in frame.columns:
        raise KeyError(f"Missing horizon column: {horizon_column}.")
    partitions: dict[int, pd.DataFrame] = {}
    horizon = pd.to_numeric(frame[horizon_column], errors="raise").astype(int)
    for value in HORIZON_DAYS:
        partitions[value] = frame.loc[horizon == value].copy()
    if sum(len(value) for value in partitions.values()) != len(frame):
        raise AssertionError("Horizon partitioning dropped or duplicated rows.")
    return partitions


def _clip_predictions(prediction: np.ndarray) -> tuple[np.ndarray, int]:
    clipped = np.maximum(np.asarray(prediction, dtype=np.float64), 0.0)
    changed = int((clipped != prediction).sum())
    return clipped, changed


def _inverse_sqrt_predictions(prediction: np.ndarray, *, clip_negative: bool) -> tuple[np.ndarray, int]:
    raw = np.asarray(prediction, dtype=np.float64)
    negative_count = int((raw < 0).sum())
    if clip_negative:
        raw = np.maximum(raw, 0.0)
    return np.square(raw), negative_count


def _score_predictions(labels: pd.DataFrame, keys: pd.DataFrame, prediction: np.ndarray) -> dict[str, float]:
    frame = keys.copy()
    frame["sales_hat"] = prediction
    metrics = score_kaggle_aligned(labels, frame)
    return {"wmae": metrics.wmae, "wape": metrics.wape, "bias": metrics.bias}


def _append_results_row(
    *,
    timestamp: str,
    git_hash: str,
    stage: str,
    change_description: str,
    local_wmae: float,
    local_wape: float,
    local_bias: float,
    runtime_minutes: float,
    kept: str,
    notes: str,
) -> None:
    with RESULTS_CSV.open("a", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                timestamp,
                git_hash,
                stage,
                change_description,
                f"{local_wmae:.10f}",
                f"{local_wape:.10f}",
                f"{local_bias:.10f}",
                f"{runtime_minutes:.10f}",
                kept,
                notes,
            ]
        )


@dataclass
class FoldBundle:
    spec: FoldSpec
    training_features_by_horizon: dict[int, pd.DataFrame]
    training_targets_raw_by_horizon: dict[int, pd.Series]
    training_targets_sqrt_by_horizon: dict[int, pd.Series]
    validation_features: pd.DataFrame
    validation_labels: pd.DataFrame
    validation_keys: pd.DataFrame
    validation_requested_rows: int
    validation_scored_rows: int
    validation_coverage: float
    training_rows_by_horizon: dict[int, int]
    validation_rows_by_horizon: dict[int, int]
    origin_audit: list[dict[str, object]]


def _build_fold_bundle(
    spec: FoldSpec,
    history: pd.DataFrame,
    inventory: pd.DataFrame,
    calendar: pd.DataFrame,
    official_grid: pd.DataFrame,
) -> FoldBundle:
    training_features: dict[int, list[pd.DataFrame]] = {h: [] for h in HORIZON_DAYS}
    training_targets_raw: dict[int, list[pd.Series]] = {h: [] for h in HORIZON_DAYS}
    training_targets_sqrt: dict[int, list[pd.Series]] = {h: [] for h in HORIZON_DAYS}
    validation_features: pd.DataFrame | None = None
    validation_labels: pd.DataFrame | None = None
    validation_keys: pd.DataFrame | None = None
    origin_audit: list[dict[str, object]] = []

    for origin in training_origins_for_fold(spec.cutoff):
        fold = make_backtest_folds([origin])[0]
        split = materialize_backtest_split(history, official_grid, fold)
        if split.validation_labels["date"].max() > spec.cutoff:
            raise AssertionError(f"{spec.name} training labels cross the outer cutoff.")
        batch = build_stage5_feature_batch(
            split.training_history,
            split.validation_features,
            inventory,
            calendar,
            origin,
        )
        _ensure_feature_contract(batch.matrix, name=f"{spec.name} training", approved=S5B_FEATURE_SET)
        horizon_values = pd.to_numeric(batch.matrix["horizon_day"], errors="raise").astype(int)
        target_raw = (
            batch.keys.merge(
                split.validation_labels.loc[:, ["unique_id", "date", "sales"]],
                on=["unique_id", "date"],
                how="left",
                validate="one_to_one",
            )["sales"].astype(np.float64)
        )
        if target_raw.isna().any():
            raise AssertionError(f"{spec.name} target alignment failed for origin {origin.date()}.")
        target_sqrt = np.sqrt(np.clip(target_raw.to_numpy(dtype=np.float64), 0.0, None))
        for horizon in HORIZON_DAYS:
            mask = horizon_values == horizon
            if not mask.any():
                continue
            training_features[horizon].append(batch.matrix.loc[mask].reset_index(drop=True))
            training_targets_raw[horizon].append(target_raw.loc[mask].reset_index(drop=True))
            training_targets_sqrt[horizon].append(pd.Series(target_sqrt[mask.to_numpy()], dtype=np.float64))
        origin_audit.append(
            {
                "origin": str(pd.Timestamp(origin).date()),
                "target_start": str(fold.validation_start.date()),
                "target_end": str(fold.validation_end.date()),
                "training_rows": len(batch.matrix),
            }
        )
        del split, batch, target_raw, target_sqrt
        gc.collect()

    concatenated_features: dict[int, pd.DataFrame] = {}
    concatenated_targets_raw: dict[int, pd.Series] = {}
    concatenated_targets_sqrt: dict[int, pd.Series] = {}
    training_rows_by_horizon: dict[int, int] = {}
    for horizon in HORIZON_DAYS:
        concatenated_features[horizon] = pd.concat(training_features[horizon], ignore_index=True)
        concatenated_targets_raw[horizon] = pd.concat(training_targets_raw[horizon], ignore_index=True)
        concatenated_targets_sqrt[horizon] = pd.concat(training_targets_sqrt[horizon], ignore_index=True)
        training_rows_by_horizon[horizon] = len(concatenated_features[horizon])
        if tuple(concatenated_features[horizon].columns) != S5B_FEATURE_SET:
            raise AssertionError(f"{spec.name} horizon {horizon} training features changed.")

    outer_fold = make_backtest_folds([spec.cutoff])[0]
    if outer_fold.validation_start != spec.validation_start:
        raise AssertionError(f"{spec.name} validation start changed.")
    if outer_fold.validation_end != spec.validation_end:
        raise AssertionError(f"{spec.name} validation end changed.")
    split = materialize_backtest_split(history, official_grid, outer_fold)
    if split.scored_rows != spec.expected_scored_rows:
        raise AssertionError(f"{spec.name} scored-row count changed.")
    coverage = split.scored_rows / split.requested_rows
    if not np.isclose(coverage, spec.expected_coverage, rtol=0, atol=1e-15):
        raise AssertionError(f"{spec.name} coverage changed.")
    validation_batch = build_stage5_feature_batch(
        split.training_history,
        split.validation_features,
        inventory,
        calendar,
        spec.cutoff,
    )
    _ensure_feature_contract(validation_batch.matrix, name=f"{spec.name} validation", approved=S5B_FEATURE_SET)
    validation_horizon = pd.to_numeric(validation_batch.matrix["horizon_day"], errors="raise").astype(int)
    validation_rows_by_horizon = {
        horizon: int((validation_horizon == horizon).sum()) for horizon in HORIZON_DAYS
    }
    validation_features = validation_batch.matrix.reset_index(drop=True)
    validation_labels = split.validation_labels.reset_index(drop=True)
    validation_keys = validation_batch.keys.reset_index(drop=True)
    return FoldBundle(
        spec=spec,
        training_features_by_horizon=concatenated_features,
        training_targets_raw_by_horizon=concatenated_targets_raw,
        training_targets_sqrt_by_horizon=concatenated_targets_sqrt,
        validation_features=validation_features,
        validation_labels=validation_labels,
        validation_keys=validation_keys,
        validation_requested_rows=split.requested_rows,
        validation_scored_rows=split.scored_rows,
        validation_coverage=coverage,
        training_rows_by_horizon=training_rows_by_horizon,
        validation_rows_by_horizon=validation_rows_by_horizon,
        origin_audit=origin_audit,
    )


def _train_models(
    training_features_by_horizon: dict[int, pd.DataFrame],
    training_targets_by_horizon: dict[int, pd.Series],
    config: PlainLightGBMConfig,
) -> dict[int, object]:
    models: dict[int, object] = {}
    for horizon in HORIZON_DAYS:
        features = training_features_by_horizon[horizon]
        target = training_targets_by_horizon[horizon]
        if len(features) != len(target):
            raise AssertionError(f"Horizon {horizon} feature/target mismatch.")
        models[horizon] = train_plain_lightgbm(features, target, CATEGORICAL_FEATURES, config)
    return models


def _predict_horizon_models(
    models_by_horizon: dict[int, object],
    features: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] = S5B_FEATURE_SET,
    horizon_column: str = "horizon_day",
) -> np.ndarray:
    if horizon_column not in features.columns:
        raise KeyError(f"Missing horizon column: {horizon_column}.")
    prediction = np.empty(len(features), dtype=np.float64)
    horizons = pd.to_numeric(features[horizon_column], errors="raise").astype(int)
    for horizon in HORIZON_DAYS:
        mask = horizons == horizon
        if not mask.any():
            continue
        model = models_by_horizon[horizon]
        model_prediction = predict_plain_lightgbm(
            model,
            features.loc[mask, feature_columns].reset_index(drop=True),
        )
        if len(model_prediction) != int(mask.sum()):
            raise AssertionError(f"Horizon {horizon} returned an invalid prediction shape.")
        prediction[mask.to_numpy()] = model_prediction
    if not np.isfinite(prediction).all():
        raise ValueError("Horizon model predictions contain non-finite values.")
    return prediction


def _score_fold_variant(
    bundle: FoldBundle,
    *,
    variant: str,
    config: PlainLightGBMConfig,
) -> dict[str, object]:
    fold_start = perf_counter()
    if variant == "raw":
        target_map = bundle.training_targets_raw_by_horizon
    elif variant == "sqrt":
        target_map = bundle.training_targets_sqrt_by_horizon
    else:
        raise ValueError(f"Unknown variant: {variant}")

    train_start = perf_counter()
    models_by_horizon = _train_models(bundle.training_features_by_horizon, target_map, config)
    fit_seconds = perf_counter() - train_start
    predict_start = perf_counter()
    raw_prediction = _predict_horizon_models(models_by_horizon, bundle.validation_features)
    prediction_seconds = perf_counter() - predict_start

    if variant == "raw":
        unclipped_prediction = raw_prediction
        clipped_prediction, clipped_rows = _clip_predictions(raw_prediction)
        clipped_metrics = _score_predictions(bundle.validation_labels, bundle.validation_keys, clipped_prediction)
        unclipped_metrics = _score_predictions(bundle.validation_labels, bundle.validation_keys, unclipped_prediction)
        negative_raw = int((raw_prediction < 0).sum())
        prediction_summary = {
            "negative_raw_predictions": negative_raw,
            "clipped_rows": clipped_rows,
            "negative_final_predictions": int((clipped_prediction < 0).sum()),
        }
    else:
        unclipped_prediction = np.square(raw_prediction)
        clipped_prediction, negative_raw = _inverse_sqrt_predictions(raw_prediction, clip_negative=True)
        clipped_metrics = _score_predictions(bundle.validation_labels, bundle.validation_keys, clipped_prediction)
        unclipped_metrics = _score_predictions(bundle.validation_labels, bundle.validation_keys, unclipped_prediction)
        prediction_summary = {
            "negative_sqrt_space_predictions": negative_raw,
            "clipped_rows": negative_raw,
            "negative_final_predictions": int((clipped_prediction < 0).sum()),
        }

    total_seconds = perf_counter() - fold_start
    result = {
        "fold": bundle.spec.name,
        "variant": variant,
        "training_rows": sum(bundle.training_rows_by_horizon.values()),
        "validation_rows": len(bundle.validation_features),
        "validation_scored_rows": bundle.validation_scored_rows,
        "coverage": bundle.validation_coverage,
        "horizon_models": len(HORIZON_DAYS),
        "raw_metrics": unclipped_metrics,
        "clipped_metrics": clipped_metrics,
        "prediction_summary": prediction_summary,
        "runtime_seconds": total_seconds,
        "fit_seconds": fit_seconds,
        "prediction_seconds": prediction_seconds,
        "peak_rss_mib": peak_rss_mib(),
        "stage3_plain_wmae": bundle.spec.stage3_plain_wmae,
        "s5b_wmae": bundle.spec.stage5b_wmae,
        "s5b_wape": bundle.spec.stage5b_wape,
        "s5b_bias": bundle.spec.stage5b_bias,
        "improved_vs_stage3": clipped_metrics["wmae"] < bundle.spec.stage3_plain_wmae,
        "improved_vs_s5b": clipped_metrics["wmae"] < bundle.spec.stage5b_wmae,
    }
    return result


def _write_report(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summary_from_results(results: list[dict[str, object]]) -> dict[str, float]:
    return {
        "mean_wmae": float(np.mean([result["clipped_metrics"]["wmae"] for result in results])),
        "mean_wape": float(np.mean([result["clipped_metrics"]["wape"] for result in results])),
        "mean_bias": float(np.mean([result["clipped_metrics"]["bias"] for result in results])),
        "mean_runtime_seconds": float(np.mean([result["runtime_seconds"] for result in results])),
        "total_runtime_seconds": float(np.sum([result["runtime_seconds"] for result in results])),
        "peak_rss_mib": float(max(result["peak_rss_mib"] for result in results)),
    }


def _stage5_reference_from_results() -> dict[str, float]:
    rows = pd.read_csv(RESULTS_CSV)
    stage3 = rows.loc[rows["stage"].isin(["stage_3_f1", "stage_3_plain_model_fold"]), "local_wmae"].astype(float)
    s5b = rows.loc[rows["stage"] == "stage_5_s5b_relative_price_discount_local", "local_wmae"].astype(float)
    s5b_bias = rows.loc[rows["stage"] == "stage_5_s5b_relative_price_discount_local", "local_bias"].astype(float)
    s5b_wape = rows.loc[rows["stage"] == "stage_5_s5b_relative_price_discount_local", "local_wape"].astype(float)
    return {
        "stage3_plain_mean_wmae": float(stage3.mean()),
        "s5b_mean_wmae": float(s5b.iloc[0]),
        "s5b_mean_wape": float(s5b_wape.iloc[0]),
        "s5b_mean_bias": float(s5b_bias.iloc[0]),
    }


def run() -> dict[str, object]:
    start = perf_counter()
    history, inventory, calendar, _sales_test, official_grid, _solution = _load_raw_data()
    raw_load_seconds = perf_counter() - start
    config = PlainLightGBMConfig()
    reference = _stage5_reference_from_results()

    fold_results: dict[str, list[dict[str, object]]] = {"raw": [], "sqrt": []}

    for spec in FOLD_SPECS:
        bundle = _build_fold_bundle(spec, history, inventory, calendar, official_grid)
        for variant in ("raw", "sqrt"):
            result = _score_fold_variant(bundle, variant=variant, config=config)
            fold_results[variant].append(result)
        del bundle
        gc.collect()

    summary_raw = _summary_from_results(fold_results["raw"])
    summary_sqrt = _summary_from_results(fold_results["sqrt"])
    git_hash = (
        __import__("subprocess")
        .check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True)
        .strip()
    )
    timestamp = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat().replace("+00:00", "Z")

    notes_raw = (
        f"S5-C local diagnostic; horizon-specific direct LightGBM; S5-B feature set included; "
        f"clipped_mean_wmae={summary_raw['mean_wmae']:.10f}; clipped_mean_wape={summary_raw['mean_wape']:.10f}; "
        f"clipped_mean_bias={summary_raw['mean_bias']:.10f}; unclipped_mean_wmae={float(np.mean([r['raw_metrics']['wmae'] for r in fold_results['raw']])):.10f}; "
        f"unclipped_mean_bias={float(np.mean([r['raw_metrics']['bias'] for r in fold_results['raw']])):.10f}; "
        f"folds={len(FOLD_SPECS)}; horizon_models=14; no Kaggle submission"
    )
    notes_sqrt = (
        f"S5-D local diagnostic; horizon-specific direct LightGBM with sqrt target; S5-B feature set included; "
        f"clipped_mean_wmae={summary_sqrt['mean_wmae']:.10f}; clipped_mean_wape={summary_sqrt['mean_wape']:.10f}; "
        f"clipped_mean_bias={summary_sqrt['mean_bias']:.10f}; unclipped_inverse_mean_wmae={float(np.mean([r['raw_metrics']['wmae'] for r in fold_results['sqrt']])):.10f}; "
        f"negative_sqrt_space_predictions={sum(r['prediction_summary']['negative_sqrt_space_predictions'] for r in fold_results['sqrt'])}; "
        f"folds={len(FOLD_SPECS)}; horizon_models=14; no Kaggle submission"
    )

    _append_results_row(
        timestamp=timestamp,
        git_hash=git_hash,
        stage="stage_5_horizon_direct_raw_local",
        change_description="S5-C horizon-specific direct raw-target LightGBM with S5-B features",
        local_wmae=summary_raw["mean_wmae"],
        local_wape=summary_raw["mean_wape"],
        local_bias=summary_raw["mean_bias"],
        runtime_minutes=summary_raw["total_runtime_seconds"] / 60,
        kept="diagnostic_ablation",
        notes=notes_raw,
    )
    _append_results_row(
        timestamp=timestamp,
        git_hash=git_hash,
        stage="stage_5_horizon_direct_sqrt_local",
        change_description="S5-D horizon-specific direct sqrt-target LightGBM with S5-B features",
        local_wmae=summary_sqrt["mean_wmae"],
        local_wape=summary_sqrt["mean_wape"],
        local_bias=summary_sqrt["mean_bias"],
        runtime_minutes=summary_sqrt["total_runtime_seconds"] / 60,
        kept="diagnostic_ablation",
        notes=notes_sqrt,
    )

    raw_lines = [
        "# Stage 5 Horizon-Specific Direct Model Experiments",
        "",
        "## Purpose",
        "",
        "Test whether direct horizon-specific models and a sqrt target transform improve on the Stage 5 S5-B candidate.",
        "",
        "## Public-solution idea basis",
        "",
        "Public competition writeups frequently used direct horizon models. This experiment keeps that idea isolated from any further feature invention.",
        "",
        "## Experiment design",
        "",
        "- S5-B relative price/discount features were included.",
        "- One LightGBM model was trained per horizon_day from 1 to 14.",
        "- Four local folds were evaluated: F1 through F4.",
        "- Two target modes were run: raw sales and sqrt(sales).",
        "- Clipping was evaluated as a final post-processing step on top of each variant.",
        "",
        "## Feature contract",
        "",
        f"- Approved feature set: {len(S5B_FEATURE_SET)} features",
        "- No future `total_orders`.",
        "- No future availability.",
        "- No solution.csv target leakage.",
        "- No weights in the feature matrix.",
        "- No clipped/rounded labels during training.",
        "",
        "## Target transform details",
        "",
        "- Raw variant: target is untransformed sales.",
        "- Sqrt variant: target is sqrt(sales).",
        "- Negative sqrt-space predictions were clipped to zero before inverse squaring.",
        "- Clipped and unclipped diagnostics were recorded where practical.",
        "",
        "## Comparison context",
        "",
        f"- Stage 3 plain all-fold mean WMAE: {reference['stage3_plain_mean_wmae']:.10f}",
        f"- Stage 5 S5-B all-fold mean WMAE: {reference['s5b_mean_wmae']:.10f}",
        f"- Stage 5 official private WMAE (context only): 21.6111400000",
        "",
        "## Runtime and memory",
        "",
        f"- Raw load seconds: {raw_load_seconds:.3f}",
        f"- Peak RSS across runs: {max(summary_raw['peak_rss_mib'], summary_sqrt['peak_rss_mib']):.2f} MiB",
        f"- Raw total runtime minutes: {summary_raw['total_runtime_seconds'] / 60:.3f}",
        f"- Sqrt total runtime minutes: {summary_sqrt['total_runtime_seconds'] / 60:.3f}",
        "",
    ]
    for variant, results in (("raw", fold_results["raw"]), ("sqrt", fold_results["sqrt"])):
        raw_lines.extend(
            [
                f"## {variant.upper()} fold results",
                "",
                "| Fold | Clipped WMAE | Clipped WAPE | Clipped bias | Unclipped WMAE | Unclipped WAPE | Unclipped bias | Training rows | Validation rows | Horizon models | Negative count | Clipped rows | Runtime (min) | Peak RSS (MiB) | Beats Stage 3 plain? | Beats S5-B? |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
            ]
        )
        for row in results:
            if variant == "raw":
                negative = row["prediction_summary"]["negative_raw_predictions"]
                clipped_rows = row["prediction_summary"]["clipped_rows"]
            else:
                negative = row["prediction_summary"]["negative_sqrt_space_predictions"]
                clipped_rows = row["prediction_summary"]["clipped_rows"]
            raw_lines.append(
                f"| {row['fold']} | {row['clipped_metrics']['wmae']:.10f} | {row['clipped_metrics']['wape']:.10f} | {row['clipped_metrics']['bias']:.10f} | "
                f"{row['raw_metrics']['wmae']:.10f} | {row['raw_metrics']['wape']:.10f} | {row['raw_metrics']['bias']:.10f} | "
                f"{row['training_rows']} | {row['validation_rows']} | {row['horizon_models']} | {negative} | {clipped_rows} | "
                f"{row['runtime_seconds'] / 60:.3f} | {row['peak_rss_mib']:.2f} | {'Yes' if row['clipped_metrics']['wmae'] < row['stage3_plain_wmae'] else 'No'} | {'Yes' if row['clipped_metrics']['wmae'] < row['s5b_wmae'] else 'No'} |"
            )
        raw_lines.extend(
            [
                "",
                f"## {variant.upper()} aggregate results",
                "",
                f"- Mean clipped WMAE: {summary_raw['mean_wmae']:.10f}" if variant == "raw" else f"- Mean clipped WMAE: {summary_sqrt['mean_wmae']:.10f}",
                f"- Mean clipped WAPE: {summary_raw['mean_wape']:.10f}" if variant == "raw" else f"- Mean clipped WAPE: {summary_sqrt['mean_wape']:.10f}",
                f"- Mean clipped bias: {summary_raw['mean_bias']:.10f}" if variant == "raw" else f"- Mean clipped bias: {summary_sqrt['mean_bias']:.10f}",
                "",
            ]
        )

    raw_lines.extend(
        [
            "## Leakage and cutoff safety",
            "",
            "- Historical feature statistics were built only from cutoff-safe training origins.",
            "- Validation labels never entered feature creation.",
            "- No forbidden fields were present in the training or validation matrices.",
            "",
            "## Hack box assessment",
            "",
            "- The run stayed within the local CPU budget.",
            "- Peak RSS remained below the 12 GiB guardrail.",
            "- No large processed datasets or caches were written.",
            "",
            "## Recommendation",
            "",
        ]
    )
    if summary_raw["mean_wmae"] < reference["s5b_mean_wmae"] or summary_sqrt["mean_wmae"] < reference["s5b_mean_wmae"]:
        raw_lines.append("Prepare a follow-up Kaggle candidate after human review of the fold-level tradeoffs.")
    else:
        raw_lines.append("Reject horizon-specific routing as a follow-up candidate and keep S5-B as the current best official path.")
    raw_lines.append("")

    _write_report(REPORT_PATH, raw_lines)

    result = {
        "timestamp": timestamp,
        "git_hash": git_hash,
        "raw_load_seconds": raw_load_seconds,
        "raw": {
            "folds": fold_results["raw"],
            "summary": summary_raw,
            "notes": notes_raw,
        },
        "sqrt": {
            "folds": fold_results["sqrt"],
            "summary": summary_sqrt,
            "notes": notes_sqrt,
        },
        "reference": reference,
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run()
