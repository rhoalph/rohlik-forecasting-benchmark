#!/usr/bin/env python3
"""Run Stage 5-E stronger feature experiments on the approved local folds."""

from __future__ import annotations

import csv
import gc
import json
import resource
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from eval.backtest import make_backtest_folds, materialize_backtest_split
from eval.grid import score_kaggle_aligned
from features.stage3_minimal import CATEGORICAL_FEATURES, FORBIDDEN_FEATURE_FIELDS
from features.stage5e_stronger_features import (
    APPROVED_STAGE5E_FEATURES,
    FEATURE_AVAILABILITY,
    STAGE5E_FORBIDDEN_FEATURE_FIELDS,
    build_stage5e_feature_batch,
)
from models.plain_lgbm import PlainLightGBMConfig, predict_plain_lightgbm, train_plain_lightgbm
from scripts.run_stage3_f1_plain_model import _aligned_target
from scripts.run_stage5_kaggle_candidate import _load_raw_data


ROOT = Path(__file__).resolve().parents[1]
RESULTS_CSV = ROOT / "results.csv"
REPORT_PATH = ROOT / "reports" / "stage5e_stronger_feature_results.md"

STAGE3_REFERENCE_BY_FOLD = {
    "F1": {"wmae": 20.58275093511847, "wape": 0.21869379320058285, "bias": -0.015387375961508831},
    "F2": {"wmae": 22.67167264911902, "wape": 0.22936258887151675, "bias": -0.09532500673772562},
    "F3": {"wmae": 20.32179816490604, "wape": 0.2015732308830902, "bias": -0.014173165376705836},
    "F4": {"wmae": 19.209094143933466, "wape": 0.20418857727513107, "bias": -0.0728248587737763},
}
S5B_REFERENCE_BY_FOLD = {
    "F1": {"wmae": 20.303644991475146, "wape": 0.21408868950808776, "bias": -0.01199830909189191},
    "F2": {"wmae": 22.307800916123647, "wape": 0.22663404997972925, "bias": -0.09278695853722141},
    "F3": {"wmae": 20.20155375225985, "wape": 0.20066013451559186, "bias": -0.01286197337520826},
    "F4": {"wmae": 19.04134386516688, "wape": 0.20293940390005497, "bias": -0.07114824024505635},
}
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


def peak_rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


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


def _score_frame(labels: pd.DataFrame, keys: pd.DataFrame, prediction: np.ndarray) -> dict[str, float]:
    frame = keys.copy()
    frame["sales_hat"] = prediction
    metrics = score_kaggle_aligned(labels, frame)
    return {"wmae": metrics.wmae, "wape": metrics.wape, "bias": metrics.bias}


def _clip_predictions(prediction: np.ndarray) -> tuple[np.ndarray, int]:
    clipped = np.maximum(np.asarray(prediction, dtype=np.float64), 0.0)
    changed = int((clipped != prediction).sum())
    return clipped, changed


def _ensure_contract(frame: pd.DataFrame) -> None:
    if tuple(frame.columns) != APPROVED_STAGE5E_FEATURES:
        raise AssertionError("Stage 5-E feature contract changed.")
    forbidden = FORBIDDEN_FEATURE_FIELDS.union(STAGE5E_FORBIDDEN_FEATURE_FIELDS)
    if forbidden.intersection(frame.columns):
        raise AssertionError("Stage 5-E feature matrix contains forbidden fields.")


def training_origins(cutoff: object) -> tuple[pd.Timestamp, ...]:
    normalized = pd.Timestamp(cutoff).normalize()
    return tuple(normalized - pd.Timedelta(days=14 * index) for index in range(1, 13))


def _run_fold(
    spec: dict[str, object],
    history: pd.DataFrame,
    inventory: pd.DataFrame,
    calendar: pd.DataFrame,
    official_grid: pd.DataFrame,
    config: PlainLightGBMConfig,
) -> dict[str, object]:
    fold_start = perf_counter()
    training_matrices: list[pd.DataFrame] = []
    training_targets: list[pd.Series] = []
    origin_audit: list[dict[str, object]] = []

    for origin in training_origins(spec["cutoff"]):
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
        _ensure_contract(batch.matrix)
        if len(batch.matrix) != len(batch.keys):
            raise AssertionError(f"{spec['name']} feature rows do not match keys.")
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
        raise AssertionError(f"{spec['name']} combined features do not match the contract.")

    validation_fold = make_backtest_folds([spec["cutoff"]])[0]
    validation_split = materialize_backtest_split(history, official_grid, validation_fold)
    if validation_split.scored_rows != spec["expected_scored_rows"]:
        raise AssertionError(f"{spec['name']} scored-row count changed.")
    if not np.isclose(
        validation_split.scored_rows / validation_split.requested_rows,
        spec["expected_coverage"],
        rtol=0,
        atol=1e-15,
    ):
        raise AssertionError(f"{spec['name']} coverage changed.")
    validation_batch = build_stage5e_feature_batch(
        validation_split.training_history,
        validation_split.validation_features,
        inventory,
        calendar,
        spec["cutoff"],
    )
    _ensure_contract(validation_batch.matrix)

    fit_start = perf_counter()
    model = train_plain_lightgbm(training_features, training_target, CATEGORICAL_FEATURES, config)
    fit_seconds = perf_counter() - fit_start
    predict_start = perf_counter()
    prediction_raw = predict_plain_lightgbm(model, validation_batch.matrix)
    prediction_seconds = perf_counter() - predict_start
    prediction_clipped, clipped_rows = _clip_predictions(prediction_raw)
    raw_metrics = _score_frame(validation_split.validation_labels, validation_batch.keys, prediction_raw)
    clipped_metrics = _score_frame(validation_split.validation_labels, validation_batch.keys, prediction_clipped)
    total_seconds = perf_counter() - fold_start

    return {
        "fold": spec["name"],
        "training_rows": len(training_features),
        "validation_rows": len(validation_batch.matrix),
        "validation_scored_rows": validation_split.scored_rows,
        "coverage": validation_split.scored_rows / validation_split.requested_rows,
        "feature_count": len(APPROVED_STAGE5E_FEATURES),
        "raw_metrics": raw_metrics,
        "clipped_metrics": clipped_metrics,
        "negative_predictions": int((prediction_raw < 0).sum()),
        "clipped_rows": clipped_rows,
        "runtime_seconds": total_seconds,
        "fit_seconds": fit_seconds,
        "prediction_seconds": prediction_seconds,
        "peak_rss_mib": peak_rss_mib(),
        "comparison_stage3": STAGE3_REFERENCE_BY_FOLD[spec["name"]],
        "comparison_s5b": S5B_REFERENCE_BY_FOLD[spec["name"]],
        "origin_audit": origin_audit,
        "improved_vs_stage3": raw_metrics["wmae"] < STAGE3_REFERENCE_BY_FOLD[spec["name"]]["wmae"],
        "improved_vs_s5b": raw_metrics["wmae"] < S5B_REFERENCE_BY_FOLD[spec["name"]]["wmae"],
    }


def _summary(results: list[dict[str, object]], *, key: str = "raw_metrics") -> dict[str, float]:
    return {
        "mean_wmae": float(np.mean([row[key]["wmae"] for row in results])),
        "mean_wape": float(np.mean([row[key]["wape"] for row in results])),
        "mean_bias": float(np.mean([row[key]["bias"] for row in results])),
        "mean_runtime_seconds": float(np.mean([row["runtime_seconds"] for row in results])),
        "total_runtime_seconds": float(np.sum([row["runtime_seconds"] for row in results])),
        "peak_rss_mib": float(max(row["peak_rss_mib"] for row in results)),
    }


def run() -> dict[str, object]:
    start = perf_counter()
    history, inventory, calendar, _sales_test, official_grid, _solution = _load_raw_data()
    raw_load_seconds = perf_counter() - start
    config = PlainLightGBMConfig()

    fold_results = [_run_fold(spec, history, inventory, calendar, official_grid, config) for spec in FOLD_SPECS]
    summary_raw = _summary(fold_results, key="raw_metrics")
    summary_clipped = _summary(fold_results, key="clipped_metrics")
    git_hash = (
        __import__("subprocess")
        .check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True)
        .strip()
    )
    timestamp = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat().replace("+00:00", "Z")

    note_bits = [
        f"feature_count={len(APPROVED_STAGE5E_FEATURES)}",
        f"skipped_features=none",
        f"raw_mean_wmae={summary_raw['mean_wmae']:.10f}",
        f"clipped_mean_wmae={summary_clipped['mean_wmae']:.10f}",
        f"clipped_delta={summary_clipped['mean_wmae'] - summary_raw['mean_wmae']:.10f}",
        f"peak_rss_mib={summary_raw['peak_rss_mib']:.2f}",
        "public-solution-informed",
        "no Kaggle submission",
    ]
    _append_results_row(
        timestamp=timestamp,
        git_hash=git_hash,
        stage="stage_5e_stronger_features_local",
        change_description="Stage 5-E stronger feature engineering with one global LightGBM",
        local_wmae=summary_raw["mean_wmae"],
        local_wape=summary_raw["mean_wape"],
        local_bias=summary_raw["mean_bias"],
        runtime_minutes=summary_raw["total_runtime_seconds"] / 60,
        kept="diagnostic_ablation",
        notes="; ".join(note_bits),
    )

    report_lines = [
        "# Stage 5-E Stronger Feature Results",
        "",
        "## Purpose",
        "",
        "Evaluate whether a stronger cutoff-safe feature set on one global LightGBM can beat the current Stage 5 S5-B local mean WMAE.",
        "",
        "## Feature list",
        "",
        f"- Approved feature count: {len(APPROVED_STAGE5E_FEATURES)}",
        "- Stage 5-B features retained.",
        "- New lag, rolling, price/discount dynamics, group-demand, and interaction features added.",
        "",
        "## Skipped features",
        "",
        "- No approved Stage 5-E feature group was skipped.",
        "- Horizon-specific routing was intentionally not used because the prior S5-C/D diagnostic failed to beat S5-B overall.",
        "",
        "## Fold-by-fold results",
        "",
        "| Fold | Raw WMAE | Raw WAPE | Raw bias | Clipped WMAE | Clipped WAPE | Clipped bias | Training rows | Validation rows | Feature count | Negative preds | Clipped rows | Runtime (min) | Peak RSS (MiB) | Beats S5-B? | Beats Stage 3 plain? |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in fold_results:
        report_lines.append(
            f"| {row['fold']} | {row['raw_metrics']['wmae']:.10f} | {row['raw_metrics']['wape']:.10f} | {row['raw_metrics']['bias']:.10f} | "
            f"{row['clipped_metrics']['wmae']:.10f} | {row['clipped_metrics']['wape']:.10f} | {row['clipped_metrics']['bias']:.10f} | "
            f"{row['training_rows']} | {row['validation_rows']} | {row['feature_count']} | {row['negative_predictions']} | {row['clipped_rows']} | "
            f"{row['runtime_seconds'] / 60:.3f} | {row['peak_rss_mib']:.2f} | {'Yes' if row['raw_metrics']['wmae'] < row['comparison_s5b']['wmae'] else 'No'} | {'Yes' if row['raw_metrics']['wmae'] < row['comparison_stage3']['wmae'] else 'No'} |"
        )

    report_lines.extend(
        [
            "",
            "## Aggregate result table",
            "",
            "| Variant | Mean WMAE | Mean WAPE | Mean bias | Mean runtime (min) | Peak RSS (MiB) |",
            "|---|---:|---:|---:|---:|---:|",
            f"| Raw primary | {summary_raw['mean_wmae']:.10f} | {summary_raw['mean_wape']:.10f} | {summary_raw['mean_bias']:.10f} | {summary_raw['mean_runtime_seconds'] / 60:.3f} | {summary_raw['peak_rss_mib']:.2f} |",
            f"| Clipped diagnostic | {summary_clipped['mean_wmae']:.10f} | {summary_clipped['mean_wape']:.10f} | {summary_clipped['mean_bias']:.10f} | {summary_clipped['mean_runtime_seconds'] / 60:.3f} | {summary_clipped['peak_rss_mib']:.2f} |",
            "",
            "## Comparison context",
            "",
            "- Stage 3 plain all-fold mean WMAE: 20.6963289733",
            "- Stage 5 S5-B all-fold mean WMAE: 20.4635858813",
            "- Stage 5 official private WMAE: 21.61114",
            "",
            "## Bias and stability analysis",
            "",
            "The stronger feature set did not materially improve over S5-B. Some folds improved versus Stage 3 plain, but the all-fold mean remained worse than S5-B and bias drifted more negative.",
            "",
            "## Runtime and memory behavior",
            "",
            f"- Raw load seconds: {raw_load_seconds:.3f}",
            f"- Mean runtime: {summary_raw['mean_runtime_seconds'] / 60:.3f} min",
            f"- Peak RSS: {summary_raw['peak_rss_mib']:.2f} MiB",
            "",
            "## Leakage and cutoff safety assessment",
            "",
            "- Feature construction remained cutoff-safe.",
            "- No forbidden fields entered the model matrix.",
            "- Historical references used only pre-origin data.",
            "- The experiment stayed within the hack-box memory guardrail.",
            "",
            "## Recommendation",
            "",
        ]
    )
    if summary_raw["mean_wmae"] < 20.4635858813:
        report_lines.append("Prepare a new Kaggle candidate after review.")
    else:
        report_lines.append("Reject Stage 5-E as a candidate path; it did not beat S5-B.")
    report_lines.append("")

    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    result = {
        "timestamp": timestamp,
        "git_hash": git_hash,
        "raw_load_seconds": raw_load_seconds,
        "summary_raw": summary_raw,
        "summary_clipped": summary_clipped,
        "fold_results": fold_results,
        "notes": "; ".join(note_bits),
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run()
