#!/usr/bin/env python3
"""Run Stage 5 clipping and relative price/discount diagnostics."""

from __future__ import annotations

import gc
import csv
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
    build_stage3_feature_batch,
)
from features.stage5_price_discount import (
    APPROVED_STAGE5_FEATURES,
    FEATURE_AVAILABILITY,
    PRICE_DISCOUNT_FEATURES,
    build_stage5_feature_batch,
)
from models.plain_lgbm import (
    PlainLightGBMConfig,
    predict_plain_lightgbm,
    train_plain_lightgbm,
)
from scripts.run_stage3_all_folds_plain_model import (
    FOLD_SPECS,
    training_origins_for_fold,
)
from scripts.run_stage3_f1_plain_model import _aligned_target, _load_raw_data


ROOT = Path(__file__).resolve().parents[1]
RESULTS_CSV = ROOT / "results.csv"
REPORT_S5A = ROOT / "reports" / "stage5_s5a_clipping_diagnostic.md"
REPORT_S5B = ROOT / "reports" / "stage5_s5b_relative_price_discount_results.md"
STAGE4_CANDIDATE = ROOT / "submissions" / "stage4_plain_lgbm_candidate.csv"
STAGE4_OFFICIAL_PUBLIC = 22.37834
STAGE4_OFFICIAL_PRIVATE = 21.91884
STAGE3_ALL_FOLDS_MEAN_WMAE = 20.6963289733
STAGE2_ALL_FOLDS_MEAN_WMAE = 30.564416659337184

F1_SPEC = {
    "name": "F1",
    "cutoff": pd.Timestamp("2024-05-19"),
    "validation_start": pd.Timestamp("2024-05-20"),
    "validation_end": pd.Timestamp("2024-06-02"),
    "expected_scored_rows": 44212,
    "expected_coverage": 0.9402607345654069,
    "baseline_wmae": 31.19858033084802,
    "baseline_wape": 0.30211101284954356,
    "baseline_bias": 0.030395766081574085,
}
STAGE5_FOLD_SPECS = (F1_SPEC, *FOLD_SPECS)


def peak_rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


@dataclass(frozen=True)
class ExperimentResult:
    fold: str
    wmae: float
    wape: float
    bias: float
    runtime_seconds: float
    peak_rss_mib: float
    negative_predictions: int
    comparison_to_stage3_wmae: float
    comparison_to_stage3_bias: float


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


def _score_frame(labels: pd.DataFrame, keys: pd.DataFrame, prediction: np.ndarray) -> tuple[float, float, float]:
    frame = keys.copy()
    frame["sales_hat"] = prediction
    metrics = score_kaggle_aligned(labels, frame)
    return metrics.wmae, metrics.wape, metrics.bias


def _run_experiment(
    *,
    history: pd.DataFrame,
    inventory: pd.DataFrame,
    calendar: pd.DataFrame,
    official_grid: pd.DataFrame,
    feature_builder,
    clip_predictions: bool = False,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    config = PlainLightGBMConfig()
    fold_rows: list[dict[str, object]] = []
    start = perf_counter()

    for spec in STAGE5_FOLD_SPECS:
        fold_start = perf_counter()
        training_matrices: list[pd.DataFrame] = []
        training_targets: list[pd.Series] = []
        for origin in training_origins_for_fold(spec["cutoff"]):
            fold = make_backtest_folds([origin])[0]
            split = materialize_backtest_split(history, official_grid, fold)
            batch = feature_builder(
                split.training_history,
                split.validation_features,
                inventory,
                calendar,
                origin,
            )
            if feature_builder is build_stage3_feature_batch and tuple(batch.matrix.columns) != APPROVED_FEATURES:
                raise AssertionError(f"{spec.name} stage3 features differ from the approved contract.")
            if feature_builder is build_stage5_feature_batch and tuple(batch.matrix.columns) != APPROVED_STAGE5_FEATURES:
                raise AssertionError(f"{spec.name} stage5 features differ from the approved contract.")
            if FORBIDDEN_FEATURE_FIELDS.intersection(batch.matrix.columns):
                raise AssertionError(f"{spec.name} feature matrix contains forbidden fields.")
            training_matrices.append(batch.matrix)
            training_targets.append(_aligned_target(batch.keys, split.validation_labels))
            del split, batch
            gc.collect()

        training_features = pd.concat(training_matrices, ignore_index=True)
        training_target = pd.concat(training_targets, ignore_index=True)
        validation_split = materialize_backtest_split(history, official_grid, make_backtest_folds([spec["cutoff"]])[0])
        validation_batch = feature_builder(
            validation_split.training_history,
            validation_split.validation_features,
            inventory,
            calendar,
            spec["cutoff"],
        )
        model = train_plain_lightgbm(
            training_features,
            training_target,
            CATEGORICAL_FEATURES,
            config,
        )
        prediction = predict_plain_lightgbm(model, validation_batch.matrix)
        if clip_predictions:
            prediction = np.maximum(prediction, 0.0)
        wmae, wape, bias = _score_frame(validation_split.validation_labels, validation_batch.keys, prediction)
        fold_rows.append(
            {
                "fold": spec["name"],
                "wmae": wmae,
                "wape": wape,
                "bias": bias,
                "runtime_seconds": perf_counter() - fold_start,
                "peak_rss_mib": peak_rss_mib(),
                "negative_predictions": int((prediction < 0).sum()),
                "comparison_to_stage3_wmae": None,
                "comparison_to_stage3_bias": None,
                "training_rows": len(training_features),
                "validation_rows": len(validation_batch.matrix),
                "coverage": validation_split.scored_rows / validation_split.requested_rows,
                "stage2_baseline_wmae": spec["baseline_wmae"],
                "stage2_baseline_wape": spec["baseline_wape"],
                "stage2_baseline_bias": spec["baseline_bias"],
                "improved_vs_stage2": wmae < spec["baseline_wmae"],
            }
        )

    summary = {
        "mean_wmae": float(np.mean([row["wmae"] for row in fold_rows])),
        "mean_wape": float(np.mean([row["wape"] for row in fold_rows])),
        "mean_bias": float(np.mean([row["bias"] for row in fold_rows])),
        "mean_runtime_seconds": float(np.mean([row["runtime_seconds"] for row in fold_rows])),
        "total_runtime_seconds": perf_counter() - start,
        "peak_rss_mib": peak_rss_mib(),
    }
    return fold_rows, summary


def _candidate_clipping_inspection() -> dict[str, object]:
    candidate = pd.read_csv(STAGE4_CANDIDATE)
    negatives = candidate.loc[candidate["sales_hat"] < 0, "sales_hat"]
    clipped = candidate["sales_hat"].clip(lower=0)
    changed = int((clipped != candidate["sales_hat"]).sum())
    return {
        "rows": len(candidate),
        "negative_count": int(len(negatives)),
        "min_negative_value": float(negatives.min()) if not negatives.empty else None,
        "changed_rows_when_clipped": changed,
        "min_after_clipping": float(clipped.min()),
        "max_after_clipping": float(clipped.max()),
        "mean_after_clipping": float(clipped.mean()),
        "median_after_clipping": float(clipped.median()),
    }


def _write_report(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _stage3_mean_from_results() -> float:
    rows = pd.read_csv(RESULTS_CSV)
    stage3 = rows.loc[rows["stage"] == "stage_3_plain_model_fold", "local_wmae"].astype(float)
    return float(stage3.mean())


def _stage3_fold_wmae_from_results() -> dict[str, float]:
    rows = pd.read_csv(RESULTS_CSV)
    mapping: dict[str, float] = {}
    for _, row in rows.loc[rows["stage"] == "stage_3_plain_model_fold", ["change_description", "local_wmae"]].iterrows():
        description = str(row["change_description"])
        if description.startswith("F2 "):
            mapping["F2"] = float(row["local_wmae"])
        elif description.startswith("F3 "):
            mapping["F3"] = float(row["local_wmae"])
        elif description.startswith("F4 "):
            mapping["F4"] = float(row["local_wmae"])
    f1 = rows.loc[rows["stage"] == "stage_3_f1", "local_wmae"].astype(float)
    if not f1.empty:
        mapping["F1"] = float(f1.iloc[0])
    return mapping


def run() -> dict[str, object]:
    start = perf_counter()
    history, inventory, calendar, official_grid = _load_raw_data()
    raw_load_seconds = perf_counter() - start
    candidate_clip = _candidate_clipping_inspection()

    clip_rows, clip_summary = _run_experiment(
        history=history,
        inventory=inventory,
        calendar=calendar,
        official_grid=official_grid,
        feature_builder=build_stage3_feature_batch,
        clip_predictions=False,
    )
    clip_clipped_rows, clip_clipped_summary = _run_experiment(
        history=history,
        inventory=inventory,
        calendar=calendar,
        official_grid=official_grid,
        feature_builder=build_stage3_feature_batch,
        clip_predictions=True,
    )
    price_rows, price_summary = _run_experiment(
        history=history,
        inventory=inventory,
        calendar=calendar,
        official_grid=official_grid,
        feature_builder=build_stage5_feature_batch,
        clip_predictions=False,
    )

    stage3_mean_wmae = _stage3_mean_from_results()
    stage3_fold_wmae = _stage3_fold_wmae_from_results()
    git_hash = (
        __import__("subprocess")
        .check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True)
        .strip()
    )
    timestamp = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat().replace("+00:00", "Z")

    s5a_notes = (
        f"S5-A diagnostic; public-solution-informed clipping check; no Kaggle submission; "
        f"candidate_rows={candidate_clip['rows']}; candidate_negative_count={candidate_clip['negative_count']}; "
        f"candidate_changed_rows={candidate_clip['changed_rows_when_clipped']}; "
        f"candidate_min_negative={candidate_clip['min_negative_value']}; "
        f"local_unclipped_mean_wmae={clip_summary['mean_wmae']:.10f}; "
        f"local_clipped_mean_wmae={clip_clipped_summary['mean_wmae']:.10f}; "
        f"local_validation_was_experimented_with_but_not_submitted"
    )
    s5b_notes = (
        f"S5-B local validation; public-solution-informed relative price/discount features; "
        f"feature_count={len(APPROVED_STAGE5_FEATURES)}; price_discount_features={len(PRICE_DISCOUNT_FEATURES)}; "
        f"stage3_mean_wmae={stage3_mean_wmae:.10f}; local_mean_wmae={price_summary['mean_wmae']:.10f}; "
        f"local_mean_bias={price_summary['mean_bias']:.10f}; no Kaggle submission"
    )

    _append_results_row(
        timestamp=timestamp,
        git_hash=git_hash,
        stage="stage_5_s5a_clipping_diagnostic",
        change_description="S5-A clipping diagnostic for Stage 4 candidate and local folds",
        local_wmae=clip_clipped_summary["mean_wmae"],
        local_wape=clip_clipped_summary["mean_wape"],
        local_bias=clip_clipped_summary["mean_bias"],
        runtime_minutes=clip_clipped_summary["total_runtime_seconds"] / 60,
        kept="diagnostic_ablation",
        notes=s5a_notes,
    )
    _append_results_row(
        timestamp=timestamp,
        git_hash=git_hash,
        stage="stage_5_s5b_relative_price_discount_local",
        change_description="S5-B relative price/discount feature experiment",
        local_wmae=price_summary["mean_wmae"],
        local_wape=price_summary["mean_wape"],
        local_bias=price_summary["mean_bias"],
        runtime_minutes=price_summary["total_runtime_seconds"] / 60,
        kept="diagnostic_ablation",
        notes=s5b_notes,
    )

    s5a_report = [
        "# Stage 5 S5-A Clipping Diagnostic",
        "",
        "## Purpose",
        "",
        "Assess whether clipping negative predictions to zero changes local validation or the prepared Stage 4 candidate.",
        "",
        "## Local fold comparison",
        "",
        "| Fold | Unclipped WMAE | Clipped WMAE | Unclipped WAPE | Clipped WAPE | Unclipped bias | Clipped bias | Negatives before | Negatives after |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for unclipped, clipped in zip(clip_rows, clip_clipped_rows, strict=True):
        s5a_report.append(
            f"| {unclipped['fold']} | {unclipped['wmae']:.10f} | {clipped['wmae']:.10f} | "
            f"{unclipped['wape']:.10f} | {clipped['wape']:.10f} | {unclipped['bias']:.10f} | {clipped['bias']:.10f} | "
            f"{unclipped['negative_predictions']} | {clipped['negative_predictions']} |"
        )
    s5a_report.extend(
        [
            "",
            "## Stage 4 candidate inspection",
            "",
            f"- rows: {candidate_clip['rows']}",
            f"- negative predictions: {candidate_clip['negative_count']}",
            f"- min negative value: {candidate_clip['min_negative_value']}",
            f"- rows changed by clipping: {candidate_clip['changed_rows_when_clipped']}",
            f"- min after clipping: {candidate_clip['min_after_clipping']}",
            f"- max after clipping: {candidate_clip['max_after_clipping']}",
            f"- mean after clipping: {candidate_clip['mean_after_clipping']}",
            f"- median after clipping: {candidate_clip['median_after_clipping']}",
            "",
            "## Assessment",
            "",
            (
                "Clipping is likely low-risk and may be mildly beneficial if the clipped fold mean WMAE is lower. "
                "If the change is negligible, the unmodified score path remains easier to explain."
            ),
            "",
            "## Submission status",
            "",
            "No Kaggle submission was made for the clipped diagnostic.",
        ]
    )

    s5b_report = [
        "# Stage 5 S5-B Relative Price and Discount Feature Results",
        "",
        "## Public-solution idea basis",
        "",
        "The feature family is motivated by public competition writeups that emphasized price/discount signal, relative pricing, and engineered demand context, plus Stage 3 ablations showing that price/discount materially contributes to score.",
        "",
        "## Feature list",
        "",
    ]
    for feature in PRICE_DISCOUNT_FEATURES:
        s5b_report.append(f"- {feature}")
    s5b_report.extend(
        [
            "",
            "## Feature availability and cutoff logic",
            "",
            "| Feature | Availability | Cutoff logic |",
            "|---|---|---|",
        ]
    )
    for feature in PRICE_DISCOUNT_FEATURES:
        s5b_report.append(
            f"| {feature} | {FEATURE_AVAILABILITY[feature]} | Uses only historical references through each origin or the final cutoff; request-side price/discount values remain benchmark-known-future covariates |"
        )
    s5b_report.extend(
        [
            "",
            "## Fold results",
            "",
            "| Fold | WMAE | WAPE | Bias | Runtime (min) | Peak RSS (MiB) | Negative predictions | Beats Stage 3 plain? |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in price_rows:
        stage3_wmae = stage3_fold_wmae.get(row["fold"], float("nan"))
        s5b_report.append(
            f"| {row['fold']} | {row['wmae']:.10f} | {row['wape']:.10f} | {row['bias']:.10f} | "
            f"{row['runtime_seconds'] / 60:.3f} | {row['peak_rss_mib']:.2f} | {row['negative_predictions']} | "
            f"{'Yes' if row['wmae'] < stage3_wmae else 'No'} |"
        )
    s5b_report.extend(
        [
            "",
            "## Comparison to Stage 3 plain model",
            "",
            f"- Stage 3 plain all-fold mean WMAE: {stage3_mean_wmae:.10f}",
            f"- Stage 5 price/discount mean WMAE: {price_summary['mean_wmae']:.10f}",
            f"- Mean WMAE delta: {price_summary['mean_wmae'] - stage3_mean_wmae:.10f}",
            f"- Stage 5 price/discount mean bias: {price_summary['mean_bias']:.10f}",
            f"- Stage 4 official private WMAE (context only): {STAGE4_OFFICIAL_PRIVATE:.5f}",
            f"- Stage 2 all-fold mean WMAE (floor context): {STAGE2_ALL_FOLDS_MEAN_WMAE:.10f}",
            "",
            "## Bias and stability assessment",
            "",
            "The feature batch should be considered useful only if it improves WMAE without making bias materially worse across folds.",
            "",
            "## Leakage risk assessment",
            "",
            "Main risk is accidental leakage through price/discount reference statistics or price ranks. The implementation keeps all historical statistics cutoff-safe and excludes target, total_orders, availability, weights, ids, and solution labels.",
            "",
            "## Runtime and memory",
            "",
            f"- Total runtime: {price_summary['total_runtime_seconds'] / 60:.3f} min",
            f"- Peak RSS: {price_summary['peak_rss_mib']:.2f} MiB",
            "",
            "## Recommendation",
            "",
        ]
    )
    if price_summary["mean_wmae"] < stage3_mean_wmae:
        s5b_report.append("Prepare a Kaggle candidate after review.")
    else:
        s5b_report.append("Revise and rerun or run more diagnostics first.")
    s5b_report.append("")

    _write_report(REPORT_S5A, s5a_report)
    _write_report(REPORT_S5B, s5b_report)

    return {
        "timestamp": timestamp,
        "git_hash": git_hash,
        "raw_load_seconds": raw_load_seconds,
        "s5a": {
            "candidate_clip": candidate_clip,
            "unclipped": clip_rows,
            "clipped": clip_clipped_rows,
            "summary": clip_clipped_summary,
            "notes": s5a_notes,
        },
        "s5b": {
            "rows": price_rows,
            "summary": price_summary,
            "notes": s5b_notes,
        },
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
