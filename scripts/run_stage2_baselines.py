#!/usr/bin/env python3
"""Run Stage 2 naive baselines sequentially on all four approved folds."""

from __future__ import annotations

import gc
import json
import resource
from collections import defaultdict
from pathlib import Path
from statistics import mean
from time import perf_counter

import pandas as pd

from baselines.naive import BASELINES, prepare_context
from eval.backtest import DEFAULT_CUTOFFS, make_backtest_folds, materialize_backtest_split
from eval.grid import build_official_test_grid, score_kaggle_aligned


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def peak_rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main() -> None:
    total_start = perf_counter()
    load_start = perf_counter()
    history = pd.read_csv(
        RAW / "sales_train.csv",
        usecols=["unique_id", "date", "sales"],
        dtype={"unique_id": "uint16", "sales": "float64"},
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
    gc.collect()
    shared_load_seconds = perf_counter() - load_start

    folds_output: list[dict[str, object]] = []
    metrics_by_baseline: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for fold in make_backtest_folds(DEFAULT_CUTOFFS):
        fold_start = perf_counter()
        preparation_start = perf_counter()
        split = materialize_backtest_split(history, official_grid, fold)
        labels = split.validation_labels.loc[
            :, ["unique_id", "date", "sales", "weight"]
        ].copy()
        context = prepare_context(
            split.training_history,
            split.validation_features.loc[:, ["unique_id", "date"]],
            fold.cutoff,
            horizon_days=fold.horizon_days,
        )
        requested_rows = split.requested_rows
        scored_rows = split.scored_rows
        missing_label_rows = split.missing_label_rows
        excluded_training_missing_targets = split.excluded_training_missing_targets
        coverage = scored_rows / requested_rows
        del split
        gc.collect()
        preparation_seconds = perf_counter() - preparation_start

        baseline_results: list[dict[str, object]] = []
        for name, baseline in BASELINES:
            baseline_start = perf_counter()
            output = baseline(context)
            metrics = score_kaggle_aligned(labels, output.predictions)
            runtime_seconds = perf_counter() - baseline_start
            row = {
                "baseline": name,
                "wmae": metrics.wmae,
                "wape": metrics.wape,
                "bias": metrics.bias,
                "runtime_seconds": runtime_seconds,
                "runtime_minutes": runtime_seconds / 60,
                "scored_rows": metrics.rows,
                "requested_rows": requested_rows,
                "coverage": coverage,
                "primary_fallback_rows": output.primary_fallback_rows,
                "global_fallback_rows": output.global_fallback_rows,
            }
            baseline_results.append(row)
            for metric_name in ("wmae", "wape", "bias", "runtime_seconds"):
                metrics_by_baseline[name][metric_name].append(float(row[metric_name]))
            del output, metrics
            gc.collect()

        folds_output.append(
            {
                "fold": fold.name,
                "cutoff": str(fold.cutoff.date()),
                "validation_start": str(fold.validation_start.date()),
                "validation_end": str(fold.validation_end.date()),
                "training_rows": len(context.training),
                "training_ids": int(context.training["unique_id"].nunique()),
                "requested_rows": requested_rows,
                "scored_rows": scored_rows,
                "missing_label_rows": missing_label_rows,
                "coverage": coverage,
                "excluded_training_missing_targets": excluded_training_missing_targets,
                "global_training_median": context.global_median,
                "fold_preparation_seconds": preparation_seconds,
                "fold_total_seconds": perf_counter() - fold_start,
                "baselines": baseline_results,
            }
        )
        del context, labels, baseline_results
        gc.collect()

    averages = []
    for name, _ in BASELINES:
        values = metrics_by_baseline[name]
        averages.append(
            {
                "baseline": name,
                "mean_wmae": mean(values["wmae"]),
                "mean_wape": mean(values["wape"]),
                "mean_bias": mean(values["bias"]),
                "mean_runtime_seconds": mean(values["runtime_seconds"]),
            }
        )
    averages.sort(key=lambda row: float(row["mean_wmae"]))
    for rank, row in enumerate(averages, start=1):
        row["wmae_rank"] = rank

    result = {
        "stage": "stage_2_all_fold_naive_baselines",
        "cutoffs": list(DEFAULT_CUTOFFS),
        "shared_load_seconds": shared_load_seconds,
        "total_runtime_seconds": perf_counter() - total_start,
        "peak_process_rss_mib": peak_rss_mib(),
        "folds": folds_output,
        "macro_average_by_baseline": averages,
        "processed_data_written": False,
        "kaggle_submission_made": False,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
