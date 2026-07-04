#!/usr/bin/env python3
"""Run approved Stage 2 naive baselines on F1 only and print JSON results."""

from __future__ import annotations

import gc
import json
import resource
from pathlib import Path
from time import perf_counter

import pandas as pd

from baselines.naive import BASELINES, prepare_context
from eval.backtest import make_backtest_folds, materialize_backtest_split
from eval.grid import build_official_test_grid, score_kaggle_aligned


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
F1_CUTOFF = "2024-05-19"


def peak_rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main() -> None:
    total_start = perf_counter()
    preparation_start = perf_counter()

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
    fold = make_backtest_folds([F1_CUTOFF])[0]
    split = materialize_backtest_split(history, official_grid, fold)
    del history, sales_test, weights, official_grid
    gc.collect()

    labels = split.validation_labels.loc[:, ["unique_id", "date", "sales", "weight"]].copy()
    context = prepare_context(
        split.training_history,
        split.validation_features.loc[:, ["unique_id", "date"]],
        fold.cutoff,
        horizon_days=fold.horizon_days,
    )
    del split.training_history, split.validation_features, split.validation_labels
    gc.collect()
    preparation_seconds = perf_counter() - preparation_start

    baseline_results: list[dict[str, object]] = []
    coverage = split.scored_rows / split.requested_rows
    for name, baseline in BASELINES:
        baseline_start = perf_counter()
        output = baseline(context)
        metrics = score_kaggle_aligned(labels, output.predictions)
        runtime_seconds = perf_counter() - baseline_start
        baseline_results.append(
            {
                "baseline": name,
                "wmae": metrics.wmae,
                "wape": metrics.wape,
                "bias": metrics.bias,
                "runtime_seconds": runtime_seconds,
                "runtime_minutes": runtime_seconds / 60,
                "scored_rows": metrics.rows,
                "requested_rows": split.requested_rows,
                "coverage": coverage,
                "primary_fallback_rows": output.primary_fallback_rows,
                "global_fallback_rows": output.global_fallback_rows,
            }
        )
        del output, metrics
        gc.collect()

    result = {
        "stage": "stage_2_f1_naive_baselines",
        "fold": fold.name,
        "cutoff": str(fold.cutoff.date()),
        "validation_start": str(fold.validation_start.date()),
        "validation_end": str(fold.validation_end.date()),
        "training_rows": len(context.training),
        "training_ids": int(context.training["unique_id"].nunique()),
        "requested_rows": split.requested_rows,
        "scored_rows": split.scored_rows,
        "missing_label_rows": split.missing_label_rows,
        "coverage": coverage,
        "excluded_training_missing_targets": split.excluded_training_missing_targets,
        "global_training_median": context.global_median,
        "shared_preparation_seconds": preparation_seconds,
        "total_runtime_seconds": perf_counter() - total_start,
        "peak_process_rss_mib": peak_rss_mib(),
        "baselines": baseline_results,
        "processed_data_written": False,
        "kaggle_submission_made": False,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
