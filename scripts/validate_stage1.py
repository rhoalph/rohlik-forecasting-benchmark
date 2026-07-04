"""Validate the Stage 1 grid contract and split coverage on the raw CSVs."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import pandas as pd

from eval.backtest import make_backtest_folds, materialize_backtest_split
from eval.grid import build_official_test_grid, validate_solution_template


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def main() -> None:
    sales_test = pd.read_csv(RAW / "sales_test.csv", usecols=["unique_id", "date"])
    weights = pd.read_csv(RAW / "test_weights.csv")
    solution = pd.read_csv(RAW / "solution.csv")
    history = pd.read_csv(
        RAW / "sales_train.csv",
        usecols=["unique_id", "date", "sales"],
    )

    grid = build_official_test_grid(sales_test, weights)
    validate_solution_template(grid, solution)

    result = {
        "official_grid_rows": len(grid),
        "official_grid_ids": int(grid["unique_id"].nunique()),
        "official_horizon_days": sorted(int(day) for day in grid["horizon_day"].unique()),
        "solution_template_exact_match": True,
        "folds": [],
    }
    for fold in make_backtest_folds():
        split = materialize_backtest_split(history, grid, fold)
        result["folds"].append(
            {
                "fold": fold.name,
                "cutoff": str(fold.cutoff.date()),
                "validation_start": str(fold.validation_start.date()),
                "validation_end": str(fold.validation_end.date()),
                "training_rows": len(split.training_history),
                "training_ids": int(split.training_history["unique_id"].nunique()),
                "requested_rows": split.requested_rows,
                "scored_rows": split.scored_rows,
                "missing_label_rows": split.missing_label_rows,
                "coverage": split.scored_rows / split.requested_rows,
                "excluded_training_missing_targets": split.excluded_training_missing_targets,
            }
        )
        del split
        gc.collect()

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
