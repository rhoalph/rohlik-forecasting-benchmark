"""Reviewed-but-not-yet-frozen evaluation interface for Stage 1."""

from eval.backtest import (
    DEFAULT_CUTOFFS,
    BacktestFold,
    BacktestSplit,
    iter_backtest_splits,
    make_backtest_folds,
    materialize_backtest_split,
)
from eval.grid import (
    OFFICIAL_TEST_ORIGIN,
    build_official_test_grid,
    score_kaggle_aligned,
    shift_official_grid,
    validate_solution_template,
)
from eval.metrics import MetricResult, bias, score_metrics, wape, wmae

__all__ = [
    "DEFAULT_CUTOFFS",
    "OFFICIAL_TEST_ORIGIN",
    "BacktestFold",
    "BacktestSplit",
    "MetricResult",
    "bias",
    "build_official_test_grid",
    "iter_backtest_splits",
    "make_backtest_folds",
    "materialize_backtest_split",
    "score_kaggle_aligned",
    "score_metrics",
    "shift_official_grid",
    "validate_solution_template",
    "wape",
    "wmae",
]
