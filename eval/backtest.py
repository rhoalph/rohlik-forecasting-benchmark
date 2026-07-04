"""Four-fold rolling-origin backtest using the official Kaggle request mask."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import pandas as pd

from dataguard.availability import assert_future_columns_allowed, select_future_covariates
from dataguard.cutoff import (
    assert_disjoint_keys,
    assert_history_at_or_before_cutoff,
    assert_target_not_present,
    assert_validation_within_window,
    filter_history_at_cutoff,
    validation_bounds,
)
from eval.grid import KEY_COLUMNS, shift_official_grid


DEFAULT_CUTOFFS = (
    "2024-05-19",
    "2024-05-05",
    "2024-04-21",
    "2024-04-07",
)


@dataclass(frozen=True)
class BacktestFold:
    """Immutable date contract for one forecast origin."""

    name: str
    cutoff: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    horizon_days: int


@dataclass
class BacktestSplit:
    """Materialized train history, safe future features, and isolated labels."""

    fold: BacktestFold
    training_history: pd.DataFrame
    validation_features: pd.DataFrame
    validation_labels: pd.DataFrame
    requested_rows: int
    scored_rows: int
    missing_label_rows: int
    excluded_training_missing_targets: int


def make_backtest_folds(
    cutoffs: Iterable[object] = DEFAULT_CUTOFFS,
    *,
    horizon_days: int = 14,
) -> tuple[BacktestFold, ...]:
    """Create validated fixed-horizon fold definitions."""

    folds: list[BacktestFold] = []
    seen: set[pd.Timestamp] = set()
    for index, cutoff in enumerate(cutoffs, start=1):
        normalized = pd.Timestamp(cutoff).normalize()
        if normalized in seen:
            raise ValueError(f"Duplicate backtest cutoff: {normalized.date()}.")
        seen.add(normalized)
        start, end = validation_bounds(normalized, horizon_days)
        folds.append(
            BacktestFold(
                name=f"F{index}",
                cutoff=normalized,
                validation_start=start,
                validation_end=end,
                horizon_days=horizon_days,
            )
        )

    for left_index, left in enumerate(folds):
        for right in folds[left_index + 1 :]:
            overlap = max(left.validation_start, right.validation_start) <= min(
                left.validation_end, right.validation_end
            )
            if overlap:
                raise ValueError(f"Validation windows overlap: {left.name} and {right.name}.")
    return tuple(folds)


def _normalized_history(history: pd.DataFrame, *, date_col: str) -> pd.DataFrame:
    required = {"unique_id", date_col}
    missing = required - set(history.columns)
    if missing:
        raise KeyError(f"history is missing columns: {sorted(missing)}.")
    normalized = history.copy()
    normalized[date_col] = pd.to_datetime(normalized[date_col], errors="coerce").dt.normalize()
    if normalized[date_col].isna().any():
        raise ValueError("history contains missing or invalid dates.")
    if normalized.loc[:, ["unique_id", date_col]].duplicated().any():
        raise ValueError("history contains duplicate unique_id/date keys.")
    return normalized


def materialize_backtest_split(
    history: pd.DataFrame,
    official_grid: pd.DataFrame,
    fold: BacktestFold,
    *,
    target_col: str = "sales",
    date_col: str = "date",
) -> BacktestSplit:
    """Build one leakage-controlled split from raw historical rows.

    Validation labels are physically separated from validation features.
    Missing labels are excluded rather than treated as zero. The returned
    future feature frame is filtered through the availability registry, so
    future total_orders, availability, target, and evaluation weight are absent.
    """

    if target_col not in history.columns:
        raise KeyError(f"history is missing target column {target_col!r}.")
    normalized = _normalized_history(history, date_col=date_col)

    unfiltered_training = filter_history_at_cutoff(
        normalized,
        fold.cutoff,
        date_col=date_col,
    )
    training_missing = int(unfiltered_training[target_col].isna().sum())
    training = unfiltered_training.dropna(subset=[target_col]).copy()
    assert_history_at_or_before_cutoff(training, fold.cutoff, date_col=date_col)

    shifted_grid = shift_official_grid(official_grid, fold.cutoff)
    lookup = normalized.rename(columns={date_col: "date"})
    validation = shifted_grid.merge(
        lookup,
        on=list(KEY_COLUMNS),
        how="left",
        validate="one_to_one",
        suffixes=("", "_history"),
    )
    label_available = validation[target_col].notna()

    history_columns = [
        "date" if column == date_col else column
        for column in history.columns
    ]
    safe_features = select_future_covariates(validation.loc[:, history_columns])
    if "horizon_day" not in safe_features.columns:
        safe_features["horizon_day"] = validation["horizon_day"]
    safe_features = safe_features.loc[label_available].reset_index(drop=True)
    labels = validation.loc[
        label_available,
        ["grid_order", "unique_id", "date", target_col, "weight"],
    ].reset_index(drop=True)

    assert_future_columns_allowed(safe_features.columns)
    assert_target_not_present(safe_features, target_col=target_col)
    assert_validation_within_window(
        safe_features,
        fold.cutoff,
        horizon_days=fold.horizon_days,
        date_col="date",
    )
    assert_validation_within_window(
        labels,
        fold.cutoff,
        horizon_days=fold.horizon_days,
        date_col="date",
    )
    assert_disjoint_keys(training, labels, key_cols=("unique_id", "date"))

    return BacktestSplit(
        fold=fold,
        training_history=training,
        validation_features=safe_features,
        validation_labels=labels,
        requested_rows=len(shifted_grid),
        scored_rows=len(labels),
        missing_label_rows=int((~label_available).sum()),
        excluded_training_missing_targets=training_missing,
    )


def iter_backtest_splits(
    history: pd.DataFrame,
    official_grid: pd.DataFrame,
    *,
    cutoffs: Iterable[object] = DEFAULT_CUTOFFS,
    horizon_days: int = 14,
    target_col: str = "sales",
    date_col: str = "date",
) -> Iterator[BacktestSplit]:
    """Yield the four splits one at a time to bound memory use."""

    for fold in make_backtest_folds(cutoffs, horizon_days=horizon_days):
        yield materialize_backtest_split(
            history,
            official_grid,
            fold,
            target_col=target_col,
            date_col=date_col,
        )
