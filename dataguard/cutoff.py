"""Cutoff and validation-window assertions used by every backtest fold."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import pandas as pd


class LeakageError(ValueError):
    """Raised when temporal or target isolation is violated."""


def _as_timestamp(value: object, *, name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a valid scalar date, got {value!r}.") from exc
    if pd.isna(timestamp):
        raise ValueError(f"{name} must not be missing.")
    if timestamp.tz is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp.normalize()


def _parsed_dates(frame: pd.DataFrame, date_col: str) -> pd.Series:
    if date_col not in frame.columns:
        raise KeyError(f"Missing required date column {date_col!r}.")
    parsed = pd.to_datetime(frame[date_col], errors="coerce")
    if parsed.isna().any():
        bad_count = int(parsed.isna().sum())
        raise ValueError(f"{date_col!r} contains {bad_count} missing or invalid dates.")
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_localize(None)
    return parsed.dt.normalize()


def validation_bounds(cutoff: object, horizon_days: int = 14) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return inclusive validation bounds immediately after ``cutoff``."""

    if not isinstance(horizon_days, int) or isinstance(horizon_days, bool) or horizon_days <= 0:
        raise ValueError("horizon_days must be a positive integer.")
    normalized = _as_timestamp(cutoff, name="cutoff")
    return (
        normalized + pd.Timedelta(days=1),
        normalized + pd.Timedelta(days=horizon_days),
    )


def assert_history_at_or_before_cutoff(
    frame: pd.DataFrame,
    cutoff: object,
    *,
    date_col: str = "date",
) -> None:
    """Assert that no row in a historical frame is later than ``cutoff``."""

    dates = _parsed_dates(frame, date_col)
    normalized = _as_timestamp(cutoff, name="cutoff")
    future = dates > normalized
    if future.any():
        raise LeakageError(
            f"Historical frame contains {int(future.sum())} rows after cutoff "
            f"{normalized.date()}; latest date is {dates.max().date()}."
        )


def filter_history_at_cutoff(
    frame: pd.DataFrame,
    cutoff: object,
    *,
    date_col: str = "date",
) -> pd.DataFrame:
    """Copy rows at or before ``cutoff`` and normalize their date column."""

    dates = _parsed_dates(frame, date_col)
    normalized = _as_timestamp(cutoff, name="cutoff")
    result = frame.loc[dates <= normalized].copy()
    result[date_col] = dates.loc[result.index]
    assert_history_at_or_before_cutoff(result, normalized, date_col=date_col)
    return result


def assert_validation_within_window(
    frame: pd.DataFrame,
    cutoff: object,
    *,
    horizon_days: int = 14,
    date_col: str = "date",
) -> None:
    """Assert that every validation row lies in cutoff+1 through cutoff+horizon."""

    dates = _parsed_dates(frame, date_col)
    start, end = validation_bounds(cutoff, horizon_days)
    outside = (dates < start) | (dates > end)
    if outside.any():
        raise LeakageError(
            f"Validation frame contains {int(outside.sum())} rows outside "
            f"{start.date()} through {end.date()}."
        )


def assert_source_dates_at_or_before_cutoff(
    source_dates: Iterable[object],
    cutoff: object,
    *,
    source_name: str = "target-derived input",
) -> None:
    """Assert lineage dates for a target-derived value do not cross cutoff."""

    series = pd.Series(list(source_dates), dtype="object")
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"{source_name} contains missing or invalid source dates.")
    normalized = _as_timestamp(cutoff, name="cutoff")
    future = parsed.dt.normalize() > normalized
    if future.any():
        raise LeakageError(
            f"{source_name} uses {int(future.sum())} source dates after cutoff "
            f"{normalized.date()}."
        )


def assert_target_not_present(frame: pd.DataFrame, *, target_col: str = "sales") -> None:
    """Keep labels physically separate from validation feature frames."""

    if target_col in frame.columns:
        raise LeakageError(f"Target column {target_col!r} is present in a feature frame.")


def assert_disjoint_keys(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    key_cols: Sequence[str] = ("unique_id", "date"),
) -> None:
    """Assert two frames share no composite keys."""

    missing_left = set(key_cols) - set(left.columns)
    missing_right = set(key_cols) - set(right.columns)
    if missing_left or missing_right:
        raise KeyError(
            f"Missing key columns; left={sorted(missing_left)}, right={sorted(missing_right)}."
        )
    overlap = left.loc[:, key_cols].merge(
        right.loc[:, key_cols],
        on=list(key_cols),
        how="inner",
    )
    if not overlap.empty:
        raise LeakageError(f"Frames share {len(overlap)} rows on keys {tuple(key_cols)}.")
