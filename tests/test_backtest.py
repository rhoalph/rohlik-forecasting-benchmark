import pandas as pd
import pytest

from dataguard.availability import UnknownFieldError
from eval.backtest import make_backtest_folds, materialize_backtest_split
from eval.grid import build_official_test_grid


def _official_grid() -> pd.DataFrame:
    dates = pd.date_range("2024-06-03", periods=14)
    test = pd.DataFrame(
        [(unique_id, date) for unique_id in (1, 2) for date in dates],
        columns=["unique_id", "date"],
    )
    weights = pd.DataFrame({"unique_id": [1, 2], "weight": [1.0, 2.0]})
    return build_official_test_grid(test, weights)


def _history() -> pd.DataFrame:
    rows = []
    for unique_id in (1, 2):
        for date in pd.date_range("2024-01-01", "2024-01-28"):
            if unique_id == 2 and date == pd.Timestamp("2024-01-28"):
                continue
            sales = float(unique_id)
            if (unique_id, date) in {
                (1, pd.Timestamp("2024-01-02")),
                (2, pd.Timestamp("2024-01-27")),
            }:
                sales = float("nan")
            rows.append(
                {
                    "unique_id": unique_id,
                    "date": date,
                    "warehouse": "Prague_1",
                    "total_orders": 1000.0,
                    "sales": sales,
                    "sell_price_main": 10.0,
                    "availability": 1.0,
                    "type_0_discount": 0.1,
                }
            )
    return pd.DataFrame(rows)


def test_default_folds_match_approved_dates() -> None:
    folds = make_backtest_folds()
    assert [str(fold.cutoff.date()) for fold in folds] == [
        "2024-05-19",
        "2024-05-05",
        "2024-04-21",
        "2024-04-07",
    ]
    assert all((fold.validation_end - fold.validation_start).days == 13 for fold in folds)


def test_materialized_split_isolates_labels_and_unsafe_future_fields() -> None:
    fold = make_backtest_folds(["2024-01-14"])[0]
    split = materialize_backtest_split(_history(), _official_grid(), fold)

    assert split.training_history["date"].max() == pd.Timestamp("2024-01-14")
    assert split.validation_features["date"].min() == pd.Timestamp("2024-01-15")
    assert split.validation_features["date"].max() == pd.Timestamp("2024-01-28")
    assert split.requested_rows == 28
    assert split.scored_rows == 26
    assert split.missing_label_rows == 2
    assert split.excluded_training_missing_targets == 1

    assert "sales" not in split.validation_features
    assert "total_orders" not in split.validation_features
    assert "availability" not in split.validation_features
    assert "weight" not in split.validation_features
    assert "sell_price_main" in split.validation_features
    assert "type_0_discount" in split.validation_features
    assert "horizon_day" in split.validation_features
    assert "sales" in split.validation_labels
    assert "weight" in split.validation_labels


def test_materialization_rejects_unclassified_columns() -> None:
    history = _history()
    history["unreviewed_feature"] = 1
    fold = make_backtest_folds(["2024-01-14"])[0]
    with pytest.raises(UnknownFieldError, match="not classified"):
        materialize_backtest_split(history, _official_grid(), fold)


def test_overlapping_validation_windows_are_rejected() -> None:
    with pytest.raises(ValueError, match="overlap"):
        make_backtest_folds(["2024-01-01", "2024-01-10"])
