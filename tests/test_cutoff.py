import pandas as pd
import pytest

from dataguard.cutoff import (
    LeakageError,
    assert_disjoint_keys,
    assert_history_at_or_before_cutoff,
    assert_source_dates_at_or_before_cutoff,
    assert_target_not_present,
    assert_validation_within_window,
    filter_history_at_cutoff,
    validation_bounds,
)


def test_filter_history_enforces_cutoff() -> None:
    frame = pd.DataFrame(
        {
            "unique_id": [1, 1, 1],
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "sales": [1.0, 2.0, 3.0],
        }
    )
    filtered = filter_history_at_cutoff(frame, "2024-01-02")
    assert filtered["date"].max() == pd.Timestamp("2024-01-02")
    assert len(filtered) == 2
    assert_history_at_or_before_cutoff(filtered, "2024-01-02")


def test_history_assertion_detects_future_row() -> None:
    frame = pd.DataFrame({"date": ["2024-01-01", "2024-01-03"]})
    with pytest.raises(LeakageError, match="after cutoff"):
        assert_history_at_or_before_cutoff(frame, "2024-01-02")


def test_validation_window_is_cutoff_plus_one_through_fourteen() -> None:
    start, end = validation_bounds("2024-01-01", 14)
    assert start == pd.Timestamp("2024-01-02")
    assert end == pd.Timestamp("2024-01-15")
    valid = pd.DataFrame({"date": pd.date_range(start, end)})
    assert_validation_within_window(valid, "2024-01-01", horizon_days=14)

    invalid = pd.DataFrame({"date": ["2024-01-01", "2024-01-16"]})
    with pytest.raises(LeakageError, match="outside"):
        assert_validation_within_window(invalid, "2024-01-01", horizon_days=14)


def test_target_source_lineage_must_stop_at_cutoff() -> None:
    assert_source_dates_at_or_before_cutoff(
        ["2024-01-01", "2024-01-02"],
        "2024-01-02",
    )
    with pytest.raises(LeakageError, match="source dates after cutoff"):
        assert_source_dates_at_or_before_cutoff(
            ["2024-01-01", "2024-01-03"],
            "2024-01-02",
        )


def test_labels_cannot_be_present_in_feature_frame() -> None:
    with pytest.raises(LeakageError, match="Target column"):
        assert_target_not_present(pd.DataFrame({"sales": [1.0]}))


def test_training_and_validation_keys_must_be_disjoint() -> None:
    train = pd.DataFrame({"unique_id": [1], "date": [pd.Timestamp("2024-01-01")]})
    validation = pd.DataFrame({"unique_id": [1], "date": [pd.Timestamp("2024-01-01")]})
    with pytest.raises(LeakageError, match="share"):
        assert_disjoint_keys(train, validation)
