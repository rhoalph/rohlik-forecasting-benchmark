import pandas as pd
import pytest

from baselines.naive import (
    item_weekday_median_forecast,
    prepare_context,
    same_weekday_last_week_forecast,
    trailing_7_day_mean_forecast,
)
from dataguard.cutoff import LeakageError


def _training() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unique_id": [1] * 14 + [2],
            "date": list(pd.date_range("2024-01-01", "2024-01-14"))
            + [pd.Timestamp("2024-01-14")],
            "sales": [float(value) for value in range(1, 15)] + [20.0],
        }
    )


def _validation() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unique_id": [1, 1, 2, 3],
            "date": pd.to_datetime(
                ["2024-01-15", "2024-01-22", "2024-01-15", "2024-01-15"]
            ),
        }
    )


def test_same_weekday_never_uses_validation_period_sales() -> None:
    context = prepare_context(_training(), _validation(), "2024-01-14")
    result = same_weekday_last_week_forecast(context).predictions

    assert result.loc[0, "sales_hat"] == 8.0  # 2024-01-08 is cutoff-safe t-7.
    assert result.loc[1, "sales_hat"] == 14.0  # t-7 crosses cutoff, so last-value fallback.
    assert result.loc[2, "sales_hat"] == 20.0  # Missing t-7, so last-value fallback.
    assert result.loc[3, "sales_hat"] == pytest.approx(8.0)  # Global-median fallback.


def test_rolling_and_weekday_statistics_use_cutoff_history_only() -> None:
    context = prepare_context(_training(), _validation(), "2024-01-14")
    rolling = trailing_7_day_mean_forecast(context).predictions
    weekday = item_weekday_median_forecast(context).predictions

    assert rolling.loc[0, "sales_hat"] == pytest.approx(11.0)
    assert rolling.loc[2, "sales_hat"] == 20.0
    assert rolling.loc[3, "sales_hat"] == pytest.approx(8.0)
    assert weekday["sales_hat"].notna().all()


def test_context_rejects_training_rows_after_cutoff() -> None:
    training = _training()
    training.loc[len(training)] = [1, pd.Timestamp("2024-01-15"), 999.0]
    with pytest.raises(LeakageError, match="after cutoff"):
        prepare_context(training, _validation(), "2024-01-14")
