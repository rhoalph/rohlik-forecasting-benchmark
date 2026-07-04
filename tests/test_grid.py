import pandas as pd
import pytest

from eval.grid import (
    build_official_test_grid,
    score_kaggle_aligned,
    shift_official_grid,
    validate_solution_template,
)


def _test_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2024-06-03", periods=14)
    test = pd.DataFrame(
        [(unique_id, date) for unique_id in (10, 20) for date in dates],
        columns=["unique_id", "date"],
    )
    weights = pd.DataFrame({"unique_id": [10, 20], "weight": [1.0, 3.0]})
    return test, weights


def test_official_grid_repeats_id_weight_and_preserves_order() -> None:
    test, weights = _test_data()
    grid = build_official_test_grid(test, weights)

    assert len(grid) == 28
    assert grid["grid_order"].tolist() == list(range(28))
    assert set(grid["horizon_day"]) == set(range(1, 15))
    assert grid.loc[grid["unique_id"].eq(10), "weight"].eq(1.0).all()
    assert grid.loc[grid["unique_id"].eq(20), "weight"].eq(3.0).all()


def test_shifted_grid_uses_local_cutoff_and_same_mask() -> None:
    test, weights = _test_data()
    grid = build_official_test_grid(test, weights)
    shifted = shift_official_grid(grid, "2024-01-01")
    assert shifted["date"].min() == pd.Timestamp("2024-01-02")
    assert shifted["date"].max() == pd.Timestamp("2024-01-15")
    assert shifted[["unique_id", "horizon_day", "weight"]].equals(
        grid[["unique_id", "horizon_day", "weight"]]
    )


def test_solution_template_must_match_exact_order() -> None:
    test, weights = _test_data()
    grid = build_official_test_grid(test, weights)
    solution = pd.DataFrame(
        {
            "id": test["unique_id"].astype(str)
            + "_"
            + pd.to_datetime(test["date"]).dt.strftime("%Y-%m-%d"),
            "sales_hat": 0,
        }
    )
    validate_solution_template(grid, solution)
    with pytest.raises(ValueError, match="ordering"):
        validate_solution_template(grid, solution.iloc[::-1].reset_index(drop=True))


def test_grid_scoring_joins_by_keys_and_uses_repeated_weights() -> None:
    labels = pd.DataFrame(
        {
            "unique_id": [10, 20],
            "date": ["2024-01-02", "2024-01-02"],
            "sales": [10.0, 20.0],
            "weight": [1.0, 3.0],
        }
    )
    predictions = pd.DataFrame(
        {
            "unique_id": [20, 10],
            "date": ["2024-01-02", "2024-01-02"],
            "sales_hat": [24.0, 12.0],
        }
    )
    result = score_kaggle_aligned(labels, predictions)
    assert result.wmae == pytest.approx((1.0 * 2.0 + 3.0 * 4.0) / 4.0)
    assert result.wape == pytest.approx(6.0 / 30.0)
    assert result.bias == pytest.approx(6.0 / 30.0)


def test_grid_scoring_rejects_missing_prediction() -> None:
    labels = pd.DataFrame(
        {
            "unique_id": [10, 20],
            "date": ["2024-01-02", "2024-01-02"],
            "sales": [10.0, 20.0],
            "weight": [1.0, 3.0],
        }
    )
    predictions = pd.DataFrame(
        {"unique_id": [10], "date": ["2024-01-02"], "sales_hat": [12.0]}
    )
    with pytest.raises(ValueError, match="missing=1"):
        score_kaggle_aligned(labels, predictions)


def test_official_grid_rejects_missing_weight() -> None:
    test, _ = _test_data()
    weights = pd.DataFrame({"unique_id": [10], "weight": [1.0]})
    with pytest.raises(ValueError, match="lack test weights"):
        build_official_test_grid(test, weights)
