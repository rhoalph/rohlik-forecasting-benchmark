from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.stage3_minimal import FORBIDDEN_FEATURE_FIELDS
from features.stage5e_stronger_features import APPROVED_STAGE5E_FEATURES
from features.stage5h_supply_chain_category_pressure import (
    APPROVED_STAGE5H_FEATURES,
    FEATURE_AVAILABILITY,
    STAGE5H_EXTRA_FEATURES,
    STAGE5H_FORBIDDEN_FEATURE_FIELDS,
    _history_cutoff,
    _window_group_stats,
    _window_item_share_stats,
    build_stage5h_feature_batch,
)
from scripts.run_stage5e_kaggle_candidate import _build_official_request_frame, _load_raw_data
from scripts.run_stage5h_supply_chain_category_pressure_experiment import _blend_predictions


def test_stage5h_contract_extends_stage5e_explicitly() -> None:
    assert len(APPROVED_STAGE5H_FEATURES) == len(APPROVED_STAGE5E_FEATURES) + len(STAGE5H_EXTRA_FEATURES)
    assert APPROVED_STAGE5H_FEATURES[: len(APPROVED_STAGE5E_FEATURES)] == APPROVED_STAGE5E_FEATURES
    assert STAGE5H_FORBIDDEN_FEATURE_FIELDS == FORBIDDEN_FEATURE_FIELDS
    assert FEATURE_AVAILABILITY["wh_cat_l2_reversion_pressure"] == "historical only"
    assert FEATURE_AVAILABILITY["discount_x_wh_cat_l2_reversion_pressure"] == "known future"
    assert FEATURE_AVAILABILITY["relative_price_x_wh_cat_l2_reversion_pressure"] == "known future"


def test_stage5h_window_stats_and_shares_are_cutoff_safe() -> None:
    origin = pd.Timestamp("2024-01-10")
    history = pd.DataFrame(
        {
            "unique_id": [1, 1, 1, 2, 2, 2],
            "date": pd.to_datetime(
                [
                    "2024-01-08",
                    "2024-01-09",
                    "2023-12-01",
                    "2023-12-02",
                    "2023-12-03",
                    "2023-12-04",
                ]
            ),
            "sales": [10.0, 20.0, 7.0, 5.0, 5.0, 5.0],
        }
    )
    cutoff = _history_cutoff(history, origin)
    assert cutoff["date"].max() <= origin
    group_7 = _window_group_stats(
        cutoff.assign(warehouse=["W1", "W1", "W1", "W1", "W1", "W1"], L2_category_name_en=["A", "A", "A", "B", "B", "B"]),
        origin,
        7,
    )
    assert set(group_7["L2_category_name_en"]) == {"A"}
    assert np.isclose(group_7.loc[group_7["L2_category_name_en"] == "A", "sales_sum"].iloc[0], 30.0)
    shares_7 = _window_item_share_stats(
        cutoff.assign(warehouse=["W1", "W1", "W1", "W1", "W1", "W1"], L2_category_name_en=["A", "A", "A", "B", "B", "B"]),
        origin,
        7,
    )
    assert set(shares_7["L2_category_name_en"]) == {"A"}
    assert np.isclose(shares_7.loc[shares_7["unique_id"] == 1, "item_share"].iloc[0], 1.0)
    assert all(value >= 0 for value in shares_7["item_share"].tolist())


def test_stage5h_feature_batch_builds_with_allowed_fields() -> None:
    history, inventory, calendar, sales_test, official_grid, _solution = _load_raw_data()
    request = _build_official_request_frame(sales_test, official_grid)

    grouped_inventory = inventory.groupby(["warehouse", "L2_category_name_en"], observed=True)
    selected_ids = None
    selected_group = None
    request_ids = set(request["unique_id"].tolist())
    for (warehouse, category), group in grouped_inventory:
        candidate_ids = [uid for uid in group["unique_id"].tolist() if uid in request_ids]
        if len(candidate_ids) >= 2:
            selected_ids = candidate_ids[:2]
            selected_group = (warehouse, category)
            break
    assert selected_ids is not None, "expected at least one warehouse/category pair with two items"
    assert selected_group is not None

    request_subset = pd.concat(
        [request.loc[request["unique_id"].eq(uid)].head(1) for uid in selected_ids],
        ignore_index=True,
    )
    assert len(request_subset) == 2

    history_subset = history.loc[
        history["unique_id"].isin(selected_ids) & (history["date"] <= pd.Timestamp("2024-06-02"))
    ].copy()
    assert not history_subset.empty

    batch = build_stage5h_feature_batch(
        history_subset,
        request_subset,
        inventory,
        calendar,
        pd.Timestamp("2024-06-02"),
    )

    assert tuple(batch.matrix.columns) == APPROVED_STAGE5H_FEATURES
    assert tuple(batch.matrix.columns[: len(APPROVED_STAGE5E_FEATURES)]) == APPROVED_STAGE5E_FEATURES
    assert not STAGE5H_FORBIDDEN_FEATURE_FIELDS.intersection(batch.matrix.columns)
    assert not FORBIDDEN_FEATURE_FIELDS.intersection(batch.matrix.columns)
    assert np.isfinite(batch.matrix.to_numpy(dtype=np.float64, copy=False)).all()
    assert np.allclose(
        batch.matrix["wh_cat_l2_reversion_pressure"].to_numpy(dtype=np.float64),
        batch.matrix["wh_cat_l2_sales_7d_mean"].to_numpy(dtype=np.float64)
        / batch.matrix["wh_cat_l2_sales_28d_mean"].to_numpy(dtype=np.float64)
        - 1.0,
    )
    assert (batch.matrix["item_share_of_wh_cat_l2_7d"] >= 0).all()
    assert (batch.matrix["item_share_of_wh_cat_l2_28d"] >= 0).all()


def test_stage5h_fixed_blend_requires_key_alignment() -> None:
    frame = pd.DataFrame(
        {
            "unique_id": [1, 2],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "sales_hat": [1.0, 2.0],
        }
    )
    blended, clipped, audit = _blend_predictions(frame, frame.copy())
    assert np.allclose(blended, np.array([1.0, 2.0]))
    assert np.allclose(clipped, blended)
    assert audit["weights"] == [0.5, 0.5]
    assert audit["clipped_rows"] == 0
    with pytest.raises(AssertionError):
        _blend_predictions(frame, frame.iloc[::-1].reset_index(drop=True))
