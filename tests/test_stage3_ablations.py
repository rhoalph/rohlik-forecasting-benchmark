import pandas as pd
import pytest

from features.stage3_minimal import (
    APPROVED_FEATURES,
    CATEGORICAL_FEATURES,
    DISCOUNT_COLUMNS,
)
from scripts.run_stage3_f1_ablations import (
    ABLATION_FEATURES,
    HISTORICAL_FEATURES,
    KNOWN_FUTURE_FEATURES,
    MOST_RECENT_ORIGIN,
    PRIORITY2_CATEGORICAL_FEATURES,
    PRIORITY2_FEATURES,
    STATIC_FEATURES,
    categorical_features_for,
    select_ablation_matrix,
)


def test_approved_ablation_feature_contracts_are_exact() -> None:
    commercial = {"sell_price_main", *DISCOUNT_COLUMNS}
    assert ABLATION_FEATURES["A1"] == tuple(
        feature for feature in APPROVED_FEATURES if feature not in commercial
    )
    assert ABLATION_FEATURES["A2"] == HISTORICAL_FEATURES
    assert ABLATION_FEATURES["A3"] == STATIC_FEATURES
    assert ABLATION_FEATURES["A4"] == (*STATIC_FEATURES, *KNOWN_FUTURE_FEATURES)
    assert {key: len(value) for key, value in ABLATION_FEATURES.items()} == {
        "A1": 28,
        "A2": 12,
        "A3": 7,
        "A4": 24,
    }


def test_ablation_subsets_add_nothing_and_remove_expected_families() -> None:
    approved = set(APPROVED_FEATURES)
    assert all(set(features) <= approved for features in ABLATION_FEATURES.values())
    assert not {"sell_price_main", *DISCOUNT_COLUMNS}.intersection(
        ABLATION_FEATURES["A1"]
    )
    assert set(ABLATION_FEATURES["A2"]).isdisjoint(
        set(STATIC_FEATURES) | set(KNOWN_FUTURE_FEATURES)
    )
    assert set(ABLATION_FEATURES["A4"]).isdisjoint(HISTORICAL_FEATURES)


def test_ablation_matrix_selection_is_ordered_and_fail_closed() -> None:
    matrix = pd.DataFrame({feature: [0] for feature in APPROVED_FEATURES})
    for ablation_id, expected in ABLATION_FEATURES.items():
        selected = select_ablation_matrix(matrix, ablation_id)
        assert tuple(selected.columns) == expected
    with pytest.raises(ValueError, match="Unknown approved ablation"):
        select_ablation_matrix(matrix, "A5")
    with pytest.raises(KeyError, match="missing features"):
        select_ablation_matrix(matrix.drop(columns="trailing_7_mean"), "A2")


def test_categorical_lists_are_intersections_in_reference_order() -> None:
    assert categorical_features_for("A2") == ()
    assert categorical_features_for("A3") == STATIC_FEATURES
    assert categorical_features_for("A4") == (
        *STATIC_FEATURES,
        "day_of_week",
        "month",
    )


def test_priority2_contract_changes_only_origin_count_or_category_treatment() -> None:
    assert str(MOST_RECENT_ORIGIN.date()) == "2024-05-05"
    assert PRIORITY2_FEATURES == {
        "A5": APPROVED_FEATURES,
        "A6": APPROVED_FEATURES,
    }
    assert PRIORITY2_CATEGORICAL_FEATURES == {
        "A5": CATEGORICAL_FEATURES,
        "A6": (),
    }
    assert len(CATEGORICAL_FEATURES) == 9
