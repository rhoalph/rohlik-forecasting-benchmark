import pandas as pd

from scripts.run_stage3_all_folds_plain_model import (
    FOLD_SPECS,
    SUSPICIOUS_RELATIVE_IMPROVEMENT,
    suspicious_improvement,
    training_origins_for_fold,
)


def test_stage3_outer_fold_contract_matches_frozen_dates_and_coverage() -> None:
    assert [spec.name for spec in FOLD_SPECS] == ["F2", "F3", "F4"]
    assert [str(spec.cutoff.date()) for spec in FOLD_SPECS] == [
        "2024-05-05",
        "2024-04-21",
        "2024-04-07",
    ]
    assert [spec.expected_scored_rows for spec in FOLD_SPECS] == [43433, 42794, 42035]
    assert [spec.validation_end - spec.validation_start for spec in FOLD_SPECS] == [
        pd.Timedelta(days=13),
        pd.Timedelta(days=13),
        pd.Timedelta(days=13),
    ]


def test_each_fold_has_twelve_safe_nonoverlapping_training_origins() -> None:
    for spec in FOLD_SPECS:
        origins = training_origins_for_fold(spec.cutoff)
        assert len(origins) == 12
        assert origins[0] == spec.cutoff - pd.Timedelta(days=14)
        assert origins[-1] == spec.cutoff - pd.Timedelta(days=168)
        assert all(
            left - right == pd.Timedelta(days=14)
            for left, right in zip(origins, origins[1:])
        )
        assert all(origin + pd.Timedelta(days=14) <= spec.cutoff for origin in origins)


def test_suspicious_score_rule_is_strictly_greater_than_twenty_percent() -> None:
    assert SUSPICIOUS_RELATIVE_IMPROVEMENT == 0.20
    assert not suspicious_improvement(100.0, 80.0)
    assert suspicious_improvement(100.0, 79.999)
    assert not suspicious_improvement(100.0, 90.0)
