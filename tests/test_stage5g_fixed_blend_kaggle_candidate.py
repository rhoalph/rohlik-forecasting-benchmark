from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from features.stage5e_stronger_features import APPROVED_STAGE5E_FEATURES
from scripts.run_stage5e_kaggle_candidate import _apply_zero_clipping, _validate_submission_frame
from scripts.run_stage5g_fixed_blend_kaggle_candidate import (
    FIXED_BLEND_WEIGHTS,
    RAW_MODEL_CONFIG,
    TWEEDIE_MODEL_CONFIG,
    TWEEDIE_VARIANCE_POWER,
    _blend_fixed_predictions,
    _require_stage5e_contract,
)


def test_stage5g_fixed_blend_is_globally_fixed() -> None:
    assert FIXED_BLEND_WEIGHTS == (0.5, 0.5)
    assert np.isclose(sum(FIXED_BLEND_WEIGHTS), 1.0)
    assert TWEEDIE_VARIANCE_POWER == 1.1
    assert RAW_MODEL_CONFIG.objective == "regression_l1"
    assert TWEEDIE_MODEL_CONFIG.objective == "tweedie"

    signature = inspect.signature(_blend_fixed_predictions)
    assert list(signature.parameters) == ["raw_prediction_frame", "tweedie_prediction_frame"]


def test_stage5g_fixed_blend_alignment_and_clipping() -> None:
    raw = pd.DataFrame(
        {
            "unique_id": [1, 2],
            "date": pd.to_datetime(["2024-06-03", "2024-06-04"]),
            "sales_hat": [-4.0, 4.0],
        }
    )
    tweedie = pd.DataFrame(
        {
            "unique_id": [1, 2],
            "date": pd.to_datetime(["2024-06-03", "2024-06-04"]),
            "sales_hat": [2.0, 6.0],
        }
    )

    blended, clipped, audit = _blend_fixed_predictions(raw, tweedie)

    assert blended.tolist() == [-1.0, 5.0]
    assert clipped.tolist() == [0.0, 5.0]
    assert audit["negative_before_clip"] == 1
    assert audit["clipped_rows"] == 1

    with pytest.raises(AssertionError):
        _blend_fixed_predictions(raw, tweedie.iloc[::-1].reset_index(drop=True))


def test_stage5g_stage5e_feature_contract_is_unchanged() -> None:
    frame = pd.DataFrame(columns=APPROVED_STAGE5E_FEATURES)
    _require_stage5e_contract(frame)

    with pytest.raises(AssertionError):
        _require_stage5e_contract(frame.rename(columns={APPROVED_STAGE5E_FEATURES[0]: "wrong"}))


def test_stage5g_submission_validation_catches_shape_and_order() -> None:
    official_grid = pd.DataFrame(
        {
            "unique_id": [1, 2],
            "test_date": pd.to_datetime(["2024-06-03", "2024-06-04"]),
        }
    )
    solution = pd.DataFrame({"id": ["1_2024-06-03", "2_2024-06-04"]})
    submission = pd.DataFrame({"id": ["1_2024-06-03", "2_2024-06-04"], "sales_hat": [1.0, 2.0]})

    _validate_submission_frame(official_grid, solution, submission)

    with pytest.raises(AssertionError):
        _validate_submission_frame(
            official_grid,
            solution,
            pd.DataFrame({"id": ["2_2024-06-04", "1_2024-06-03"], "sales_hat": [2.0, 1.0]}),
        )

    with pytest.raises(AssertionError):
        _validate_submission_frame(
            official_grid,
            solution,
            pd.DataFrame({"id": ["1_2024-06-03"], "sales_hat": [1.0]}),
        )

    with pytest.raises(AssertionError):
        _validate_submission_frame(
            official_grid,
            solution,
            pd.DataFrame({"id": ["1_2024-06-03", "2_2024-06-04"], "sales_hat": [1.0, None]}),
        )


def test_stage5g_zero_clipping_matches_existing_behavior() -> None:
    prediction = np.array([-1.0, 0.0, 2.5], dtype=float)
    clipped, changed = _apply_zero_clipping(prediction)

    assert changed == 1
    assert clipped.tolist() == [0.0, 0.0, 2.5]
