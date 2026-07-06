from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.run_stage5e_kaggle_candidate import _apply_zero_clipping, _validate_submission_frame


def test_stage5e_candidate_submission_validation_checks_order_and_shape() -> None:
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
            pd.DataFrame({"id": ["1_2024-06-03", "1_2024-06-03"], "sales_hat": [1.0, 1.0]}),
        )


def test_stage5e_candidate_clipping_only_changes_negative_predictions() -> None:
    prediction = np.array([-2.5, -0.1, 0.0, 1.2], dtype=float)
    clipped, changed = _apply_zero_clipping(prediction)

    assert changed == 2
    assert clipped.tolist() == [0.0, 0.0, 0.0, 1.2]
    assert np.all(clipped >= 0.0)
