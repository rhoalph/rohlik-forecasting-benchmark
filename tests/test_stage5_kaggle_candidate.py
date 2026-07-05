from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_stage5_kaggle_candidate import _apply_zero_clipping


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submissions" / "stage5_s5b_price_discount_candidate.csv"
SOLUTION = ROOT / "data" / "raw" / "solution.csv"


def test_stage5_candidate_file_matches_solution_shape_and_order() -> None:
    submission = pd.read_csv(SUBMISSION)
    solution = pd.read_csv(SOLUTION)

    assert list(submission.columns) == ["id", "sales_hat"]
    assert len(submission) == 47021
    assert len(solution) == 47021
    assert submission["id"].duplicated().sum() == 0
    assert submission["id"].astype(str).reset_index(drop=True).equals(
        solution["id"].astype(str).reset_index(drop=True)
    )
    assert submission["sales_hat"].notna().all()
    assert np.issubdtype(submission["sales_hat"].dtype, np.number)
    assert int((submission["sales_hat"] < 0).sum()) == 0


def test_stage5_zero_clipping_changes_only_negative_values() -> None:
    values = np.array([-3.5, 0.0, 2.25, -0.1, 4.0], dtype=np.float64)
    clipped, changed = _apply_zero_clipping(values)

    assert np.array_equal(clipped, np.array([0.0, 0.0, 2.25, 0.0, 4.0]))
    assert changed == 2

