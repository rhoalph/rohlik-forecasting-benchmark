from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_stage5_s5a_s5b_experiments import _candidate_clipping_inspection


def test_clipping_changes_only_negative_predictions() -> None:
    series = pd.Series([-2.0, 0.0, 1.5, -0.1, 4.2])
    clipped = series.clip(lower=0)
    assert np.array_equal(clipped.to_numpy(), np.array([0.0, 0.0, 1.5, 0.0, 4.2]))
    assert int((clipped != series).sum()) == 2


def test_candidate_clipping_inspection_reads_stage4_candidate_without_modifying_it() -> None:
    result = _candidate_clipping_inspection()
    assert result["rows"] == 47021
    assert result["negative_count"] == 20
    assert result["changed_rows_when_clipped"] == 20
    assert result["min_negative_value"] < 0
