import numpy as np
import pytest

from eval.metrics import bias, score_metrics, wape, wmae


def test_metric_formulas_match_hand_calculation() -> None:
    actual = [10.0, 20.0, 30.0]
    prediction = [12.0, 18.0, 24.0]
    weight = [1.0, 2.0, 3.0]

    assert wmae(actual, prediction, weight) == pytest.approx(4.0)
    assert wape(actual, prediction) == pytest.approx(10.0 / 60.0)
    assert bias(actual, prediction) == pytest.approx(-6.0 / 60.0)

    result = score_metrics(actual, prediction, weight)
    assert result.wmae == pytest.approx(4.0)
    assert result.wape == pytest.approx(1.0 / 6.0)
    assert result.bias == pytest.approx(-0.1)
    assert result.rows == 3


def test_positive_bias_means_overforecasting() -> None:
    assert bias([5.0, 5.0], [6.0, 7.0]) == pytest.approx(0.3)


@pytest.mark.parametrize(
    ("actual", "prediction", "weight", "message"),
    [
        ([1.0], [1.0, 2.0], [1.0], "equal length"),
        ([1.0], [1.0], [1.0, 2.0], "same length"),
        ([1.0], [np.nan], [1.0], "NaN or infinite"),
        ([1.0], [1.0], [-1.0], "negative"),
        ([1.0], [1.0], [0.0], "positive sum"),
    ],
)
def test_wmae_rejects_invalid_input(actual, prediction, weight, message) -> None:
    with pytest.raises(ValueError, match=message):
        wmae(actual, prediction, weight)


def test_wape_and_bias_reject_zero_denominators() -> None:
    with pytest.raises(ValueError, match="undefined"):
        wape([0.0, 0.0], [1.0, 1.0])
    with pytest.raises(ValueError, match="undefined"):
        bias([-1.0, 1.0], [0.0, 0.0])
