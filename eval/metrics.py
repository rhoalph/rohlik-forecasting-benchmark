"""Metric implementations for the Rohlik benchmark.

Kaggle defines the competition metric as scikit-learn's weighted mean absolute
error with the supplied per-inventory weights. Therefore WMAE is the sum of
weighted absolute errors divided by the sum of sample weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


ArrayLike = Iterable[float] | np.ndarray


@dataclass(frozen=True)
class MetricResult:
    """All Stage 1 metrics on the same aligned prediction rows."""

    wmae: float
    wape: float
    bias: float
    rows: int


def _finite_vector(name: str, values: ArrayLike) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}.")
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return array


def _aligned_actual_prediction(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    actual = _finite_vector("y_true", y_true)
    prediction = _finite_vector("y_pred", y_pred)
    if actual.shape != prediction.shape:
        raise ValueError(
            f"y_true and y_pred must have equal length; got {len(actual)} and {len(prediction)}."
        )
    return actual, prediction


def wmae(y_true: ArrayLike, y_pred: ArrayLike, sample_weight: ArrayLike) -> float:
    """Return Kaggle's official weighted mean absolute error.

    Formula: ``sum(weight * abs(y_true - y_pred)) / sum(weight)``.
    A test weight is repeated for every requested row of its ``unique_id``.
    """

    actual, prediction = _aligned_actual_prediction(y_true, y_pred)
    weight = _finite_vector("sample_weight", sample_weight)
    if weight.shape != actual.shape:
        raise ValueError(
            "sample_weight must have the same length as y_true; "
            f"got {len(weight)} and {len(actual)}."
        )
    if (weight < 0).any():
        raise ValueError("sample_weight must not contain negative values.")
    denominator = float(weight.sum())
    if denominator <= 0:
        raise ValueError("sample_weight must have a positive sum.")
    return float(np.dot(weight, np.abs(actual - prediction)) / denominator)


def wape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return total absolute error divided by total absolute actual demand.

    The return value is a ratio: multiply by 100 when displaying a percentage.
    Competition sample weights are intentionally not used in this business
    metric; volume itself supplies the weighting.
    """

    actual, prediction = _aligned_actual_prediction(y_true, y_pred)
    denominator = float(np.abs(actual).sum())
    if denominator <= 0:
        raise ValueError("WAPE is undefined when total absolute actual demand is zero.")
    return float(np.abs(actual - prediction).sum() / denominator)


def bias(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return signed forecast error divided by total actual demand.

    Positive values indicate over-forecasting and negative values indicate
    under-forecasting. The return value is a ratio.
    """

    actual, prediction = _aligned_actual_prediction(y_true, y_pred)
    denominator = float(actual.sum())
    if denominator == 0:
        raise ValueError("Bias is undefined when total actual demand is zero.")
    return float((prediction - actual).sum() / denominator)


def score_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sample_weight: ArrayLike,
) -> MetricResult:
    """Compute WMAE, WAPE, and bias after one shared validation pass."""

    actual, prediction = _aligned_actual_prediction(y_true, y_pred)
    weight = _finite_vector("sample_weight", sample_weight)
    if weight.shape != actual.shape:
        raise ValueError(
            "sample_weight must have the same length as y_true; "
            f"got {len(weight)} and {len(actual)}."
        )
    return MetricResult(
        wmae=wmae(actual, prediction, weight),
        wape=wape(actual, prediction),
        bias=bias(actual, prediction),
        rows=len(actual),
    )
