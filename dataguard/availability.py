"""Central field-availability registry.

The adopted policy permits prices and discounts supplied by Kaggle as known
future covariates. ``total_orders`` remains historical-only even though Kaggle
supplies it in the test file. This deliberate exclusion keeps realized order
volume out of forecast-window features.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import pandas as pd


class Availability(str, Enum):
    """When a field may be used relative to a forecast origin."""

    HISTORICAL_ONLY = "historical_only"
    KNOWN_FUTURE = "known_future"
    STATIC_METADATA = "static_metadata"
    EVALUATION_ONLY = "evaluation_only"
    TARGET = "target"
    OUTPUT_ONLY = "output_only"


@dataclass(frozen=True)
class FieldSpec:
    """Availability declaration for one raw or derived field."""

    availability: Availability
    source: str
    rationale: str


class UnknownFieldError(ValueError):
    """Raised when code tries to use a field absent from the registry."""


DISCOUNT_COLUMNS = tuple(f"type_{index}_discount" for index in range(7))
CATEGORY_COLUMNS = tuple(f"L{index}_category_name_en" for index in range(1, 5))


FIELD_REGISTRY: dict[str, FieldSpec] = {
    "unique_id": FieldSpec(
        Availability.STATIC_METADATA,
        "sales/inventory/weights",
        "Warehouse-specific inventory identifier.",
    ),
    "product_unique_id": FieldSpec(
        Availability.STATIC_METADATA,
        "inventory",
        "Cross-warehouse product identifier.",
    ),
    "name": FieldSpec(
        Availability.STATIC_METADATA,
        "inventory",
        "Anonymized product name supplied as snapshot metadata.",
    ),
    "warehouse": FieldSpec(
        Availability.STATIC_METADATA,
        "sales/inventory/calendar",
        "Warehouse assignment does not vary within the supplied snapshot.",
    ),
    "date": FieldSpec(
        Availability.KNOWN_FUTURE,
        "sales/calendar",
        "Requested forecast date is known at the forecast origin.",
    ),
    "horizon_day": FieldSpec(
        Availability.KNOWN_FUTURE,
        "derived from date",
        "Integer offset from the forecast origin; target-independent.",
    ),
    "holiday_name": FieldSpec(
        Availability.KNOWN_FUTURE,
        "calendar",
        "Published calendar information.",
    ),
    "holiday": FieldSpec(
        Availability.KNOWN_FUTURE,
        "calendar",
        "Published calendar information.",
    ),
    "shops_closed": FieldSpec(
        Availability.KNOWN_FUTURE,
        "calendar",
        "Scheduled calendar closure.",
    ),
    "winter_school_holidays": FieldSpec(
        Availability.KNOWN_FUTURE,
        "calendar",
        "Published school calendar.",
    ),
    "school_holidays": FieldSpec(
        Availability.KNOWN_FUTURE,
        "calendar",
        "Published school calendar.",
    ),
    "sell_price_main": FieldSpec(
        Availability.KNOWN_FUTURE,
        "sales_test",
        "Adopted Kaggle-provided covariate policy.",
    ),
    "total_orders": FieldSpec(
        Availability.HISTORICAL_ONLY,
        "sales",
        "Explicitly excluded from future features despite being in sales_test.",
    ),
    "availability": FieldSpec(
        Availability.HISTORICAL_ONLY,
        "sales_train",
        "Not known at forecast time and absent from sales_test.",
    ),
    "sales": FieldSpec(
        Availability.TARGET,
        "sales_train",
        "May be used only as history through the cutoff or as isolated labels.",
    ),
    "weight": FieldSpec(
        Availability.EVALUATION_ONLY,
        "test_weights",
        "Official WMAE sample weight; never a predictive feature.",
    ),
    "id": FieldSpec(
        Availability.OUTPUT_ONLY,
        "solution",
        "Submission key assembled from unique_id and date.",
    ),
    "sales_hat": FieldSpec(
        Availability.OUTPUT_ONLY,
        "solution",
        "Submission prediction, not an input field.",
    ),
}

for column in DISCOUNT_COLUMNS:
    FIELD_REGISTRY[column] = FieldSpec(
        Availability.KNOWN_FUTURE,
        "sales_test",
        "Adopted Kaggle-provided promotion covariate policy.",
    )

for column in CATEGORY_COLUMNS:
    FIELD_REGISTRY[column] = FieldSpec(
        Availability.STATIC_METADATA,
        "inventory",
        "Competition snapshot category hierarchy.",
    )


def classify_field(field: str) -> FieldSpec:
    """Return the declared availability for ``field``.

    Unknown fields fail closed. Every new raw or derived feature must be added
    deliberately to this registry before it can enter a forecast-window frame.
    """

    try:
        return FIELD_REGISTRY[field]
    except KeyError as exc:
        raise UnknownFieldError(
            f"Field {field!r} is not classified; add it to FIELD_REGISTRY before use."
        ) from exc


def assert_future_columns_allowed(columns: Iterable[str]) -> None:
    """Reject fields that are unavailable as forecast-window covariates."""

    invalid: list[str] = []
    for column in columns:
        availability = classify_field(column).availability
        if availability not in {
            Availability.KNOWN_FUTURE,
            Availability.STATIC_METADATA,
        }:
            invalid.append(f"{column} ({availability.value})")
    if invalid:
        raise ValueError(
            "Forecast-window features contain unavailable fields: " + ", ".join(invalid)
        )


def select_future_covariates(frame: pd.DataFrame) -> pd.DataFrame:
    """Return only declared known-future and static columns, preserving order.

    All input columns must be classified. Failing on unknown fields prevents a
    newly added column from bypassing availability review. Under the current
    policy this includes price and discount fields and excludes ``total_orders``,
    ``availability``, ``sales``, and ``weight``.
    """

    selected: list[str] = []
    for column in frame.columns:
        availability = classify_field(column).availability
        if availability in {
            Availability.KNOWN_FUTURE,
            Availability.STATIC_METADATA,
        }:
            selected.append(column)
    return frame.loc[:, selected].copy()
