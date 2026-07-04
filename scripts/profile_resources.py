#!/usr/bin/env python3
"""Profile one Rohlik CSV and optimize its in-memory representation.

The default input is ``data/raw/sales_train.csv``. This script never writes
processed data; it loads one CSV, reports deep memory use, applies conservative
in-memory dtype conversions, prints the result, and exits.
"""

from __future__ import annotations

import argparse
import resource
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "raw" / "sales_train.csv"

FLOAT32_COLUMNS = {
    "sell_price_main",
    "availability",
    *(f"type_{index}_discount" for index in range(7)),
}
METRIC_PRECISION_COLUMNS = {"sales", "weight"}
INTEGER_SEMANTIC_COLUMNS = {
    "total_orders",
    "holiday",
    "shops_closed",
    "winter_school_holidays",
    "school_holidays",
    "sales_hat",
}
CATEGORICAL_TEXT_COLUMNS = {
    "warehouse",
    "holiday_name",
    "name",
    "L1_category_name_en",
    "L2_category_name_en",
    "L3_category_name_en",
    "L4_category_name_en",
}
ID_COLUMNS = {"unique_id", "product_unique_id"}


@dataclass(frozen=True)
class Conversion:
    column: str
    before: str
    after: str
    reason: str


def mib(byte_count: int) -> float:
    return byte_count / (1024**2)


def deep_memory(frame: pd.DataFrame) -> int:
    return int(frame.memory_usage(index=True, deep=True).sum())


def smallest_integer_dtype(series: pd.Series) -> str | np.dtype:
    """Return the smallest exact integer dtype, preserving missing values."""

    values = series.dropna()
    if values.empty:
        return "UInt8" if series.isna().any() else np.dtype("uint8")
    numeric = pd.to_numeric(values, errors="raise")
    rounded = np.rint(numeric.to_numpy(dtype=np.float64))
    if not np.array_equal(numeric.to_numpy(dtype=np.float64), rounded):
        raise ValueError(f"{series.name} is not integer-valued and cannot be downcast exactly.")

    minimum = int(rounded.min())
    maximum = int(rounded.max())
    nullable = bool(series.isna().any())
    if minimum >= 0:
        candidates = (
            ("UInt8", np.dtype("uint8")),
            ("UInt16", np.dtype("uint16")),
            ("UInt32", np.dtype("uint32")),
            ("UInt64", np.dtype("uint64")),
        )
        for nullable_name, numpy_dtype in candidates:
            if maximum <= np.iinfo(numpy_dtype).max:
                return nullable_name if nullable else numpy_dtype
    else:
        candidates = (
            ("Int8", np.dtype("int8")),
            ("Int16", np.dtype("int16")),
            ("Int32", np.dtype("int32")),
            ("Int64", np.dtype("int64")),
        )
        for nullable_name, numpy_dtype in candidates:
            limits = np.iinfo(numpy_dtype)
            if minimum >= limits.min and maximum <= limits.max:
                return nullable_name if nullable else numpy_dtype
    raise ValueError(f"{series.name} values exceed supported 64-bit integer ranges.")


def safe_float32(series: pd.Series) -> tuple[pd.Series, float, float]:
    """Downcast to float32 only when values remain numerically close."""

    converted = series.astype(np.float32)
    original = series.to_numpy(dtype=np.float64, na_value=np.nan)
    roundtrip = converted.to_numpy(dtype=np.float64, na_value=np.nan)
    finite = np.isfinite(original)
    if not finite.any():
        return converted, 0.0, 0.0
    absolute_error = np.abs(original[finite] - roundtrip[finite])
    scale = np.maximum(np.abs(original[finite]), 1.0)
    relative_error = absolute_error / scale
    maximum_absolute = float(absolute_error.max(initial=0.0))
    maximum_relative = float(relative_error.max(initial=0.0))
    if maximum_relative > 1e-6:
        raise ValueError(
            f"{series.name} float32 relative error {maximum_relative:.3g} exceeds 1e-6."
        )
    return converted, maximum_absolute, maximum_relative


def optimize_in_memory(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[Conversion]]:
    """Apply reviewed, schema-aware dtype candidates in place."""

    conversions: list[Conversion] = []
    for column in frame.columns:
        before = str(frame[column].dtype)

        if column == "date":
            parsed = pd.to_datetime(frame[column], format="%Y-%m-%d", errors="raise")
            if parsed.isna().any():
                raise ValueError("date contains missing values after parsing.")
            frame[column] = parsed
            reason = "exact YYYY-MM-DD parsing"

        elif column in METRIC_PRECISION_COLUMNS:
            frame[column] = frame[column].astype(np.float64)
            reason = "kept float64 to preserve exact WMAE target/weight precision"

        elif column in FLOAT32_COLUMNS:
            converted, max_abs, max_rel = safe_float32(frame[column])
            frame[column] = converted
            reason = (
                "float32 candidate; "
                f"max_abs_error={max_abs:.6g}, max_relative_error={max_rel:.6g}"
            )

        elif column in INTEGER_SEMANTIC_COLUMNS:
            dtype = smallest_integer_dtype(frame[column])
            frame[column] = frame[column].astype(dtype)
            reason = "smallest exact integer dtype"

        elif column in ID_COLUMNS:
            unique_ratio = frame[column].nunique(dropna=False) / max(len(frame), 1)
            if unique_ratio <= 0.10:
                frame[column] = frame[column].astype("category")
                reason = f"repeated ID encoded as category; unique_ratio={unique_ratio:.4f}"
            else:
                dtype = smallest_integer_dtype(frame[column])
                frame[column] = frame[column].astype(dtype)
                reason = f"ID downcast exactly; unique_ratio={unique_ratio:.4f}"

        elif column in CATEGORICAL_TEXT_COLUMNS:
            unique_ratio = frame[column].nunique(dropna=False) / max(len(frame), 1)
            candidate = frame[column].astype("category")
            if candidate.memory_usage(deep=True) < frame[column].memory_usage(deep=True):
                frame[column] = candidate
                reason = f"repeated text encoded as category; unique_ratio={unique_ratio:.4f}"
            else:
                reason = "kept string; category would not reduce memory"

        else:
            reason = "kept unchanged; no reviewed conversion rule"

        conversions.append(
            Conversion(
                column=column,
                before=before,
                after=str(frame[column].dtype),
                reason=reason,
            )
        )
    return frame, conversions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"CSV to profile (default: {DEFAULT_INPUT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = args.input.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path, low_memory=False)
    before_bytes = deep_memory(frame)
    original_shape = frame.shape
    original_dtypes = {column: str(dtype) for column, dtype in frame.dtypes.items()}

    frame, conversions = optimize_in_memory(frame)
    after_bytes = deep_memory(frame)
    reduction = 1.0 - (after_bytes / before_bytes)
    peak_rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    print(f"file: {path}")
    print(f"shape: {original_shape[0]} rows x {original_shape[1]} columns")
    print(f"memory_before: {before_bytes} bytes ({mib(before_bytes):.2f} MiB)")
    print(f"memory_after: {after_bytes} bytes ({mib(after_bytes):.2f} MiB)")
    print(f"memory_reduction: {reduction:.2%}")
    print(f"peak_process_rss: {peak_rss_mib:.2f} MiB")
    print("candidate_dtypes:")
    for conversion in conversions:
        print(
            f"  {conversion.column}: {conversion.before} -> {conversion.after}; "
            f"{conversion.reason}"
        )
    print(f"original_dtypes: {original_dtypes}")
    print("processed_data_written: no")


if __name__ == "__main__":
    main()
