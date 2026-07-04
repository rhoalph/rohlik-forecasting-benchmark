# Raw Data Contract

## Scope and inspection method

This contract describes the six competition CSV files unpacked under `data/raw`. Counts and nulls were computed over every row, not from samples. Date ranges are inclusive. No feature engineering, imputation, or model training was performed.

## Forecast unit, target, and horizon

- Forecast unit: one requested `unique_id` and `date` pair. `unique_id` identifies a warehouse-specific inventory item.
- Target: `sales` in `sales_train.csv`; the required submission prediction is named `sales_hat`.
- Last training date: 2024-06-02.
- Kaggle forecast dates: 2024-06-03 through 2024-06-16, exactly 14 calendar days.
- Test grid: 47,021 unique item-date rows covering 3,625 `unique_id` values. It is sparse: 2,688 IDs have all 14 dates and the remainder have 1–13 requested dates. Missing item-date rows must not automatically be interpreted as zero demand.
- Geographic scope: seven warehouses (`Brno_1`, `Budapest_1`, `Frankfurt_1`, `Munich_1`, `Prague_1`, `Prague_2`, `Prague_3`).
- Official scoring weights: `test_weights.csv` supplies a weight by `unique_id`. The weights belong to evaluation, not to the target or raw demand history.

## Table summary

| File | Rows | Natural key | Date range | Missing values |
|---|---:|---|---|---|
| `calendar.csv` | 23,016 | (`warehouse`, `date`) | 2016-01-01 to 2024-12-31 | `holiday_name`: 22,086; all other columns: 0 |
| `inventory.csv` | 5,432 | `unique_id` | Not applicable | None |
| `sales_train.csv` | 4,007,419 | (`unique_id`, `date`) | 2020-08-01 to 2024-06-02 | `sales`: 52; `total_orders`: 52; all other columns: 0 |
| `sales_test.csv` | 47,021 | (`unique_id`, `date`) | 2024-06-03 to 2024-06-16 | None |
| `test_weights.csv` | 5,390 | `unique_id` | Not applicable | None |
| `solution.csv` | 47,021 | `id` | Encodes the test dates | None |

All natural keys above are unique. Every train and test ID joins to inventory. Every train and test (`warehouse`, `date`) joins to calendar. Every test ID joins to a weight. The solution IDs match the test keys exactly.

The 52 missing `sales` values and 52 missing `total_orders` values occur on exactly the same rows, dated 2021-05-21 through 2021-12-10: 46 Munich rows and 6 Frankfurt rows. Stage 1 must define explicit exclusion or handling; these rows cannot be scored as labels.

## `calendar.csv`

One row per warehouse and calendar date. Each warehouse has 3,288 dates spanning the full range, with no duplicate keys.

| Column | Type | Nulls | Contract classification | Notes |
|---|---|---:|---|---|
| `date` | date string | 0 | Known future | Calendar date; join only with both date and warehouse. |
| `holiday_name` | string | 22,086 | Known future | Null means no named holiday; 37 non-null names. |
| `holiday` | binary integer | 0 | Known future | Public/special holiday indicator. |
| `shops_closed` | binary integer | 0 | Known future | Scheduled closure indicator. |
| `winter_school_holidays` | binary integer | 0 | Known future | School-calendar indicator. |
| `school_holidays` | binary integer | 0 | Known future | School-calendar indicator. |
| `warehouse` | string | 0 | Static key | Required because calendars differ by warehouse/country. |

Calendar fields are legitimate known-future inputs. The calendar extends beyond the test horizon, so code must still restrict joins to the requested prediction dates and must never infer target values from future sales rows.

## `inventory.csv`

One static metadata row per warehouse-specific `unique_id`. There are 5,432 IDs and 2,670 cross-warehouse `product_unique_id`/`name` values.

| Column | Type | Nulls | Cardinality | Contract classification |
|---|---|---:|---:|---|
| `unique_id` | integer | 0 | 5,432 | Static primary key |
| `product_unique_id` | integer | 0 | 2,670 | Static product identity across warehouses |
| `name` | string | 0 | 2,670 | Static, anonymized product name |
| `L1_category_name_en` | string | 0 | 3 | Static hierarchy |
| `L2_category_name_en` | string | 0 | 47 | Static hierarchy |
| `L3_category_name_en` | string | 0 | 177 | Static hierarchy |
| `L4_category_name_en` | string | 0 | 68 | Static hierarchy |
| `warehouse` | string | 0 | 7 | Static warehouse assignment |

The file is treated as a competition-provided snapshot. If metadata could change in a real deployment, a point-in-time source would be required; this dataset provides no effective dates.

## `sales_train.csv`

Historical daily item records. It contains 5,390 IDs and 1,402 distinct dates. Coverage starts later for Frankfurt (2021-12-08) and Munich (2021-05-20) than for the other warehouses (2020-08-01). All warehouses end on 2024-06-02.

| Column | Type | Nulls | Contract classification | Availability rule |
|---|---|---:|---|---|
| `unique_id` | integer | 0 | Static key | May identify history only through the cutoff. |
| `date` | date string | 0 | Time key | Determines cutoff eligibility. |
| `warehouse` | string | 0 | Static key | Must agree with inventory. |
| `total_orders` | float | 52 | Historical; benchmark-known future in test; operational-risk field | Historical values are safe through cutoff. Future values are supplied by Kaggle, but may represent realized daily activity and need human approval before use. |
| `sales` | float | 52 | Target; historical only | Allowed as a label/history only on or before cutoff. Any value after cutoff is forbidden for feature computation. |
| `sell_price_main` | float | 0 | Historical; benchmark-known future in test | Future values may be used only under a declared assumption that the supplied price is known/scheduled at forecast origin. |
| `availability` | float | 0 | Historical only | Absent from test. Contemporaneous validation availability would create train/test mismatch and possible leakage; only cutoff-safe lagged history may later be considered. |
| `type_0_discount` … `type_6_discount` | float | 0 | Historical; benchmark-known future in test | Future values may be used only under a declared scheduled-promotion assumption. Negative historical values exist in types 0, 4, and 6 and require investigation rather than silent clipping. |

Observed ranges include zero sales, `availability` from 0.01 to 1.0, and prices from 0.02 to 21,682.99. Range extremes are not changed at Stage 0.

## `sales_test.csv`

The prediction grid has the following columns, all with zero missing values:

`unique_id`, `date`, `warehouse`, `total_orders`, `sell_price_main`, `type_0_discount`, `type_1_discount`, `type_2_discount`, `type_3_discount`, `type_4_discount`, `type_5_discount`, `type_6_discount`.

The absence of `sales` identifies it as the hidden target. The absence of `availability` means validation must not rely on same-day availability. Fields present in test are available to a Kaggle notebook, but presence alone does not prove they would be known at a real operational forecast origin:

- `date`, keys, and joined calendar fields: known future.
- price and discount fields: benchmark-known future; operational status requires confirmation that they are scheduled values.
- `total_orders`: benchmark-known future but high leakage/business-realism risk because it may summarize realized future-day order volume.

## `test_weights.csv`

| Column | Type | Nulls | Notes |
|---|---|---:|---|
| `unique_id` | integer | 0 | 5,390 unique IDs; all test IDs are covered. |
| `weight` | float | 0 | Evaluation weight, range approximately 0.0401 to 73.2817. |

There are 1,765 weighted IDs absent from the test prediction grid. Weights must be joined by `unique_id` for scoring and must not be joined positionally. Treat `weight` as evaluation metadata, not a predictive feature.

## `solution.csv`

| Column | Type | Nulls | Notes |
|---|---|---:|---|
| `id` | string | 0 | Unique key formatted as `<unique_id>_<YYYY-MM-DD>`; exact match to all 47,021 test keys. |
| `sales_hat` | integer in template | 0 | Placeholder target prediction; every supplied value is zero. |

The zero values are a submission template, not labels or a baseline result. Predictions must preserve the exact ID set and order required by submission generation.

## Field availability summary

### Historical only

- `sales` through each forecast cutoff.
- `availability` through each cutoff.
- Past values of `total_orders`, prices, and discounts.
- Any target-derived lag, rolling statistic, aggregate, or encoding, computed strictly from rows at or before cutoff.

### Known future

- Forecast `date` and requested `unique_id` grid.
- Static warehouse/product metadata.
- Calendar and holiday fields for the forecast dates.
- For the Kaggle benchmark only, test-supplied price, discount, and `total_orders` values are technically available. Their operational interpretation must remain explicit.

### Static metadata

- Inventory product identity and category hierarchy.
- Warehouse assignment.
- Evaluation weight is static but evaluation-only.

### Forbidden or leakage-risk inputs

- Any `sales` value after a forecast cutoff.
- Same-window target aggregates, target encodings, rolling windows, or lags that include validation/test labels.
- Same-day `availability` in validation, because it is unavailable in test.
- Future `total_orders` under a strict operational backtest unless its forecast-origin availability is proven.
- Price or promotion values presented as operationally known without confirming they were scheduled at forecast origin.
- `sales_hat` from the solution template as training information.
- Evaluation weights as a model input unless explicitly reviewed; they are intended for scoring.
- Rows from train and test concatenated before target-derived transforms are fitted.
