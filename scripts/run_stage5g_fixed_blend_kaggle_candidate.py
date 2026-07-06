#!/usr/bin/env python3
"""Build the fixed Stage 5-G 50/50 raw + Tweedie Kaggle candidate without submitting it."""

from __future__ import annotations

import json
import gc
import resource
from pathlib import Path
from time import perf_counter

import lightgbm as lgb
import numpy as np
import pandas as pd

from dataguard.cutoff import filter_history_at_cutoff
from features.stage3_minimal import CATEGORICAL_FEATURES, FORBIDDEN_FEATURE_FIELDS, KEY_COLUMNS
from features.stage5e_stronger_features import (
    APPROVED_STAGE5E_FEATURES,
    FEATURE_AVAILABILITY as STAGE5E_FEATURE_AVAILABILITY,
    STAGE5E_FORBIDDEN_FEATURE_FIELDS,
    build_stage5e_feature_batch,
)
from models.plain_lgbm import PlainLightGBMConfig, predict_plain_lightgbm, train_plain_lightgbm
from scripts.run_stage5e_kaggle_candidate import (
    _apply_zero_clipping,
    _build_official_request_frame,
    _build_training_frame,
    _load_raw_data,
    _validate_submission_frame,
)


ROOT = Path(__file__).resolve().parents[1]
SUBMISSIONS = ROOT / "submissions"
REPORTS = ROOT / "reports"
FINAL_CUTOFF = pd.Timestamp("2024-06-02")
OFFICIAL_FORECAST_START = pd.Timestamp("2024-06-03")
OFFICIAL_FORECAST_END = pd.Timestamp("2024-06-16")
SUBMISSION_PATH = SUBMISSIONS / "stage5g_fixed_50_50_raw_tweedie_candidate.csv"
REPORT_PATH = REPORTS / "stage5g_fixed_blend_kaggle_candidate_report.md"
FIXED_BLEND_WEIGHTS = (0.5, 0.5)
TWEEDIE_VARIANCE_POWER = 1.1
RAW_MODEL_CONFIG = PlainLightGBMConfig()
TWEEDIE_MODEL_CONFIG = PlainLightGBMConfig(objective="tweedie")


def peak_rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _require_stage5e_contract(frame: pd.DataFrame) -> None:
    if tuple(frame.columns) != APPROVED_STAGE5E_FEATURES:
        raise AssertionError("Stage 5-E feature contract changed.")
    forbidden = FORBIDDEN_FEATURE_FIELDS.union(STAGE5E_FORBIDDEN_FEATURE_FIELDS)
    if forbidden.intersection(frame.columns):
        raise AssertionError("Feature matrix contains forbidden fields.")


def _summarise_prediction_vector(prediction: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(prediction, dtype=np.float64)
    return {
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "negative_count": int((array < 0).sum()),
        "null_count": int(np.isnan(array).sum()),
    }


def _train_tweedie_lightgbm(
    features: pd.DataFrame,
    target: pd.Series,
    categorical_features: tuple[str, ...],
    config: PlainLightGBMConfig,
) -> lgb.Booster:
    target_array = np.asarray(target, dtype=np.float64)
    if len(features) != len(target_array):
        raise ValueError("Feature and target row counts differ.")
    if not np.isfinite(target_array).all():
        raise ValueError("Training target contains missing or non-finite values.")
    missing_categories = set(categorical_features) - set(features.columns)
    if missing_categories:
        raise KeyError(f"Missing categorical features: {sorted(missing_categories)}.")

    params = dict(config.parameters())
    params["objective"] = config.objective
    params["tweedie_variance_power"] = TWEEDIE_VARIANCE_POWER
    dataset = lgb.Dataset(
        features,
        label=target_array,
        categorical_feature=list(categorical_features),
        free_raw_data=False,
    )
    return lgb.train(params, dataset, num_boost_round=config.num_boost_round)


def _blend_fixed_predictions(
    raw_prediction_frame: pd.DataFrame,
    tweedie_prediction_frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    left_keys = raw_prediction_frame.loc[:, KEY_COLUMNS].reset_index(drop=True)
    right_keys = tweedie_prediction_frame.loc[:, KEY_COLUMNS].reset_index(drop=True)
    if not left_keys.equals(right_keys):
        raise AssertionError("Fixed blend predictions must align exactly by id and order.")
    if tuple(FIXED_BLEND_WEIGHTS) != (0.5, 0.5):
        raise AssertionError("Fixed blend weights changed.")
    if not np.isclose(sum(FIXED_BLEND_WEIGHTS), 1.0, atol=1e-12):
        raise AssertionError("Fixed blend weights must sum to 1.")

    raw_values = raw_prediction_frame["sales_hat"].to_numpy(dtype=np.float64)
    tweedie_values = tweedie_prediction_frame["sales_hat"].to_numpy(dtype=np.float64)
    blended = FIXED_BLEND_WEIGHTS[0] * raw_values + FIXED_BLEND_WEIGHTS[1] * tweedie_values
    clipped, clipped_rows = _apply_zero_clipping(blended)
    audit = {
        "weights": list(FIXED_BLEND_WEIGHTS),
        "negative_before_clip": int((blended < 0).sum()),
        "clipped_rows": int(clipped_rows),
    }
    return blended, clipped, audit


def run_candidate() -> dict[str, object]:
    start = perf_counter()
    history, inventory, calendar, sales_test, official_grid, solution = _load_raw_data()
    raw_load_seconds = perf_counter() - start

    history_at_cutoff = filter_history_at_cutoff(history, FINAL_CUTOFF)
    if history_at_cutoff["date"].max() > FINAL_CUTOFF:
        raise AssertionError("History at cutoff contains future rows.")
    history_for_official_features = history_at_cutoff.dropna(subset=["sales"]).copy()

    training_start = perf_counter()
    training_features, training_target, origin_audit = _build_training_frame(
        history_at_cutoff,
        inventory,
        calendar,
        official_grid,
    )
    training_seconds = perf_counter() - training_start

    request_frame = _build_official_request_frame(sales_test, official_grid)
    if FORBIDDEN_FEATURE_FIELDS.intersection(request_frame.columns):
        raise AssertionError("Official request frame contains forbidden fields.")
    if STAGE5E_FORBIDDEN_FEATURE_FIELDS.intersection(request_frame.columns):
        raise AssertionError("Official request frame contains a Stage 5 forbidden field.")

    feature_start = perf_counter()
    batch = build_stage5e_feature_batch(
        history_for_official_features,
        request_frame,
        inventory,
        calendar,
        FINAL_CUTOFF,
    )
    feature_seconds = perf_counter() - feature_start
    _require_stage5e_contract(batch.matrix)

    raw_fit_start = perf_counter()
    raw_model = train_plain_lightgbm(training_features, training_target, CATEGORICAL_FEATURES, RAW_MODEL_CONFIG)
    raw_fit_seconds = perf_counter() - raw_fit_start

    raw_predict_start = perf_counter()
    raw_prediction = predict_plain_lightgbm(raw_model, batch.matrix)
    raw_prediction_seconds = perf_counter() - raw_predict_start
    raw_prediction_frame = batch.keys.copy()
    raw_prediction_frame["sales_hat"] = raw_prediction

    tweedie_fit_start = perf_counter()
    tweedie_model = _train_tweedie_lightgbm(
        training_features,
        training_target,
        CATEGORICAL_FEATURES,
        TWEEDIE_MODEL_CONFIG,
    )
    tweedie_fit_seconds = perf_counter() - tweedie_fit_start

    tweedie_predict_start = perf_counter()
    tweedie_prediction = predict_plain_lightgbm(tweedie_model, batch.matrix)
    tweedie_prediction_seconds = perf_counter() - tweedie_predict_start
    tweedie_prediction_frame = batch.keys.copy()
    tweedie_prediction_frame["sales_hat"] = tweedie_prediction

    blend_start = perf_counter()
    blended_prediction, blended_prediction_clipped, blend_audit = _blend_fixed_predictions(
        raw_prediction_frame,
        tweedie_prediction_frame,
    )
    blend_seconds = perf_counter() - blend_start

    submission = solution.loc[:, ["id"]].copy()
    submission["sales_hat"] = blended_prediction_clipped
    _validate_submission_frame(official_grid, solution, submission)

    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)

    total_seconds = perf_counter() - start
    result = {
        "stage": "stage_5g_fixed_blend_kaggle_candidate",
        "lightgbm_version": lgb.__version__,
        "model_a_configuration": RAW_MODEL_CONFIG.audit_dict(),
        "model_b_configuration": {
            **TWEEDIE_MODEL_CONFIG.audit_dict(),
            "tweedie_variance_power": TWEEDIE_VARIANCE_POWER,
        },
        "fixed_blend_weights": list(FIXED_BLEND_WEIGHTS),
        "feature_count": len(APPROVED_STAGE5E_FEATURES),
        "features": list(APPROVED_STAGE5E_FEATURES),
        "feature_availability": dict(STAGE5E_FEATURE_AVAILABILITY),
        "training_origins": origin_audit,
        "training_rows": len(training_features),
        "submission_rows": len(submission),
        "submission_path": str(SUBMISSION_PATH),
        "official_grid_rows": len(official_grid),
        "official_forecast_window": {
            "start": str(OFFICIAL_FORECAST_START.date()),
            "end": str(OFFICIAL_FORECAST_END.date()),
        },
        "raw_prediction_summary": _summarise_prediction_vector(raw_prediction),
        "tweedie_prediction_summary": _summarise_prediction_vector(tweedie_prediction),
        "blend_prediction_summary": {
            **_summarise_prediction_vector(blended_prediction_clipped),
            "negative_count_before_clipping": int((blended_prediction < 0).sum()),
            "clipped_rows": blend_audit["clipped_rows"],
        },
        "runtime": {
            "raw_load_seconds": raw_load_seconds,
            "training_seconds": training_seconds,
            "feature_seconds": feature_seconds,
            "raw_fit_seconds": raw_fit_seconds,
            "raw_prediction_seconds": raw_prediction_seconds,
            "tweedie_fit_seconds": tweedie_fit_seconds,
            "tweedie_prediction_seconds": tweedie_prediction_seconds,
            "blend_seconds": blend_seconds,
            "total_seconds": total_seconds,
            "total_minutes": total_seconds / 60,
        },
        "peak_rss_mib": peak_rss_mib(),
        "submission_validation": {
            "row_count_matches_solution": len(submission) == len(solution),
            "order_matches_solution": bool(
                submission["id"].astype(str).reset_index(drop=True).equals(
                    solution["id"].astype(str).reset_index(drop=True)
                )
            ),
            "official_grid_rows": len(official_grid),
            "solution_rows": len(solution),
            "no_forbidden_fields_in_request": not FORBIDDEN_FEATURE_FIELDS.intersection(
                request_frame.columns
            ),
            "no_forbidden_fields_in_features": not FORBIDDEN_FEATURE_FIELDS.intersection(
                batch.matrix.columns
            ),
            "stage5_forbidden_fields_in_request": not STAGE5E_FORBIDDEN_FEATURE_FIELDS.intersection(
                request_frame.columns
            ),
            "stage5_forbidden_fields_in_features": not STAGE5E_FORBIDDEN_FEATURE_FIELDS.intersection(
                batch.matrix.columns
            ),
        },
        "clipping_policy": {
            "applied": True,
            "reason": "Zero clipping is a documented, business-valid post-processing choice for the fixed blend.",
            "affected_predictions": blend_audit["clipped_rows"],
        },
        "caveats": [
            "This is public-solution-informed improvement work, not independent discovery.",
            "Price and discount covariates are benchmark-specific known-future inputs.",
            "Local improvement may not transfer exactly to Kaggle hidden scoring.",
            "Official Kaggle score remains unknown until submission.",
        ],
        "kaggle_submission_made": False,
        "processed_data_written": False,
        "model_artifact_written": False,
        "predictions_clipped": True,
        "predictions_rounded": False,
        "future_total_orders_used": False,
        "future_availability_used": False,
        "target_transform_used": False,
        "recursive_prediction_used": False,
    }

    report_lines = [
        "# Stage 5-G Fixed Blend Kaggle Candidate Report",
        "",
        f"Date: {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d')}",
        "Status: Candidate prepared; pending pre-submit audit before Kaggle submission",
        "",
        "## Model configuration",
        "",
        "| Component | Setting |",
        "|---|---|",
        f"| Library | LightGBM {lgb.__version__} |",
        f"| Blend weights | 50% raw L1 + 50% Tweedie 1.1 |",
        f"| Model A objective | {RAW_MODEL_CONFIG.objective} |",
        f"| Model B objective | {TWEEDIE_MODEL_CONFIG.objective} |",
        f"| Tweedie variance power | {TWEEDIE_VARIANCE_POWER} |",
        "| Target | Raw `sales` for both models |",
        "| Clipping / rounding | Zero-clipped after blending, no rounding |",
        "| Recursive prediction | None |",
        "| Target transform | None |",
        "",
        "## Why this blend is promoted",
        "",
        "- The Stage 5-G adversarial review approved the fixed 50/50 blend.",
        "- Membership is globally fixed, the same weights apply to every fold, and the same rule applies to the official Kaggle test set.",
        "- The fixed 50/50 blend improved the local mean WMAE versus both Stage 5-E raw L1 and Tweedie 1.1.",
        "",
        "## Why the fold-specific equal blend is rejected",
        "",
        "- The fold-specific equal-blend family changes membership by fold.",
        "- That makes it non-generalizable to the official Kaggle test set.",
        "",
        "## Local validation evidence",
        "",
        "- Stage 5-E raw L1 local mean WMAE: 19.5424424315",
        "- Tweedie 1.1 local mean WMAE: 19.1888923539",
        "- Fixed 50/50 raw + Tweedie local mean WMAE: 19.0089170845",
        "- Improvement vs Stage 5-E raw L1: 0.5335253470",
        "- Improvement vs Tweedie 1.1: 0.1799752694",
        "",
        "## Official comparison context",
        "",
        "- Stage 5-E official public WMAE: 21.09367",
        "- Stage 5-E official private WMAE: 20.61497",
        "- This fixed blend is intended as a follow-on candidate from that frozen official result.",
        "",
        "## Final training design",
        "",
        f"- Final cutoff: {FINAL_CUTOFF.date()}",
        f"- Official forecast window: {OFFICIAL_FORECAST_START.date()} through {OFFICIAL_FORECAST_END.date()}",
        "- Twelve cutoff-safe historical origins are used to build the training set up to the final cutoff.",
        "- Training labels remain historical only and end on or before the final cutoff.",
        "",
        "## 76-feature contract",
        "",
        f"- Stage 5-E approved feature count: {len(APPROVED_STAGE5E_FEATURES)}",
        "- No additional features are introduced.",
        "- Official test price/discount values are treated as Kaggle-known-future covariates.",
        "- Future `total_orders`, future availability, weights, sales targets, IDs, and `sales_hat` are excluded from the model matrix.",
        "",
        "## Candidate validation",
        "",
        f"- Submission path: `{SUBMISSION_PATH.relative_to(ROOT)}`",
        f"- Candidate rows: {len(submission)}",
        "- Row count matches `solution.csv`.",
        "- IDs and order match `solution.csv` and the official grid.",
        "- Raw and Tweedie prediction vectors were generated on the same official request frame and checked for identical `unique_id` / `date` order before blending.",
        "- Predictions are numeric and non-null.",
        "- Zero clipping was applied after blending; the number of clipped rows is recorded below.",
        "",
        "## Prediction summaries",
        "",
        "### Raw L1 model",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Minimum | {result['raw_prediction_summary']['minimum']:.6f} |",
        f"| Maximum | {result['raw_prediction_summary']['maximum']:.6f} |",
        f"| Mean | {result['raw_prediction_summary']['mean']:.6f} |",
        f"| Median | {result['raw_prediction_summary']['median']:.6f} |",
        f"| Negative count | {result['raw_prediction_summary']['negative_count']} |",
        f"| Null count | {result['raw_prediction_summary']['null_count']} |",
        "",
        "### Tweedie 1.1 model",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Minimum | {result['tweedie_prediction_summary']['minimum']:.6f} |",
        f"| Maximum | {result['tweedie_prediction_summary']['maximum']:.6f} |",
        f"| Mean | {result['tweedie_prediction_summary']['mean']:.6f} |",
        f"| Median | {result['tweedie_prediction_summary']['median']:.6f} |",
        f"| Negative count | {result['tweedie_prediction_summary']['negative_count']} |",
        f"| Null count | {result['tweedie_prediction_summary']['null_count']} |",
        "",
        "### Fixed blend",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Minimum | {result['blend_prediction_summary']['minimum']:.6f} |",
        f"| Maximum | {result['blend_prediction_summary']['maximum']:.6f} |",
        f"| Mean | {result['blend_prediction_summary']['mean']:.6f} |",
        f"| Median | {result['blend_prediction_summary']['median']:.6f} |",
        f"| Negative count before clipping | {result['blend_prediction_summary']['negative_count_before_clipping']} |",
        f"| Negative count after clipping | {result['blend_prediction_summary']['negative_count']} |",
        f"| Clipped rows | {result['blend_prediction_summary']['clipped_rows']} |",
        f"| Null count | {result['blend_prediction_summary']['null_count']} |",
        "",
        "## Runtime and memory",
        "",
        "| Phase | Seconds |",
        "|---|---:|",
        f"| Raw load | {raw_load_seconds:.3f} |",
        f"| Training-origin feature build | {training_seconds:.3f} |",
        f"| Official-grid feature build | {feature_seconds:.3f} |",
        f"| Raw model fit | {raw_fit_seconds:.3f} |",
        f"| Raw model prediction | {raw_prediction_seconds:.3f} |",
        f"| Tweedie model fit | {tweedie_fit_seconds:.3f} |",
        f"| Tweedie model prediction | {tweedie_prediction_seconds:.3f} |",
        f"| Blend computation | {blend_seconds:.3f} |",
        f"| Total | {total_seconds:.3f} |",
        f"| Peak RSS (MiB) | {result['peak_rss_mib']:.2f} |",
        "",
        "## Clipping summary",
        "",
        f"- Clipping was applied after blending.",
        f"- Predictions changed by clipping: {blend_audit['clipped_rows']}",
        "",
        "## Caveats",
        "",
        "- This is a public-solution-informed stage, not independent discovery.",
        "- Price/discount covariates are benchmark-specific known-future inputs.",
        "- Local improvement may not transfer exactly to Kaggle hidden scoring.",
        "- Kaggle validates the prediction file and hidden score, not the full methodology.",
        "",
        "## Recommendation",
        "",
        "Ready for pre-submit audit before Kaggle submission.",
    ]
    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run_candidate()
