"""Code-backed manifest for transparent training and live prediction lineage."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES
from .feature_sets import FEATURE_SCHEMAS


def _stage(
    stage_id: str,
    name: str,
    purpose: str,
    inputs: list[str],
    operations: list[str],
    outputs: list[str],
    fit_scope: str,
    leakage_boundary: str,
) -> dict:
    return {
        "id": stage_id,
        "name": name,
        "purpose": purpose,
        "inputs": inputs,
        "operations": operations,
        "outputs": outputs,
        "fit_scope": fit_scope,
        "leakage_boundary": leakage_boundary,
    }


def _historical_field_status(
    runners_path: Path = Path("data/processed/historical/runners.csv.gz"),
) -> list[dict[str, str]]:
    fields = {
        "horse_age": (
            "horse_age",
            "Year-propagated from timestamped HKJC horse profile page ages and identity-matched archived horse snapshots",
        ),
        "horse_country": (
            "horse_country",
            "Static Country of Origin from the official HKJC horse profile page",
        ),
        "horse_type": (
            "horse_type",
            "Final component of official HKJC Colour / Sex from the horse profile page",
        ),
        "horse_gear": (
            "gear",
            "Race-specific HKJC gear; blank or -- is the explicit NONE category rather than missing data",
        ),
    }
    available: dict[str, pd.Series] = {}
    if runners_path.exists():
        wanted = {column for column, _reason in fields.values()}
        frame = pd.read_csv(runners_path, usecols=lambda column: column in wanted, low_memory=False)
        available = {column: frame[column] for column in frame.columns}
    status = []
    for field, (column, reason) in fields.items():
        values = available.get(column)
        if values is None or values.empty:
            coverage = 0.0
        else:
            valid = values.notna()
            if not pd.api.types.is_numeric_dtype(values):
                text = values.astype("string").str.strip().str.upper()
                valid &= text.ne("") & text.ne("UNKNOWN")
            coverage = float(valid.mean())
        status.append({"field": field, "value": f"{coverage:.1%} populated", "reason": reason})
    return status


def pipeline_manifest() -> dict:
    training = [
        _stage(
            "train-source",
            "Load historical runner and race records",
            "Create one source row per horse entered in a completed race.",
            [
                "Legacy runs.csv + races.csv for dates before 2005-01-01",
                "Canonical normalized runners.csv.gz for 2005 onward",
            ],
            [
                "Parse race dates and numeric finishing positions",
                "Namespace legacy and canonical race_id and horse_id values",
                "Concatenate non-overlapping timelines and sort by date, race number, race_id, horse number",
            ],
            ["Chronologically ordered runner table", "One row per race_id + horse_no"],
            "No parameters fitted",
            "Canonical rows begin in 2005; legacy rows are cut at 2004-12-31 to prevent duplicate races.",
        ),
        _stage(
            "train-filter",
            "Validate outcomes and market rows",
            "Keep races that have a usable winner target and complete WIN odds.",
            ["result", "win_odds", "race_id"],
            [
                "Extract the leading integer from finishing position text",
                "Drop runners without a numeric result",
                "Drop races without at least one winner",
                "Drop canonical races unless every runner has decimal WIN odds greater than 1",
                "Set target_win = 1 when result = 1; split dead-heat target_probability across winners",
                "Set market_raw = 1 / win_odds and normalize market_probability to sum to 1 per race",
            ],
            ["target_win", "target_probability", "market_probability"],
            "No parameters fitted",
            "Result and current-race final odds are labels/benchmarks. Only lagged odds from earlier races may enter a fundamental feature schema.",
        ),
        _stage(
            "train-history",
            "Build point-in-time horse history",
            "Turn earlier runs into lagged form features without reading the current result.",
            ["horse_id", "date", "won", "result", "win_odds", "finish_time"],
            [
                "Group rows by horse_id after chronological sorting",
                "prior_starts = count of earlier rows",
                "prior_win_rate, prior_avg_result, prior_avg_odds = expanding statistics after shift(1)",
                "last_result, last_win_odds, last_finish_time = group shift(1)",
                "field_size = runner count within race_id",
            ],
            [
                "prior_starts", "prior_win_rate", "prior_avg_result", "prior_avg_odds",
                "last_result", "last_win_odds", "last_finish_time", "field_size",
            ],
            "No parameters fitted",
            "All horse-history calculations shift before expanding, so the current race result cannot enter its own features.",
        ),
        _stage(
            "train-split",
            "Split complete races by time",
            "Create train, validation, and final test windows without splitting runners from the same race.",
            ["Unique race_id ordered by date, race_no, race_id"],
            [
                "First 70% of races -> train",
                "Next 15% of races -> validation",
                "Final 15% of races -> test",
                "Apply race_id sets back to runner rows",
            ],
            ["RaceSplits.train", "RaceSplits.validation", "RaceSplits.test"],
            "Split boundaries are deterministic fractions of chronologically ordered races",
            "All preprocessing and estimator fitting occurs after this split. Test races remain untouched until final evaluation.",
        ),
        _stage(
            "train-preprocess",
            "Fit schema-specific model preprocessing",
            "Convert the selected versioned feature contract into an estimator matrix.",
            ["Numeric columns declared by the selected schema", "Categorical columns declared by the selected schema"],
            [
                "Numeric: median imputation fitted on train rows",
                "Numeric: add missing-value indicator columns",
                "Numeric: StandardScaler mean and variance fitted on train rows",
                "Categorical: most-frequent imputation fitted on train rows",
                "Categorical: one-hot encoding; unknown values ignored; categories with frequency below 20 grouped",
            ],
            ["Dense numeric + one-hot model matrix"],
            "The scikit-learn Pipeline fits preprocessing only when model.fit(train) executes",
            "Validation and test rows only call transform; their medians, categories, means, and variances are never fitted.",
        ),
        _stage(
            "train-model",
            "Fit fundamental winner model",
            "Estimate a raw winning strength from horse and race features only.",
            ["Preprocessed train matrix", "target_win"],
            [
                "Logistic option: max_iter=1000, configurable C and class_weight",
                "Boosted option: HistGradientBoostingClassifier with configurable learning rate, leaves, iterations, and L2",
                "predict_proba returns the positive-class score",
                "Normalize raw scores within race_id so runner probabilities sum to 1",
            ],
            ["fundamental_raw_probability per runner"],
            "Estimator fitted on train races only",
            "Current WIN and PLACE prices are excluded from every fundamental schema. Lagged prior-race prices are point-in-time history features.",
        ),
        _stage(
            "train-validation",
            "Fit calibration, market blend, and order exponents",
            "Use validation races to tune probability corrections without touching test races.",
            ["Validation fundamental probabilities", "validation market_probability", "validation results"],
            [
                "Temperature: minimize validation race log loss over exp(log_temperature), bounded to exp(-2.5)..exp(2.5)",
                "Calibrate as p^(1 / temperature), then normalize within race",
                "Market blend: minimize validation race log loss for fundamental^wf * WIN^ww * PLACE^wp, with each available weight bounded 0..4",
                "Fit Benter second- and third-place exponents independently in bounds 0.2..2.0",
            ],
            ["temperature", "fundamental_weight", "market_weight", "place_market_weight", "second exponent", "third exponent"],
            "All four parameter groups fitted on validation races only",
            "Final test metrics are not used to choose calibration, market weights, or finishing-order exponents.",
        ),
        _stage(
            "train-test",
            "Evaluate untouched test races",
            "Measure ranking, probability quality, calibration, and market disagreement.",
            ["Test features", "frozen model artifact", "test target_probability", "test market_probability"],
            [
                "Compute fundamental and combined probabilities",
                "Top-1, Top-3, mean winner rank",
                "Race log loss, race Brier, ECE, pseudo-R2",
                "Incremental pseudo-R2 versus market and disagreement buckets",
            ],
            ["results.json", "results.csv", "model artifacts", "dashboard metrics"],
            "No fitting permitted",
            "Historical market benchmark currently uses result-page final WIN odds; it is not evidence of a tradable pre-race edge.",
        ),
    ]

    live = [
        _stage(
            "live-collect",
            "Collect one timestamped race snapshot",
            "Capture the current field, prices, and horse-page information before prediction.",
            [
                "Rendered HKJC race application",
                "Horse profile and form page",
                "Trackwork, veterinary, movement, and barrier-trial pages",
            ],
            [
                "Read runner status, saddle number, horse/jockey/trainer identifiers, weights, draw, rating and gear",
                "Read WIN and PLACE prices plus odds_updated_at",
                "Store raw response, normalized runners, horse details, model CSV, and missing-feature report",
            ],
            ["raw.json", "runners.json", "horse-details.json", "model.csv", "report.json"],
            "No parameters fitted",
            "received_at and odds_updated_at preserve what was known at prediction time. Scratched, withdrawn, and reserve runners are inactive.",
        ),
        _stage(
            "live-enrich",
            "Join only earlier horse history",
            "Build form values from records dated before the live race.",
            ["horse_id", "race_date", "historical runs", "horse-page form records"],
            [
                "Filter historical rows where race_date < live race_date",
                "Use the previous eligible race for last result, odds, distance, time, and weights",
                "Compute starts, wins, seconds, thirds, average placing, recent placing average, and speed",
            ],
            ["Legacy-compatible model feature row", "missing_features", "prediction_ready"],
            "No parameters fitted",
            "Same-day or future form records are excluded. Missing required history marks the runner unratable instead of inventing values.",
        ),
        _stage(
            "live-map",
            "Map scraper columns into the selected feature contract",
            "Translate the live model CSV into the exact versioned columns accepted by the trained pipeline.",
            ["model.csv", "prediction_ready"],
            [
                "Map exp -> prior_starts",
                "Map won_odds / exp -> prior_win_rate",
                "Map cum_avg_prev_resu -> prior_avg_result",
                "Map prev_odds -> prior_avg_odds and last_win_odds",
                "Map prev_resu -> last_result; fin_time -> last_finish_time; prev_dist -> distance",
                "Set venue, config, and going to UNKNOWN when the legacy model CSV does not provide them",
                "Create any absent feature as null for fitted imputation",
            ],
            ["Live DataFrame containing the artifact's feature schema", "ratable boolean", "WIN and PLACE prices"],
            "Uses preprocessing already fitted in the model artifact",
            "A feature may be collected but still not enter the selected schema. The artifact's feature contract below is authoritative.",
        ),
        _stage(
            "live-fundamental",
            "Predict and calibrate fundamental probability",
            "Generate model-only runner probabilities using frozen training artifacts.",
            ["Live FEATURES", "fitted preprocessing", "fitted estimator", "temperature"],
            [
                "Apply train-fitted imputation, scaling, and one-hot encoding",
                "Predict positive-class score",
                "Normalize scores within race",
                "Apply p^(1 / temperature) and normalize again",
            ],
            ["calibrated fundamental probability"],
            "No live fitting",
            "The runner's current WIN and PLACE odds do not enter this fundamental path.",
        ),
        _stage(
            "live-market",
            "Convert live WIN and PLACE odds into market probabilities",
            "Represent the current public markets as separate probability forecasts.",
            ["decimal WIN odds", "decimal PLACE odds", "race_id", "odds_updated_at"],
            [
                "WIN market_raw = 1 / win_odds for odds greater than 1",
                "PLACE market_raw = 1 / place_odds for available odds greater than 1",
                "Normalize each market independently within race",
            ],
            ["market_probability", "place_market_probability"],
            "No live fitting",
            "Only a snapshot observed before the decision may be used. Result-page final odds are invalid for live decisions.",
        ),
        _stage(
            "live-combine",
            "Combine model and market forecasts",
            "Apply validation-fitted weights to produce the decision probability.",
            ["calibrated fundamental", "WIN market probability", "PLACE market probability", "three validation-fitted weights"],
            [
                "score = fundamental^wf * WIN^ww * PLACE^wp when PLACE is available",
                "Fall back to the fitted fundamental + WIN transform when PLACE is unavailable",
                "Clip each input probability to at least 1e-12",
                "Normalize score within race",
            ],
            ["combined runner probability"],
            "Weights frozen from validation races",
            "Fundamental, WIN market, and PLACE market remain separately reportable; live data never refits their weights.",
        ),
        _stage(
            "live-fallback",
            "Handle first-time and unratable runners",
            "Keep every active runner in the probability distribution without pretending the model has missing history.",
            ["combined probability", "market_probability", "ratable"],
            [
                "Assign unratable runners their normalized public-market probability",
                "Compute remaining probability mass = 1 - unratable mass",
                "Rescale rated runners proportionally into the remaining mass",
            ],
            ["final race-normalized runner strengths"],
            "No fitting",
            "Public fallback is explicit and test-covered; missing horse history cannot silently become a confident model prediction.",
        ),
        _stage(
            "live-pools",
            "Expand runner strengths into pool combinations",
            "Calculate coherent ordered and unordered finishing probabilities.",
            ["runner strengths", "second exponent", "third exponent", "runner IDs"],
            [
                "Plackett-Luce/Benter sequential selection without replacement",
                "WIN: first horse; PLACE: horse appears in top places",
                "QIN: top two in either order; QPL: selected pair both appear in top three",
                "TRI: top three unordered; TIERCE: exact top-three order",
                "FIRST4: first four in any order; QUARTET: exact top-four order",
                "Rank combinations by probability; fair_odds = 1 / probability",
            ],
            ["Ranked pool combinations", "model probability", "model fair odds"],
            "Order exponents frozen from validation races",
            "Current exotic-pool odds remain downstream price comparisons until timestamped pool histories support fitting pool-specific blends.",
        ),
        _stage(
            "live-wager",
            "Compare price and size a paper wager",
            "Convert probability disagreement into a constrained decision.",
            ["model probability", "current pool decimal odds", "bankroll", "probability_std"],
            [
                "expected_value_per_unit = probability * odds - 1",
                "Apply uncertainty haircut to probability when configured",
                "Kelly fraction = (p * odds - 1) / (odds - 1), floored at zero",
                "Multiply by fractional Kelly and enforce combination, race, meeting, and bankroll caps",
                "Abstain when edge or limits fail",
            ],
            ["WagerRecommendation or abstention", "paper execution receipt", "ledger row"],
            "No model fitting",
            "Live authenticated submission remains fail-closed behind credentials, MFA verification, confirmation, and transaction limits.",
        ),
    ]

    return {
        "feature_contracts": {
            name: schema.contract() for name, schema in FEATURE_SCHEMAS.items()
        },
        "feature_contract": {
            "name": "baseline-v1",
            "count": len(FEATURES),
            "numeric": list(NUMERIC_FEATURES),
            "categorical": list(CATEGORICAL_FEATURES),
            "target": "target_win",
            "excluded_market_fields": [
                "win_odds", "place_odds", "market_raw", "market_probability",
                "place_market_raw", "place_market_probability",
            ],
        },
        "historical_placeholders": _historical_field_status(),
        "legacy_age_provenance": {
            "field": "horse_age",
            "value": "100% populated on legacy runner rows",
            "reason": (
                "Direct race-row age from gdaley. Horse IDs are anonymized integers and source "
                "dates are deliberately obscured, so they are not presented as HKJC profile IDs "
                "or true calendar timestamps."
            ),
        },
        "live_defaults": [
            {"field": "venue", "value": "UNKNOWN", "reason": "Legacy model CSV mapping does not currently pass venue"},
            {"field": "config", "value": "UNKNOWN", "reason": "Legacy model CSV mapping does not currently pass course configuration"},
            {"field": "going", "value": "UNKNOWN", "reason": "Legacy model CSV mapping does not currently pass going"},
            {"field": "surface", "value": "null", "reason": "Not available in legacy model CSV mapping"},
            {"field": "prize", "value": "null", "reason": "Not available in legacy model CSV mapping"},
        ],
        "collected_not_consumed_by_baseline": [
            "jockey_id", "trainer_id", "place_odds", "last_speed", "avg_2last",
            "second_count", "third_count", "raced", "prev_wt", "prev_declar_wt",
            "trackwork records", "veterinary records", "movement records", "barrier trials",
        ],
        "training": training,
        "live": live,
    }
