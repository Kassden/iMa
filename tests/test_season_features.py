import unittest

import numpy as np
import pandas as pd

from ima.feature_sets import FEATURE_SCHEMAS, NOTEBOOK_RICH_SCHEMA
from ima.rich_features import prepare_rich_runner_dataset
from ima.season_features import SPEED_EXTRA_FEATURES, SUPPORTED_SCHEMAS, _QUERY_OUTCOMES, build_season_features
from ima.speed_features import build_speed_features
from tests import test_rich_features


class SeasonFeatureTests(unittest.TestCase):
    def setUp(self):
        self.history = test_rich_features.RichFeatureContractTests.source()
        self.query = self.history[self.history.race_id.eq("R2")].copy()
        self.query["horse_age"] = [4, 6]

    def build(self, history=None, query=None, **kwargs):
        return build_season_features(
            self.history if history is None else history,
            self.query if query is None else query, **kwargs,
        )

    def test_untimed_query_preserves_complete_field_ids_order_and_index(self):
        query = self.query.iloc[::-1].drop(columns=["result", "win_odds", "finish_time"])
        query.index = [91, 23]
        output = self.build(query=query)
        self.assertEqual(output.index.tolist(), [91, 23])
        for name in ("race_id", "race_no", "horse_no", "horse_id"):
            pd.testing.assert_series_equal(output[name], query[name])
        self.assertEqual(output.field_size.tolist(), [2.0, 2.0])
        self.assertEqual(output.prior_starts.tolist(), [1.0, 1.0])
        self.assertTrue(output[["result", "finish_seconds", "win_odds", "target_win", "market_probability"]].isna().all().all())

    def test_removing_current_and_later_outcomes_does_not_change_any_feature(self):
        later = self.history.copy()
        later["date"] = "2021-01-01"
        later["race_id"] += "future"
        later["horse_rating"] = 999
        later["result"] = 1
        later["finish_time"] = "0:01.00"
        full = pd.concat([self.history, later], ignore_index=True)
        actual = self.build(history=full)
        columns = list(NOTEBOOK_RICH_SCHEMA.features) + list(SPEED_EXTRA_FEATURES)
        removed = full[full.date.lt("2020-01-10")]
        pd.testing.assert_frame_equal(actual[columns], self.build(history=removed)[columns])
        blank = self.query.copy()
        blank[["result", "finish_time", "win_odds", "lengths_behind", "running_position"]] = np.nan
        pd.testing.assert_frame_equal(actual, self.build(history=removed, query=blank))

    def test_all_same_day_horse_jockey_trainer_draw_and_speed_withheld(self):
        same_day = self.history[self.history.race_id.eq("R1")].copy()
        same_day["date"] = "2020-01-10"
        same_day["race_id"] = "earlier-today"
        same_day["race_no"] = 1
        full = pd.concat([self.history, same_day], ignore_index=True)
        pd.testing.assert_frame_equal(self.build(), self.build(history=full))
        output = self.build(history=same_day)
        self.assertTrue(output.last_result.isna().all())
        self.assertTrue(output[[c for c in SPEED_EXTRA_FEATURES if c != "past_race_speed_support"]].isna().all().all())
        self.assertEqual(output.past_race_speed_support.tolist(), [0.0, 0.0])
        self.assertEqual(output.jockey_starts.tolist(), [0.0, 0.0])

    def test_missing_odds_or_winner_does_not_drop_prior_starters(self):
        historical = self.history[self.history.race_id.eq("R1")].copy()
        historical["win_odds"] = np.nan
        historical["result"] = [2, np.nan]
        historical.loc[historical.horse_id.eq("B"), "finishing_status"] = "PU"
        output = self.build(history=historical)
        self.assertEqual(output.prior_starts.tolist(), [1.0, 1.0])
        self.assertEqual(output.prior_win_rate.tolist(), [0.0, 0.0])
        self.assertTrue(output.prior_avg_odds.isna().all())
        self.assertTrue(output.last_win_odds.isna().all())

    def test_unknowns_stay_nan_and_actual_racecard_age_rating_are_used(self):
        query = self.query.copy()
        query["horse_id"] = ["new-A", "new-B"]
        query = query.drop(columns=["horse_rating", "horse_age"])
        query["rating"] = [71, np.nan]
        query["age"] = [5, np.nan]
        output = self.build(query=query)
        self.assertEqual(output.iloc[0].horse_rating, 71)
        self.assertEqual(output.iloc[0].horse_age, 5)
        self.assertTrue(pd.isna(output.iloc[1].horse_rating))
        self.assertTrue(pd.isna(output.iloc[1].horse_age))
        for name in ("last_result", "last_rating", "avg_result_3", "trackwork_7d", "season_stakes", "trials_90d"):
            self.assertTrue(output[name].isna().all(), name)
        self.assertTrue(output[[c for c in SPEED_EXTRA_FEATURES if c != "past_race_speed_support"]].isna().all().all())
        self.assertEqual(output.past_race_speed_support.tolist(), [0.0, 0.0])

    def test_six_speed_extras_use_actual_prior_individual_seconds(self):
        prior = self.history[self.history.race_id.eq("R1")].copy()
        prior["date"] = "2019-12-01"
        prior["race_id"] = "older"
        prior["finish_time"] = 100.0
        output = self.build(season_results=prior)
        self.assertEqual(len(SPEED_EXTRA_FEATURES), 6)
        self.assertEqual(set(SPEED_EXTRA_FEATURES), {"past_race_speed_" + name for name in (
            "mean_3_mps", "mean_5_mps", "recency_weighted_mps", "std_5_mps", "support", "trend_mps",
        )})
        a = output.iloc[0]
        self.assertEqual(a.past_race_speed_last_mps, 1400 / 80)
        self.assertEqual(a.past_race_speed_support, 2)
        self.assertEqual(a.past_race_speed_same_distance_mean_mps, 15.75)
        self.assertEqual(a.past_race_speed_mean_3_mps, (14 + 17.5) / 2)
        self.assertEqual(a.past_race_speed_trend_mps, 3.5)
        self.assertAlmostEqual(a.past_race_speed_std_5_mps, np.std([14, 17.5], ddof=1))
        weights = np.exp(-np.array([40, 9]) / 90)
        self.assertAlmostEqual(a.past_race_speed_recency_weighted_mps, np.average([14, 17.5], weights=weights))

    def test_schema_contracts_and_raw_canonical_aliases(self):
        query = self.query.drop(columns="race_id").rename(columns={"date": "race_date", "horse_id": "horse_page_id", "horse_rating": "rating"})
        for schema in SUPPORTED_SCHEMAS:
            output = self.build(query=query, schema=schema)
            self.assertTrue(set(FEATURE_SCHEMAS[schema].features).issubset(output.columns))
            self.assertEqual(output.attrs["feature_schema"], schema)
            self.assertEqual(output.iloc[0].race_id, "HKJC:2020-01-10:ST:R2")
            if schema == "baseline-v1":
                self.assertEqual(output.surface.tolist(), [0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            self.build(schema="unsupported")

    def test_parity_with_existing_rich_logic_for_prior_features(self):
        source = self.history.copy()
        source["horse_age"] = [4, 6, 4, 6]
        expected = prepare_rich_runner_dataset(source, strict_before_meeting=True)
        expected = expected[expected.race_id.eq("R2")].reset_index(drop=True)
        actual = self.build().reset_index(drop=True)
        # Unknown optional-feed counts intentionally differ from old zero fills.
        comparable = [name for name in NOTEBOOK_RICH_SCHEMA.features if not any(
            token in name for token in ("trackwork", "trial", "veterinary", "injury", "fracture", "surgery", "movement")
        ) and name not in ("days_since_hk_arrival",)]
        pd.testing.assert_frame_equal(actual[comparable], expected[comparable], check_dtype=False)

    def test_unknown_entity_does_not_share_other_unknown_entity_history(self):
        query = self.query.drop(columns=["jockey_id", "trainer_id"])
        history = self.history.drop(columns=["jockey_id", "trainer_id"])
        output = self.build(history=history, query=query)
        self.assertTrue(output.jockey_win_rate.isna().all())
        self.assertTrue(output.trainer_starts.isna().all())
        self.assertTrue(output.jockey_last_result.isna().all())

    def test_long_gap_time_windows_preserve_frozen_training_semantics(self):
        source = self.history.copy()
        source.loc[source.race_id.eq("R2"), "date"] = "2021-01-10"
        query = source[source.race_id.eq("R2")]
        expected = prepare_rich_runner_dataset(source, strict_before_meeting=True)
        expected = expected[expected.race_id.eq("R2")].reset_index(drop=True)
        actual = self.build(query=query).reset_index(drop=True)
        columns = ["avg_result_84d", "median_result_183d", "best_result_183d", "worst_result_183d",
                   "avg_speed_ratio_183d", "avg_same_distance_finish_time_183d", "jockey_avg_result_90d", "trainer_avg_result_90d"]
        pd.testing.assert_frame_equal(actual[columns], expected[columns], check_dtype=False)

    def test_every_rich_feature_parity_on_completed_multimeeting_fixture(self):
        blocks = []
        for number, date in enumerate(("2018-06-01", "2019-06-30", "2019-07-10", "2019-12-01", "2020-01-09", "2020-01-10")):
            block = self.history[self.history.race_id.eq("R1")].copy()
            block["date"] = date
            block["race_id"] = f"fixture-{number}"
            block["horse_age"] = [4, 6]
            block["horse_rating"] = [55 + number, 62 + number]
            block["place_odds"] = [1.4, 1.7]
            block["gear"] = ["B-/CP1/TT", "SR/TT"]
            blocks.append(block)
        debut = blocks[-1].iloc[:1].copy()
        debut["horse_id"] = "debut-C"
        debut["horse_no"] = 3
        debut["result"] = 3
        blocks[-1] = pd.concat([blocks[-1], debut], ignore_index=True)
        source = pd.concat(blocks, ignore_index=True)
        query = blocks[-1]
        expected = prepare_rich_runner_dataset(source, strict_before_meeting=True)
        expected = expected[expected.race_id.eq(query.race_id.iloc[0])].reset_index(drop=True)
        actual = self.build(history=source, query=query).reset_index(drop=True)
        # DatasetRegistry masks unaudited legacy event feeds before model input.
        # The legacy rich builder writes zero for these unknown counts; query
        # construction keeps the registry's missing-data semantics instead.
        masked_counts = {"trackwork_7d", "trackwork_14d", "trackwork_30d", "trials_90d",
                         "veterinary_events_30d", "veterinary_events_90d", "injury_events_365d",
                         "fracture_events_365d", "surgery_events_365d", "movements_365d"}
        self.assertTrue(expected[list(masked_counts)].eq(0).all().all())
        self.assertTrue(actual[list(masked_counts)].isna().all().all())
        expected[list(masked_counts)] = np.nan
        columns = list(NOTEBOOK_RICH_SCHEMA.features)
        pd.testing.assert_frame_equal(actual[columns], expected[columns], check_dtype=False)
        speeds = build_speed_features(prepare_rich_runner_dataset(source, strict_before_meeting=True))
        speeds = speeds.iloc[-len(query):].reset_index(drop=True)
        pd.testing.assert_frame_equal(actual[list(SPEED_EXTRA_FEATURES)], speeds[list(SPEED_EXTRA_FEATURES)])

    def test_debut_counts_match_zero_and_unknown_historical_results_stay_nan(self):
        debut = self.build(history=pd.DataFrame())
        self.assertEqual(debut.prior_second_count.tolist(), [0.0, 0.0])
        self.assertEqual(debut.prior_third_count.tolist(), [0.0, 0.0])
        self.assertTrue(debut.prior_second_rate.isna().all())
        self.assertTrue(debut.prior_third_rate.isna().all())
        unknown = self.history[self.history.race_id.eq("R1")].copy()
        unknown["result"] = np.nan
        self.assertTrue(self.build(history=unknown).prior_second_count.isna().all())
        unknown["finishing_status"] = "PU"
        self.assertEqual(self.build(history=unknown).prior_second_count.tolist(), [0.0, 0.0])

    def test_raw_gear_is_not_promoted_into_frozen_categorical_feature(self):
        query = self.query.assign(gear=["B", "TT"])
        self.assertEqual(self.build(query=query).horse_gear.tolist(), ["UNKNOWN", "UNKNOWN"])
        explicit = query.assign(horse_gear=["canonical-B", "canonical-TT"])
        self.assertEqual(self.build(query=explicit).horse_gear.tolist(), explicit.horse_gear.tolist())

    def test_all_supported_outcome_aliases_are_blanked_and_not_predictors(self):
        query = self.query.copy()
        for name in _QUERY_OUTCOMES:
            query[name] = 999
        query["result"] = [1, 2]
        actual = self.build(query=query)
        self.assertTrue(actual[list(_QUERY_OUTCOMES)].isna().all().all())
        scrubbed = query.copy()
        scrubbed[list(_QUERY_OUTCOMES)] = np.nan
        pd.testing.assert_frame_equal(actual, self.build(query=scrubbed))
        self.assertEqual(actual.last_result.tolist(), [1.0, 2.0])
        self.assertEqual(actual.last_win_odds.tolist(), [3.5, 4.5])

    def test_card_labels_match_canonical_training_features_and_preserve_metadata(self):
        history = self.history.drop(columns="jockey_id").copy()
        history["jockey_name"] = ["E C W Wong", "H T Mo", "E C W Wong", "H T Mo"]
        canonical = history[history.race_id.eq("R2")].copy()
        canonical["horse_age"] = [4, 6]
        card = canonical.copy()
        card["going"] = "Good"
        card["jockey_name"] = ["E C W Wong (-3)", "H T Mo (-2)"]
        card["apprentice_allowance_lbs"] = [3.0, np.nan]
        actual = self.build(history=history, query=card)
        expected = self.build(history=history, query=canonical)
        columns = list(NOTEBOOK_RICH_SCHEMA.features) + list(SPEED_EXTRA_FEATURES)
        pd.testing.assert_frame_equal(actual[columns], expected[columns])
        self.assertEqual(actual.going_raw.tolist(), ["Good", "Good"])
        self.assertEqual(actual.going.tolist(), ["GOOD", "GOOD"])
        self.assertEqual(actual.jockey_name_raw.tolist(), card.jockey_name.tolist())
        self.assertEqual(actual.jockey_key.tolist(), canonical.jockey_name.tolist())
        self.assertEqual(actual.jockey_starts.tolist(), [1.0, 1.0])
        self.assertEqual(actual.going_starts.tolist(), [1.0, 1.0])
        self.assertEqual(actual.apprentice_allowance_lbs.tolist(), [3.0, 2.0])
        self.assertEqual(actual.course.tolist(), card.course.tolist())
        self.assertEqual(actual.actual_weight.tolist(), card.actual_weight.tolist())
        pd.testing.assert_frame_equal(card, canonical.assign(
            going="Good", jockey_name=["E C W Wong (-3)", "H T Mo (-2)"],
            apprentice_allowance_lbs=[3.0, np.nan],
        ))

    def test_allowance_normalization_does_not_change_ids_or_interior_annotations(self):
        query = self.query.copy()
        query["jockey_name"] = ["E C W Wong (-3)", "Rider (-2) Junior"]
        output = self.build(query=query)
        self.assertEqual(output.jockey_key.tolist(), query.jockey_id.tolist())
        self.assertEqual(output.jockey_name.tolist(), ["E C W Wong", "Rider (-2) Junior"])
        self.assertEqual(output.iloc[0].apprentice_allowance_lbs, 3)
        self.assertTrue(pd.isna(output.iloc[1].apprentice_allowance_lbs))
        second = self.build(query=output)
        self.assertEqual(second.jockey_name_raw.tolist(), output.jockey_name_raw.tolist())
        self.assertEqual(second.going_raw.tolist(), output.going_raw.tolist())

    def test_unrecognized_going_and_conflicting_allowance_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "Unrecognized going"):
            self.build(query=self.query.assign(going="Godd"))
        with self.assertRaisesRegex(ValueError, "conflicts with apprentice allowance"):
            self.build(query=self.query.assign(jockey_name="E C W Wong (-3)", apprentice_allowance_lbs=2))
        unknown = self.build(query=self.query.assign(going=np.nan))
        self.assertEqual(unknown.going.tolist(), ["UNKNOWN", "UNKNOWN"])
        self.assertTrue(unknown.going_starts.isna().all())

    def test_precanonicalized_collector_labels_retain_original_evidence(self):
        query = self.query.assign(
            going="GOOD", going_raw="Good", jockey_name="E C W Wong",
            jockey_raw="E C W Wong (-3)",
        )
        output = self.build(query=query)
        self.assertEqual(output.going_raw.tolist(), ["Good", "Good"])
        self.assertEqual(output.jockey_name_raw.tolist(), ["E C W Wong (-3)"] * 2)
        self.assertEqual(output.apprentice_allowance_lbs.tolist(), [3.0, 3.0])
        output = self.build(query=query.assign(jockey_raw="Another Rider (-7)"))
        self.assertTrue(output.apprentice_allowance_lbs.isna().all())

    def test_speed_cutoff_parity_at_next_day_boundary(self):
        history = self.history[self.history.race_id.eq("R1")].copy()
        history["date"] = "2020-01-09"
        query = self.query.copy()
        source = pd.concat([history, query], ignore_index=True)
        expected = build_speed_features(prepare_rich_runner_dataset(source, strict_before_meeting=True)).iloc[-2:].reset_index(drop=True)
        actual = self.build(history=history, query=query).reset_index(drop=True)
        pd.testing.assert_frame_equal(actual[list(SPEED_EXTRA_FEATURES)], expected[list(SPEED_EXTRA_FEATURES)])
        self.assertEqual(actual.past_race_speed_support.tolist(), [0.0, 0.0])
        self.assertEqual(actual.prior_starts.tolist(), [1.0, 1.0])

    def test_delayed_publication_and_future_auxiliary_values_withheld(self):
        history = self.history.copy()
        history["outcome_available_at"] = "2020-02-01"
        self.assertEqual(self.build(history=history).prior_starts.tolist(), [0.0, 0.0])
        profiles = pd.DataFrame({"horse_id": ["A", "A", "A"], "snapshot_at": ["2020-01-09", "2020-01-10", "2021-01-01"], "season_stake": [123, 999, 999]})
        original = self.build(profiles=profiles)
        trimmed = self.build(profiles=profiles.iloc[:1])
        pd.testing.assert_frame_equal(original, trimmed)
        self.assertEqual(original.iloc[0].season_stakes, 123)

    def test_duplicates_and_missing_identity_fail_without_fabricated_numbers(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.build(query=pd.concat([self.query, self.query.iloc[:1]]))
        with self.assertRaisesRegex(ValueError, "horse_no"):
            self.build(query=self.query.drop(columns="horse_no"))
        with self.assertRaisesRegex(ValueError, "Incomplete query field"):
            self.build(query=self.query.assign(field_size=14))
        pd.testing.assert_frame_equal(self.build(), self.build(season_results=self.history))

    def test_empty_history_and_empty_query_keep_contract(self):
        output = self.build(history=pd.DataFrame())
        self.assertEqual(output.prior_starts.tolist(), [0.0, 0.0])
        self.assertTrue(output.last_result.isna().all())
        empty = self.build(query=self.query.iloc[:0])
        self.assertTrue(empty.empty)
        self.assertTrue(set(NOTEBOOK_RICH_SCHEMA.features).issubset(empty.columns))

    def test_overlapping_exports_coalesce_actual_form_and_reject_conflicts(self):
        new = self.history.copy()
        new["source_body_hash"] = "new-capture"
        new["horse_age"] = 5
        output = self.build(season_results=new)
        self.assertEqual(output.prior_starts.tolist(), [1.0, 1.0])
        bad = new.copy()
        bad.loc[0, "result"] = 9
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            self.build(season_results=bad)

    def test_nested_collector_metadata_is_not_hashed_for_deduplication(self):
        history = self.history.copy()
        history["source_cells"] = [["official", {"table": 1}]] * len(history)
        new = history.copy()
        new["source_cells"] = [["refreshed", {"table": 2}]] * len(history)
        output = self.build(history=history, season_results=new)
        self.assertEqual(output.prior_starts.tolist(), [1.0, 1.0])

    def test_auxiliary_capture_and_observed_zero_counts(self):
        events = pd.DataFrame({"horse_id": ["A", "A"], "event_date": ["2020-01-01", "2020-01-09"], "first_seen_at": ["2020-01-02", "2020-02-01"]})
        output = self.build(trackwork=events)
        self.assertEqual(output.iloc[0].trackwork_7d, 0)
        self.assertEqual(output.iloc[0].days_since_trackwork, 9)
        self.assertTrue(pd.isna(output.iloc[1].trackwork_7d))

    def test_official_card_source_and_stale_query_predictors_are_rebuilt(self):
        query = self.query.copy()
        query["source"] = "official:hkjc-racecard"
        query["prior_starts"] = 999
        query["past_race_speed_support"] = 999
        output = self.build(query=query)
        self.assertEqual(output.official_source.tolist(), [1.0, 1.0])
        self.assertEqual(output.prior_starts.tolist(), [1.0, 1.0])
        self.assertEqual(output.past_race_speed_support.tolist(), [1.0, 1.0])

    def test_multiple_query_dates_and_field_ranks_are_independent(self):
        later = self.query.copy()
        later["date"] = "2020-02-01"
        later["race_id"] = "later-query"
        together = self.build(query=pd.concat([self.query, later], ignore_index=True))
        early = together[together.race_id.eq("R2")].reset_index(drop=True)
        pd.testing.assert_frame_equal(early, self.build().reset_index(drop=True))
        self.assertEqual(together[together.race_id.eq("later-query")].prior_starts.tolist(), [2.0, 2.0])
        self.assertEqual(early.age_rank.tolist(), [1.0, 0.5])


if __name__ == "__main__":
    unittest.main()
