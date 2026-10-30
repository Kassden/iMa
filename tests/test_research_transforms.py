import pickle
import unittest

import pandas as pd

from ima.research_transforms import (
    FittedResearchTransforms,
    TransformError,
    TransformSpec,
    apply_research_transforms,
)


class ResearchTransformTests(unittest.TestCase):
    def test_clip_numeric_quantiles_are_fit_on_train_only(self):
        train = pd.DataFrame({
            "race_id": ["R1", "R1", "R2"],
            "horse_rating": [10.0, 20.0, 30.0],
        })
        score = pd.DataFrame({
            "race_id": ["R3", "R3"],
            "horse_rating": [5.0, 1000.0],
        })
        _, clipped = apply_research_transforms(
            train,
            score,
            specs=(TransformSpec(
                "clip_numeric_quantiles",
                {"columns": ["horse_rating"], "lower": 0.0, "upper": 1.0},
            ),),
        )
        self.assertEqual([10.0, 30.0], clipped["horse_rating"].tolist())

    def test_race_relative_rank_adds_train_and_score_columns(self):
        train = pd.DataFrame({
            "race_id": ["R1", "R1"],
            "horse_rating": [60.0, 70.0],
        })
        score = pd.DataFrame({
            "race_id": ["R2", "R2"],
            "horse_rating": [80.0, 75.0],
        })
        transformed_train, transformed_score = apply_research_transforms(
            train,
            score,
            specs=(TransformSpec("race_relative_rank", {"columns": ["horse_rating"]}),),
        )
        self.assertEqual([0.5, 1.0], transformed_train["horse_rating_race_rank"].tolist())
        self.assertEqual([1.0, 0.5], transformed_score["horse_rating_race_rank"].tolist())

    def test_unknown_transform_is_rejected(self):
        with self.assertRaisesRegex(TransformError, "Unknown transform"):
            apply_research_transforms(
                pd.DataFrame({"race_id": ["R1"]}),
                specs=(TransformSpec("magic", {}),),
            )

    def test_feature_creation_uses_only_declared_pregame_columns(self):
        train = pd.DataFrame({
            "race_id": ["R1", "R1"], "horse_rating": [10.0, 20.0],
            "last_speed_ratio": [0.9, 1.0],
        })
        score = pd.DataFrame({
            "race_id": ["R2", "R2"], "horse_rating": [30.0, 50.0],
            "last_speed_ratio": [0.8, 0.9],
        })
        _, transformed = apply_research_transforms(
            train, score, specs=(
                TransformSpec("race_relative_center", {"columns": ["horse_rating"]}),
                TransformSpec("signed_log1p", {"columns": ["horse_rating"]}),
                TransformSpec("numeric_interaction", {
                    "columns": ["horse_rating", "last_speed_ratio"],
                }),
            ),
        )
        self.assertEqual([-10.0, 10.0], transformed["horse_rating_race_centered"].tolist())
        self.assertEqual([24.0, 45.0], transformed["horse_rating_x_last_speed_ratio"].tolist())
        self.assertTrue(transformed["horse_rating_signed_log1p"].iloc[1] > 0)

    def test_interaction_rejects_duplicate_inputs(self):
        with self.assertRaisesRegex(TransformError, "two distinct columns"):
            apply_research_transforms(
                pd.DataFrame({"race_id": ["R1"], "horse_rating": [10.0]}),
                specs=(TransformSpec("numeric_interaction", {
                    "columns": ["horse_rating", "horse_rating"],
                }),),
            )


    def test_stateless_all_missing_training_input_is_masked_on_future_frame(self):
        early_train = pd.DataFrame({
            "race_id": ["R1", "R1"], "past_race_speed_trend_mps": [float("nan")] * 2,
        })
        future = pd.DataFrame({
            "race_id": ["R2", "R2"], "past_race_speed_trend_mps": [1.8, 2.4],
        })
        original_future = future.copy(deep=True)

        transformed_train, transformed_future = apply_research_transforms(
            early_train,
            future,
            specs=(TransformSpec("signed_log1p", {"columns": ["past_race_speed_trend_mps"]}),),
        )

        self.assertTrue(transformed_train["past_race_speed_trend_mps"].isna().all())
        self.assertTrue(transformed_future["past_race_speed_trend_mps"].isna().all())
        self.assertTrue(transformed_future["past_race_speed_trend_mps_signed_log1p"].isna().all())
        pd.testing.assert_frame_equal(future, original_future)

    def test_mixed_specs_mask_inactive_inputs_and_keep_active_transforms(self):
        train = pd.DataFrame({
            "race_id": ["R1", "R1"], "missing_feature": [None, None],
            "rating": [10.0, 20.0], "last_speed_ratio": [0.9, 1.0],
        })
        future = pd.DataFrame({
            "race_id": ["R2", "R2"], "missing_feature": [5.0, 7.0],
            "rating": [30.0, 50.0], "last_speed_ratio": [0.8, 0.9],
        })

        _, transformed = apply_research_transforms(
            train,
            future,
            specs=(
                TransformSpec("race_relative_rank", {"columns": ["missing_feature", "rating"]}),
                TransformSpec("race_relative_center", {"columns": ["missing_feature", "rating"]}),
                TransformSpec("numeric_interaction", {"columns": ["rating", "last_speed_ratio"]}),
                TransformSpec("signed_log1p", {"columns": ["rating"]}),
            ),
        )

        self.assertTrue(transformed["missing_feature"].isna().all())
        self.assertTrue(transformed["missing_feature_race_rank"].isna().all())
        self.assertTrue(transformed["missing_feature_race_centered"].isna().all())
        self.assertEqual([0.5, 1.0], transformed["rating_race_rank"].tolist())
        self.assertEqual([-10.0, 10.0], transformed["rating_race_centered"].tolist())
        self.assertEqual([24.0, 45.0], transformed["rating_x_last_speed_ratio"].tolist())
        self.assertTrue(transformed["rating_signed_log1p"].iloc[1] > 0)

    def test_clip_refuses_training_data_without_observations(self):
        with self.assertRaisesRegex(TransformError, "no training observations"):
            apply_research_transforms(
                pd.DataFrame({"value": [float("nan"), None, "unavailable"]}),
                specs=(TransformSpec("clip_numeric_quantiles", {"columns": ["value"]}),),
            )

    def test_existing_clip_infinity_behavior_is_preserved(self):
        fitted = FittedResearchTransforms.fit(
            pd.DataFrame({"value": [1.0, 3.0]}),
            (TransformSpec("clip_numeric_quantiles", {"columns": ["value"], "lower": 0, "upper": 1}),),
        )
        result = fitted.transform(pd.DataFrame({"value": [-float("inf"), float("inf")]}))
        self.assertEqual([1.0, 3.0], result["value"].tolist())

    def test_fitted_transform_pickle_roundtrip_and_legacy_state(self):
        specs = (TransformSpec("signed_log1p", {"columns": ["value"]}),)
        fitted = FittedResearchTransforms.fit(pd.DataFrame({"value": [None]}), specs)
        restored = pickle.loads(pickle.dumps(fitted))
        result = restored.transform(pd.DataFrame({"value": [9.0]}))
        self.assertTrue(result["value"].isna().all())

        legacy = FittedResearchTransforms.__new__(FittedResearchTransforms)
        legacy.specs = specs
        legacy.clip_bounds = {}
        legacy_result = legacy.transform(pd.DataFrame({"value": [9.0]}))
        self.assertEqual([9.0], legacy_result["value"].tolist())


if __name__ == "__main__":
    unittest.main()
