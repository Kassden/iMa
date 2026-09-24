import unittest

import pandas as pd

from ima.research_transforms import TransformError, TransformSpec, apply_research_transforms


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


if __name__ == "__main__":
    unittest.main()
