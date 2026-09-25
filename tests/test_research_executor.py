import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ima.research_executor import RecipeExecutionRequest, execute_recipe
from ima.research_model_package import load_research_package
from ima.research_specs import PipelineRecipe


FIXTURE = Path("tests/fixtures/research_races.csv")
PROTOCOL = {
    "min_train_races": 2,
    "calibration_races": 1,
    "score_races": 1,
    "max_folds": 2,
}


class ResearchExecutorTests(unittest.TestCase):
    def _run(self, root: Path, recipe: PipelineRecipe, dataset: Path = FIXTURE):
        return execute_recipe(RecipeExecutionRequest(
            attempt_id=f"attempt-{recipe.recipe_hash()}",
            proposal_id="proposal-fixture",
            trial_number=0,
            recipe=recipe,
            dataset_path=dataset,
            output_dir=root / recipe.recipe_hash(),
            protocol_parameters=PROTOCOL,
            code_revision="fixture-revision",
            environment_hash="fixture-environment",
        ))

    def test_executes_win_recipe_and_persists_protected_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self._run(Path(directory), PipelineRecipe())
            self.assertEqual("completed", result.status, result.error)
            self.assertEqual("development_race_log_loss", result.objective_name)
            self.assertTrue(np.isfinite(result.objective_value))
            self.assertEqual(2, len(result.metrics["folds"]))
            output = Path(directory) / result.recipe_hash
            stored = json.loads((output / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(result.lineage["protocol_id"], stored["lineage"]["protocol_id"])
            predictions = pd.read_csv(output / "predictions.csv")
            totals = predictions.groupby("race_id")["selected_probability"].sum()
            self.assertTrue(np.allclose(totals.to_numpy(), 1.0))

    def test_transform_ablation_and_window_change_effective_training(self):
        recipe = PipelineRecipe(
            feature_schema="notebook-rich-v2",
            drop_feature_families=("preferences",),
            transforms=({
                "kind": "race_relative_rank",
                "parameters": {"columns": ["horse_rating"]},
            },),
            train_window="trailing_3_years",
            model={"kind": "logit", "parameters": {"C": 0.1}},
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            enriched = pd.read_csv(FIXTURE)
            enriched["horse_rating"] = 60 + enriched["horse_no"]
            dataset = root / "enriched.csv"
            enriched.to_csv(dataset, index=False)
            result = self._run(root, recipe, dataset)
            self.assertEqual("completed", result.status, result.error)
            features = result.metrics["effective_training"][0]["features"]
            self.assertIn("horse_rating_race_rank", features)
            self.assertIn(
                "horse_rating_race_rank",
                result.metrics["effective_training"][0]["available_features"],
            )
            self.assertNotIn("distance_change", features)
            self.assertGreater(result.metrics["effective_training"][0]["rows"], 0)

    def test_saved_fitted_package_reproduces_score_probabilities(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self._run(Path(directory), PipelineRecipe())
            package = load_research_package(Path(result.artifacts["package"]))
            predictions = pd.read_csv(result.artifacts["predictions"])
            source = pd.read_csv(FIXTURE)
            final_races = predictions[predictions["fold_id"].eq("fold-002")]["race_id"]
            frame = source[source["race_id"].isin(final_races)].copy()
            direct = package.predict_proba(frame)
            expected = predictions[predictions["fold_id"].eq("fold-002")][
                "selected_probability"
            ].to_numpy()
            self.assertTrue(np.allclose(direct, expected))

    def test_hash_mismatch_fails_without_partial_success(self):
        with tempfile.TemporaryDirectory() as directory:
            request = RecipeExecutionRequest(
                attempt_id="attempt-bad-hash",
                proposal_id="proposal-bad-hash",
                trial_number=0,
                recipe=PipelineRecipe(),
                dataset_path=FIXTURE,
                output_dir=Path(directory) / "bad",
                protocol_parameters=PROTOCOL,
                dataset_hash="wrong",
            )
            result = execute_recipe(request)
            self.assertEqual("failed", result.status)
            self.assertIn("dataset hash", result.error)
            self.assertFalse((Path(directory) / "bad" / "package").exists())

    def test_win_target_excludes_dead_heat_races_before_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = pd.read_csv(FIXTURE)
            dead_heat_race = frame["race_id"].iloc[0]
            indices = frame.index[frame["race_id"].eq(dead_heat_race)][:2]
            frame.loc[indices, "target_win"] = 1
            frame.loc[frame["race_id"].eq(dead_heat_race), "target_probability"] = 0.0
            frame.loc[indices, "target_probability"] = 0.5
            dataset = root / "dead-heat.csv"
            frame.to_csv(dataset, index=False)
            result = self._run(root, PipelineRecipe(), dataset)
            self.assertEqual("completed", result.status, result.error)
            self.assertEqual(
                1, result.metrics["dataset_exclusions"]["non_single_winner_races"]
            )
            predictions = pd.read_csv(result.artifacts["predictions"])
            self.assertNotIn(dead_heat_race, set(predictions["race_id"]))

    def test_secondary_targets_train_score_controls_and_reload(self):
        recipes = [
            PipelineRecipe(
                target={"kind": "ranking_strength"},
                model={"kind": "pairwise_ranker", "parameters": {"max_iter": 20}},
                calibration={"kind": "none"},
                blend={"kind": "none"},
            ),
            PipelineRecipe(
                target={"kind": "placing_top_k", "parameters": {"top_k": 2}},
                model={"kind": "logit", "parameters": {"C": 0.5}},
                calibration={"kind": "none"},
                blend={"kind": "none"},
            ),
            PipelineRecipe(
                target={
                    "kind": "adjusted_finish_time_or_speed",
                    "parameters": {"min_coverage": 1.0},
                },
                model={"kind": "ridge_regressor", "parameters": {"alpha": 1.0}},
                calibration={"kind": "none"},
                blend={"kind": "none"},
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            for recipe in recipes:
                with self.subTest(target=recipe.target.kind):
                    result = self._run(Path(directory), recipe)
                    self.assertEqual("completed", result.status, result.error)
                    self.assertIn("shuffled_control", result.metrics["folds"][0])
                    package = load_research_package(Path(result.artifacts["package"]))
                    predictions = pd.read_csv(result.artifacts["predictions"])
                    source = pd.read_csv(FIXTURE)
                    final = predictions[predictions["fold_id"].eq("fold-002")]
                    frame = source[source["race_id"].isin(final["race_id"])].copy()
                    self.assertTrue(np.allclose(package.predict(frame), final["prediction"]))

    def test_odds_forecast_is_blocked_without_timestamped_snapshots(self):
        recipe = PipelineRecipe(
            target={"kind": "market_odds_forecast"},
            model={"kind": "ridge_regressor"},
            calibration={"kind": "none"},
            blend={"kind": "none"},
        )
        with tempfile.TemporaryDirectory() as directory:
            result = self._run(Path(directory), recipe)
            self.assertEqual("failed", result.status)
            self.assertIn("odds_snapshot_at", result.error)


if __name__ == "__main__":
    unittest.main()
