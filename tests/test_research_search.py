import tempfile
import unittest
from pathlib import Path

from ima.optimizer import research_recipe_proposals
from ima.research_search import RecipeSearchController


class ResearchSearchTests(unittest.TestCase):
    def test_seeded_suggestions_cover_required_research_axes(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = RecipeSearchController(Path(directory))
            suggestions = controller.ask(6)
        recipes = [suggestion.recipe for suggestion in suggestions]
        self.assertGreaterEqual(len({recipe.feature_schema for recipe in recipes}), 2)
        self.assertTrue(any(recipe.drop_feature_families for recipe in recipes))
        self.assertTrue(any(recipe.train_window == "trailing_3_years" for recipe in recipes))
        self.assertTrue(any(recipe.transforms for recipe in recipes))
        self.assertGreaterEqual(len({recipe.model.kind for recipe in recipes}), 2)
        self.assertEqual(len(recipes), len({recipe.recipe_hash() for recipe in recipes}))

    def test_optuna_suggestions_resume_without_duplicate_running_recipes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = RecipeSearchController(root).ask(10)
            second = RecipeSearchController(root).ask(10)
            first_hashes = {suggestion.recipe.recipe_hash() for suggestion in first}
            second_hashes = {suggestion.recipe.recipe_hash() for suggestion in second}
        self.assertTrue(first_hashes.isdisjoint(second_hashes))

    def test_tell_persists_completed_value_and_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            controller = RecipeSearchController(root)
            suggestion = controller.ask(1)[0]
            controller.tell(
                suggestion.trial_number,
                2.01,
                metrics={"race_weighted_log_loss": 2.01},
            )
            snapshot = RecipeSearchController(root).snapshot()
        self.assertEqual(1, snapshot["completed"])
        self.assertEqual(0, snapshot["running"])
        self.assertIn(suggestion.recipe.recipe_hash(), snapshot["recipe_hashes"])

    def test_optimizer_exposes_recipe_proposals_without_legacy_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            proposals = research_recipe_proposals(Path(directory), count=2)
        self.assertEqual(2, len(proposals))
        self.assertIn("recipe", proposals[0])
        self.assertIn("recipe_hash", proposals[0])

    def test_fallback_search_persists_seed_recipes_without_optuna_study(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = RecipeSearchController(Path(directory))
            controller.study = None
            suggestions = controller.ask(2)
            reloaded = RecipeSearchController(Path(directory))
            reloaded.study = None
            snapshot = reloaded.snapshot()
        self.assertEqual(2, len(suggestions))
        self.assertGreaterEqual(snapshot["trials"], 2)


if __name__ == "__main__":
    unittest.main()
