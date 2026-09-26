import unittest

from scripts.enrich_mlflow_research_identity import identity_updates


class EnrichMLflowResearchIdentityTests(unittest.TestCase):
    def test_builds_short_aliases_and_tags_from_nested_recipe_params(self):
        aliases, tags = identity_updates({
            "recipe.feature_schema": "notebook-rich-v2",
            "recipe.model.kind": "boosted",
            "recipe.train_window": "trailing_3_years",
        })

        self.assertEqual("notebook-rich-v2", aliases["feature_schema"])
        self.assertEqual("boosted", aliases["model_kind"])
        self.assertEqual("trailing_3_years", aliases["train_window"])
        self.assertEqual("notebook-rich-v2", tags["ima.feature_schema"])

    def test_does_not_try_to_overwrite_existing_params(self):
        aliases, tags = identity_updates({
            "feature_schema": "baseline-v1",
            "recipe.feature_schema": "baseline-v1",
        })

        self.assertNotIn("feature_schema", aliases)
        self.assertEqual("baseline-v1", tags["ima.feature_schema"])


if __name__ == "__main__":
    unittest.main()
