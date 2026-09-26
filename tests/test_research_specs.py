import unittest

from pydantic import ValidationError

from ima.experiments import recipe_experiment_spec
from ima.feature_sets import BASELINE_SCHEMA, RICH_SCHEMA, drop_feature_families
from ima.research_specs import PipelineRecipe, RecipeValidationError, ResearchProposal


class ResearchSpecTests(unittest.TestCase):
    def test_proposal_search_space_rejects_invalid_bounds_and_parameters(self):
        base = dict(
            proposal_id="p", hypothesis="Tune C.", changed_axes=("hyperparameters",),
            recipe=PipelineRecipe(), expected_observation="Better loss.",
            falsification_rule="No improvement.", max_trials=4,
        )
        with self.assertRaises(ValidationError):
            ResearchProposal(**base, search_space={"C": {"kind": "float", "low": 1, "high": 0}})
        with self.assertRaises(ValidationError):
            ResearchProposal(**base, search_space={"n_estimators": {"kind": "int", "low": 10, "high": 20}})
        with self.assertRaises(ValidationError):
            ResearchProposal(**base, search_space={"C": {"kind": "float", "low": 0.1, "high": 101}})
        proposal = ResearchProposal(
            **base, search_space={"C": {"kind": "float", "low": 0.1, "high": 1.0}}
        )
        self.assertEqual(4, proposal.max_trials)
        large = ResearchProposal(**(base | {"max_trials": 128}))
        self.assertEqual(128, large.max_trials)
        with self.assertRaisesRegex(ValidationError, "max_trials must be positive"):
            ResearchProposal(**(base | {"max_trials": 0}))

    def test_recipe_hash_is_stable_and_families_are_sorted(self):
        first = PipelineRecipe(
            feature_schema="benter-rich-v1",
            drop_feature_families=("preferences", "source_quality", "preferences"),
            model={"kind": "logit", "parameters": {"C": 0.5}},
        )
        second = PipelineRecipe(
            feature_schema="benter-rich-v1",
            drop_feature_families=("source_quality", "preferences"),
            model={"kind": "logit", "parameters": {"C": 0.5}},
        )
        self.assertEqual(("preferences", "source_quality"), first.drop_feature_families)
        self.assertEqual(first.recipe_hash(), second.recipe_hash())

    def test_unknown_fields_and_feature_families_are_rejected(self):
        with self.assertRaises(ValidationError):
            PipelineRecipe(nope=True)
        with self.assertRaises(ValidationError):
            PipelineRecipe(drop_feature_families=("not_a_family",))

    def test_model_target_compatibility_is_enforced(self):
        with self.assertRaises(ValidationError):
            PipelineRecipe(
                target={"kind": "adjusted_finish_time_or_speed"},
                model={"kind": "logit"},
                blend={"kind": "none"},
                calibration={"kind": "none"},
            )
        valid = PipelineRecipe(
            target={"kind": "adjusted_finish_time_or_speed"},
            model={"kind": "ridge_regressor"},
            blend={"kind": "none"},
            calibration={"kind": "none"},
        )
        self.assertEqual("ridge_regressor", valid.model.kind)

    def test_model_parameters_reject_unsupported_or_out_of_range_values(self):
        with self.assertRaisesRegex(ValidationError, "n_estimators"):
            PipelineRecipe(model={"kind": "boosted", "parameters": {"n_estimators": 100}})
        with self.assertRaisesRegex(ValidationError, "learning_rate"):
            PipelineRecipe(model={"kind": "boosted", "parameters": {"learning_rate": 0}})
        with self.assertRaisesRegex(ValidationError, "class_weight"):
            PipelineRecipe(model={"kind": "logit", "parameters": {"class_weight": "auto"}})
        valid = PipelineRecipe(model={
            "kind": "boosted",
            "parameters": {"max_iter": 200, "max_leaf_nodes": 31},
        })
        self.assertEqual(200, valid.model.parameters["max_iter"])

    def test_non_win_targets_cannot_use_win_probability_blend(self):
        with self.assertRaises(ValidationError):
            PipelineRecipe(
                target={"kind": "ranking_strength"},
                model={"kind": "pairwise_ranker"},
            )

    def test_drop_feature_families_reduces_schema_without_mutating_original(self):
        dropped = drop_feature_families(RICH_SCHEMA, ("preferences",))
        self.assertLess(len(dropped.features), len(RICH_SCHEMA.features))
        self.assertEqual(23, len(BASELINE_SCHEMA.features))
        self.assertNotIn("distance_band_win_rate", dropped.features)

    def test_executable_recipe_converts_to_legacy_experiment_spec(self):
        recipe = PipelineRecipe(model={"kind": "boosted", "parameters": {"learning_rate": 0.06}})
        spec = recipe_experiment_spec(recipe)
        self.assertTrue(spec.run_id.startswith("recipe-"))
        self.assertEqual("boosted", spec.kind)
        self.assertEqual({"learning_rate": 0.06}, spec.parameters)

    def test_transform_specs_are_validated(self):
        recipe = PipelineRecipe(
            transforms=({
                "kind": "clip_numeric_quantiles",
                "parameters": {"columns": ["horse_rating"], "lower": 0.05, "upper": 0.95},
            },)
        )
        self.assertEqual("clip_numeric_quantiles", recipe.transforms[0].kind)
        with self.assertRaises(ValidationError):
            PipelineRecipe(
                transforms=({
                    "kind": "clip_numeric_quantiles",
                    "parameters": {"columns": ["horse_rating"], "lower": 0.95, "upper": 0.05},
                },)
            )


if __name__ == "__main__":
    unittest.main()
