import unittest

import numpy as np

from ima.feature_candidates import candidate_definitions
from ima.feature_definitions import validate_definition
from ima.feature_expressions import evaluate_expression
from ima.feature_lifecycle import _numeric_metadata, deterministic_candidate_definitions


def column(unit="m", **updates):
    return dict(unit=unit, dtype="float64", temporal_scope="pre_race", target_tainted=False,
                eligible=True, source_family="distance", nonmissing=10, distinct=4, **updates)


class CandidateDefinitionsTests(unittest.TestCase):
    def setUp(self):
        self.catalog = {f"measurement_{i}": column() for i in range(6)}

    def test_order_and_domain_priority(self):
        catalog = dict(self.catalog, distance=column(), last_distance=column())
        first = candidate_definitions(catalog, limit=100)
        reverse = {key: dict(reversed(list(value.items()))) for key, value in reversed(list(catalog.items()))}
        self.assertEqual(first, candidate_definitions(reverse, limit=100))
        domain = deterministic_candidate_definitions(catalog, limit=8)
        self.assertEqual(first[:len(domain)], domain)
        self.assertEqual(first[0].name, "distance_relative_to_previous")

    def test_exclusions_page_past_eight_without_lifetime_cap(self):
        all_definitions = candidate_definitions(self.catalog, limit=100)
        self.assertGreater(len(all_definitions), 8)
        ids = [definition.content_id() for definition in all_definitions]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(candidate_definitions(self.catalog, exclude_ids=ids[:12], limit=5), all_definitions[12:17])
        self.assertEqual(candidate_definitions(self.catalog, exclude_ids=ids, limit=100), [])
        for definition in all_definitions:
            validate_definition(definition, _numeric_metadata(self.catalog))
            self.assertTrue(definition.hypothesis)
            self.assertTrue(definition.source_families)

    def test_units_and_unsafe_inputs(self):
        catalog = dict(self.catalog)
        unsafe = {
            "future_measure": {"temporal_scope": "future"},
            "outcome_measure": {"temporal_scope": "current_outcome"},
            "tainted_measure": {"target_tainted": True},
            "fitted_measure": {"fit_required": True},
            "unknown_unit": {"unit": "bananas"},
            "historical_measure": {"temporal_scope": "historical"},
            "text_measure": {"dtype": "string"},
            "constant_measure": {"distinct": 1},
            "empty_measure": {"nonmissing": 0},
            "ineligible_measure": {"eligible": False},
            "target_win": {},
            "__import__('os')": {},
        }
        for name, changes in unsafe.items():
            catalog[name] = column() | changes
        catalog.update(kilometres=column("km"), seconds=column("s"),
                       other_family=column() | {"source_family": "other"})
        candidates = candidate_definitions(catalog, limit=1000)
        self.assertEqual(candidates, candidate_definitions(self.catalog, limit=1000))

    def test_support_transform_and_scale_aliases(self):
        catalog = {"prior_count": column("1"), "earlier_seconds": column("seconds"),
                   "later_seconds": column("s"), "prior_starts": column("kg")}
        candidates = candidate_definitions(catalog, limit=100)
        transformed = next(d for d in candidates if d.name == "log_abs_support_prior_count")
        values = evaluate_expression(transformed.expression_ast, {"prior_count": [-3, 0, 4]}, 3)
        np.testing.assert_allclose(values, np.log1p([3, 0, 4]))
        self.assertTrue(any(d.expression_ast.op == "subtract" for d in candidates))
        self.assertFalse(any(d.input_refs == ("prior_starts",) for d in candidates))
        for definition in candidates:
            validate_definition(definition, _numeric_metadata(catalog))

    def test_existing_content_ids_and_names_are_skipped(self):
        definitions = candidate_definitions(self.catalog, limit=100)
        catalog = dict(self.catalog, already_built={"definition_id": definitions[0].content_id()})
        self.assertEqual(candidate_definitions(catalog, limit=3), definitions[1:4])
        catalog = dict(self.catalog, **{definitions[0].name: {}})
        self.assertNotIn(definitions[0].name, [d.name for d in candidate_definitions(catalog, limit=100)])

    def test_invalid_limits(self):
        for limit in (True, False, 0, -1, 1.5, "3", None):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                candidate_definitions(self.catalog, limit=limit)
        self.assertEqual(len(candidate_definitions(self.catalog, limit=1)), 1)
        with self.assertRaises(ValueError):
            candidate_definitions([])


if __name__ == "__main__":
    unittest.main()
