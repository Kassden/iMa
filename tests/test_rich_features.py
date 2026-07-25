import unittest

from ima.data import FEATURES
from ima.feature_sets import (
    BASELINE_SCHEMA, BENTER_COVERAGE, FEATURE_FAMILIES, RICH_SCHEMA,
    validate_feature_contract,
)


class RichFeatureContractTests(unittest.TestCase):
    def test_baseline_schema_remains_compatible(self):
        self.assertEqual(tuple(FEATURES), BASELINE_SCHEMA.features)

    def test_rich_contract_is_unique_and_family_complete(self):
        validate_feature_contract()
        assigned = {feature for values in FEATURE_FAMILIES.values() for feature in values}
        self.assertEqual(set(RICH_SCHEMA.features), assigned)
        self.assertGreater(len(RICH_SCHEMA.features), 60)

    def test_benter_coverage_discloses_unsupported_factors(self):
        coverage = {row["factor"]: row["status"] for row in BENTER_COVERAGE}
        self.assertEqual("unsupported", coverage["horse age"])
        self.assertEqual("unsupported", coverage["bad luck adjustment"])
        self.assertEqual("supported", coverage["lengths behind winner"])


if __name__ == "__main__":
    unittest.main()