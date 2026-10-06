import unittest

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ima.feature_discovery_specs import DiscoverySpec, DiscoverySpecV2, content_id, parse_discovery_spec
from ima.feature_screening import DiscoverySelection, SelectionDeferred


def training(n_features=80):
    rng = np.random.default_rng(17)
    frame = pd.DataFrame(rng.normal(size=(160,n_features)),columns=[f"dfs_{i:03d}" for i in range(n_features)])
    frame["date"] = np.repeat(pd.date_range("2020-01-01",periods=40),4)
    frame["race_id"] = np.repeat(np.arange(40),4)
    frame["label"] = rng.normal(size=len(frame))
    return frame


class SelectionTests(unittest.TestCase):
    def test_legacy_identity_and_parser(self):
        legacy = DiscoverySpec()
        self.assertEqual(legacy.discovery_id(),content_id(legacy.model_dump(mode="json")))
        self.assertNotIn("selection_shortlist",legacy.model_dump())
        self.assertEqual(1,parse_discovery_spec({}).schema_version)
        with self.assertRaises(ValueError):
            DiscoverySpec(max_selected=64)
        self.assertEqual(64,parse_discovery_spec({"schema_version":2,"max_selected":64}).max_selected)

    def test_explicit_budgets_and_all_no_hidden_cap(self):
        frame = training(140)
        with threadpool_limits(limits=1):
            for budget in (8,32,64,128,None):
                selection = DiscoverySelection.fit(frame,DiscoverySpecV2(max_selected=budget,selection="quality"),"speed_regression","label")
                self.assertEqual(140 if budget is None else budget,len(selection.columns))
                self.assertEqual(len(selection.columns),selection.report["selected_count"])

    def test_sequential_integer_not_half_and_all_bypass(self):
        frame = training(5)
        with threadpool_limits(limits=1):
            selection = DiscoverySelection.fit(frame,DiscoverySpecV2(max_selected=4,selection="sequential"),"speed_regression","label")
        self.assertEqual(4,len(selection.columns))
        with threadpool_limits(limits=1):
            selection = DiscoverySelection.fit(training(70),DiscoverySpecV2(max_selected=64,selection="sequential",selection_shortlist=64),"speed_regression","label")
        self.assertEqual(64,len(selection.columns))

    def test_embedded_more_than_32_and_deferral(self):
        frame = training(70)
        with threadpool_limits(limits=1):
            selection = DiscoverySelection.fit(frame,DiscoverySpecV2(max_selected=64,selection="embedded"),"speed_regression","label")
        self.assertEqual(64,len(selection.columns))
        with self.assertRaises(SelectionDeferred) as caught:
            DiscoverySelection.fit(frame,DiscoverySpecV2(max_selected=64,selection="sequential",selection_fit_budget=1),"speed_regression","label")
        self.assertGreater(caught.exception.report["estimated_fits"],1)
        with self.assertRaises(ValueError):
            DiscoverySpecV2(max_selected=64,selection_shortlist=12)
