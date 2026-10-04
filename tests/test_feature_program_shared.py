import pickle
import tempfile
import unittest

import numpy as np
import pandas as pd

from ima.feature_discovery_specs import DiscoverySpecV2
from ima.feature_program import materialize, synthesize
from tests.test_feature_program import fixture


class SharedDiscoveryTests(unittest.TestCase):
    def test_shared_matrix_matches_generation_and_reuses_selection_settings(self):
        frame = fixture()
        spec = DiscoverySpecV2(windows_days=(90,),selection="quality",max_selected=64)
        expected,_ = synthesize(frame,spec)
        with tempfile.TemporaryDirectory() as root:
            artifact,manifest = materialize(frame,spec,"source",root,shared=True)
            self.assertEqual("miss",artifact.cache_status)
            array = artifact.load()
            self.assertIsInstance(array,np.memmap)
            self.assertFalse(array.flags.writeable)
            np.testing.assert_allclose(array,expected.to_numpy(),equal_nan=True)
            changed = spec.model_copy(update={"selection":"mutual_information","max_selected":None,"selection_fit_budget":100})
            again,other = materialize(frame,changed,"source",root,shared=True)
            self.assertEqual(artifact.matrix_id,again.matrix_id)
            self.assertEqual("hit",again.cache_status)
            self.assertNotEqual(manifest["discovery_id"],other["discovery_id"])
            replay = pickle.loads(pickle.dumps(artifact))
            pd.testing.assert_frame_equal(replay.frame(columns=expected.columns[:2],positions=[3,6]),expected.iloc[[3,6],:2].reset_index(drop=True))

    def test_generic_measurement_and_availability(self):
        frame = fixture()
        frame["novel"] = np.arange(len(frame),dtype=float)
        frame["novel_available"] = pd.to_datetime(frame.date)+pd.Timedelta(days=1)
        metadata = {"novel":{"unit":"1","temporal_scope":"historical","available_at_column":"novel_available"}}
        spec = DiscoverySpecV2(measurements=("novel",),windows_days=(90,))
        matrix,manifest = synthesize(frame,spec,input_metadata=metadata)
        mean = next(d["feature_id"] for d in manifest["catalog"] if d["expression"] == "MEAN(starts.novel)")
        self.assertEqual(0.,matrix.loc[6,mean])
        with self.assertRaises(ValueError):
            synthesize(frame,spec)
        metadata["novel"]["target_tainted"] = True
        with self.assertRaises(ValueError):
            synthesize(frame,spec,input_metadata=metadata)

    def test_custom_formula_values_invalidate_shared_cache(self):
        frame = fixture()
        frame["novel"] = np.arange(len(frame),dtype=float)
        metadata = {"novel":{"unit":"1"}}
        spec = DiscoverySpecV2(measurements=("novel",),windows_days=(90,))
        with tempfile.TemporaryDirectory() as root:
            first,_ = materialize(frame,spec,"same-base-source",root,shared=True,input_metadata=metadata)
            changed = frame.assign(novel=frame.novel+1)
            second,_ = materialize(changed,spec,"same-base-source",root,shared=True,input_metadata=metadata)
            self.assertNotEqual(first.matrix_id,second.matrix_id)
