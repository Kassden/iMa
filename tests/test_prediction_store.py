import tempfile
import unittest
import numpy as np
import pandas as pd
from ima.prediction_store import PredictionArtifact, PredictionStore, frame_fingerprint, prediction_key


class PredictionStoreTests(unittest.TestCase):
    def test_readback_collision_row_order_and_cutoff(self):
        metadata = {"fit_scope": "forward_oof", "training_cutoff": "2020-01-01", "prediction_start": "2020-01-02"}
        key = prediction_key(metadata)
        artifact = PredictionArtifact(np.array([0.4, 0.6]), (("r", "h1"), ("r", "h2")), metadata)
        with tempfile.TemporaryDirectory() as path:
            store = PredictionStore(path)
            store.put(key, artifact)
            store.put(key, artifact)
            np.testing.assert_array_equal(store.get(key).values, artifact.values)
            with self.assertRaisesRegex(ValueError, "row mismatch"):
                store.get(key, tuple(reversed(artifact.row_keys)))
            with self.assertRaisesRegex(ValueError, "collision"):
                store.put(key, PredictionArtifact(np.array([0.3, 0.7]), artifact.row_keys, metadata))
        metadata["training_cutoff"] = "2020-01-02"
        with self.assertRaisesRegex(ValueError, "Future-trained"):
            artifact.validate()

    def test_fingerprint_binds_labels_rows_and_columns(self):
        frame = pd.DataFrame({"x": [1, 2], "target_win": [1, 0]})
        self.assertNotEqual(frame_fingerprint(frame), frame_fingerprint(frame.iloc[::-1]))
        changed = frame.copy()
        changed["target_win"] = [0, 1]
        self.assertNotEqual(frame_fingerprint(frame), frame_fingerprint(changed))
        self.assertNotEqual(frame_fingerprint(frame), frame_fingerprint(frame[["target_win", "x"]]))
