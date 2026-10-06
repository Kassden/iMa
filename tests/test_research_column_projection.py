import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ima.research_executor import RecipeExecutionRequest, _recipe_fold_inputs
from ima.research_specs import PipelineRecipe
from tests import test_research_controller as controller_fixtures


class ColumnProjectionTests(unittest.TestCase):
    def test_simple_win_projection_preserves_features_labels_population_and_metadata(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            dataset, protocol_path = controller_fixtures.ResearchControllerTests().fixture(root)
            import json
            protocol = json.loads(protocol_path.read_text())
            protocol.setdefault("whole_meeting_boundaries",True)
            frame = pd.read_csv(dataset)
            frame["unused_raw_text"] = "not a model input"
            frame.attrs["availability_policy"] = "strict"
            request = RecipeExecutionRequest("a","p",0,PipelineRecipe(schema_version=3),dataset,root/"trial",protocol)
            projected,schema,manifest,_ = _recipe_fold_inputs(request,frame)
            self.assertNotIn("unused_raw_text",projected)
            self.assertTrue(set(schema.features)<=set(projected))
            self.assertEqual(projected.attrs["availability_policy"],"strict")
            pd.testing.assert_frame_equal(projected[["race_id","horse_no","target_win","target_probability"]],
                                          frame[["race_id","horse_no","target_win","target_probability"]])
            legacy = RecipeExecutionRequest("b","p",0,PipelineRecipe(),dataset,root/"trial2",protocol)
            original,_,original_manifest,_ = _recipe_fold_inputs(legacy,frame)
            self.assertIn("unused_raw_text",original)
            self.assertEqual(manifest.protocol_id,original_manifest.protocol_id)
