import tempfile
import unittest
from pathlib import Path

from ima.research_expansion import _FoldDependencies
from ima.research_preparation import PreparationCache
from tests.test_research_preparation import build, key


class FoldDependencyStateTests(unittest.TestCase):
    def test_blocked_dependency_survives_restart_and_late_followers(self):
        with tempfile.TemporaryDirectory() as root:
            state = _FoldDependencies(root)
            for index in range(3):
                state.failed("dependency", "program-"+str(index), RuntimeError("builder failed"))
            replay = _FoldDependencies(root)
            self.assertTrue(replay.blocked("dependency"))
            self.assertFalse(replay.can_prepare("dependency"))
            self.assertEqual(replay.errors["dependency"]["failures"], 3)
            replay.close()
            state.close()

    def test_ready_lease_prevents_eviction_then_releases_on_close(self):
        with tempfile.TemporaryDirectory() as root:
            cache = PreparationCache(Path(root)/"cache")
            artifact = cache.prepare_fold(key(), build)
            state = _FoldDependencies(root)
            state.publish("dependency", {"fold": artifact})
            self.assertTrue(state.is_ready("dependency"))
            self.assertFalse(cache.evict(artifact.artifact_id))
            state.close()
            self.assertTrue(cache.evict(artifact.artifact_id))

    def test_corruption_invalidates_ready_dependency(self):
        with tempfile.TemporaryDirectory() as root:
            artifact = PreparationCache(Path(root)/"cache").prepare_fold(key(), build)
            state = _FoldDependencies(root)
            state.publish("dependency", {"fold": artifact})
            manifest = artifact.path/"manifest.json"
            manifest.write_bytes(manifest.read_bytes()+b" ")
            self.assertFalse(state.is_ready("dependency"))
            self.assertNotIn("dependency", state.ready)
            state.close()

    def test_successful_retry_clears_persisted_active_errors(self):
        with tempfile.TemporaryDirectory() as root:
            artifact = PreparationCache(Path(root)/"cache").prepare_fold(key(), build)
            state = _FoldDependencies(root)
            state.failed("dependency", "program", RuntimeError("transient"))
            state.publish("dependency", {"fold": artifact})
            self.assertNotIn("dependency", state.errors)
            self.assertFalse(_FoldDependencies(root).blocked("dependency"))
            state.close()
