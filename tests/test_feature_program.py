import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ima.feature_discovery_specs import DiscoverySpec
from ima.feature_program import materialize, synthesize
from ima.feature_screening import DiscoverySelection


def fixture():
    rows = []
    for day in range(1, 9):
        for horse in range(3):
            rows.append(dict(race_id=f"r{day}", horse_no=horse+1, horse_id=f"h{horse}", jockey_id=f"j{horse}", trainer_id="t1", date=f"2020-01-{day:02d}", finish_seconds=60+horse+day, distance=1200, lengths_raw=horse, actual_weight=120+horse, target_win=int(horse == day % 3)))
    return pd.DataFrame(rows)


class FeatureProgramTests(unittest.TestCase):
    def test_replay_and_future_invariance(self):
        frame = fixture()
        spec = DiscoverySpec(windows_days=(365,))
        first, manifest = synthesize(frame, spec)
        again, _ = synthesize(frame, spec)
        pd.testing.assert_frame_equal(first, again)
        altered = frame.copy()
        altered.loc[altered.date >= "2020-01-07", "finish_seconds"] = 999
        changed, _ = synthesize(altered, spec)
        pd.testing.assert_frame_equal(first[frame.date < "2020-01-07"], changed[frame.date < "2020-01-07"])
        self.assertEqual(manifest["engine"], "featuretools")
        mean = next(x["feature_id"] for x in manifest["catalog"] if x["expression"] == "MEAN(starts.speed_mps)")
        self.assertAlmostEqual(first.loc[6, mean], 1200/61)

    def test_cache_corruption_and_duplicate_runner(self):
        spec = DiscoverySpec(windows_days=(90,))
        with tempfile.TemporaryDirectory() as d:
            matrix, meta = materialize(fixture(), spec, "source", Path(d))
            self.assertTrue(any(c.startswith("dfs_") for c in matrix))
            path = Path(d) / meta["matrix_id"] / "matrix.parquet"
            path.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "Corrupt"):
                materialize(fixture(), spec, "source", Path(d))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            synthesize(pd.concat([fixture(), fixture()]), spec)

    def test_late_observation(self):
        frame = fixture()
        spec = DiscoverySpec(windows_days=(365,))
        base, manifest = synthesize(frame, spec)
        frame["observed_at"] = pd.to_datetime(frame.date) + pd.Timedelta(days=1)
        frame.loc[0, "observed_at"] = pd.Timestamp("2020-02-01")
        late, _ = synthesize(frame, spec)
        count = next(x["feature_id"] for x in manifest["catalog"] if x["expression"] == "COUNT(starts)")
        self.assertEqual(base.loc[6, count], 1)
        self.assertEqual(late.loc[6, count], 0)

    def test_selection_uses_train_only(self):
        train = fixture()
        for n in range(3):
            train[f"dfs_{n}"] = np.arange(len(train)) + n
        train["dfs_constant"] = 1
        selected = DiscoverySelection.fit(train, DiscoverySpec(), "win_probability", "target_win")
        self.assertNotIn("dfs_constant", selected.columns)
        self.assertLessEqual(len(selected.columns), 1)
        self.assertTrue(selected.report["fit_races_hash"])

    def test_unsafe_spec(self):
        with self.assertRaises(ValueError):
            DiscoverySpec(measurements=("current_winner",))
        with self.assertRaises(ValueError):
            DiscoverySpec(windows_days=(0,))

    def test_sequences_relative_and_source_selection(self):
        frame=fixture()
        frame["source"]="trusted"
        spec=DiscoverySpec(sequence_windows=(3,),race_relative=True,history_sources=("trusted",))
        values,manifest=synthesize(frame,spec)
        self.assertTrue(any("prior_starts" in f for f in manifest["catalog"]))
        self.assertTrue(any(f["entity"]=="race" for f in manifest["catalog"]))
        for item in manifest["catalog"]:
            if item["expression"]=="center":
                totals=values[item["feature_id"]].groupby(frame.race_id).sum()
                self.assertTrue(np.allclose(totals,0))
        with self.assertRaisesRegex(ValueError,"Unregistered"):
            synthesize(frame,DiscoverySpec(history_sources=("unknown",)))

    def test_temporal_sequential_selector(self):
        frame=fixture()
        rng=np.random.default_rng(17)
        for i in range(4): frame[f"dfs_noise_{i}"]=rng.normal(size=len(frame))
        frame["dfs_signal"]=frame.target_win+rng.normal(0,.1,len(frame))
        spec=DiscoverySpec(selection="sequential",max_selected=2)
        selection=DiscoverySelection.fit(frame,spec,"win_probability","target_win")
        self.assertEqual(selection.report["inner_fold_count"],3)
        self.assertIn("dfs_signal",selection.columns)
