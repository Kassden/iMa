import multiprocessing
import tempfile
import unittest
from unittest.mock import patch
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ima.research_preparation import PreparationKey, PreparationCache, PreparedFold, build_numeric_fold


def key(**changes):
    payload = dict(source_content_hash="source",training_row_keys=("a","b"),calibration_row_keys=("c",),score_row_keys=("d",),target_labels_hash="labels",availability_policy={"strict":True},feature_definitions=({"name":"x"},),selection_settings={"method":"quality"},fitted_transform_settings={"imputer":"median"},fold_dates=("2020-01-01",),seed=17,implementation_revision="test-v1",dependency_versions={"numpy":np.__version__})
    payload.update(changes)
    return PreparationKey(**payload)


def build():
    return PreparedFold({"train_x":np.array([[1.],[2.]]),"calibration_x":np.array([[3.]]),"score_x":np.array([[4.]])},("x",),{"train":("a","b"),"calibration":("c",),"score":("d",)},fitted_state={"value":1})


def concurrent_prepare(arguments):
    import fcntl
    root, ready, blocked = arguments
    original_flock = fcntl.flock
    def observed_flock(*args):
        try:
            return original_flock(*args)
        except BlockingIOError:
            blocked.set()
            raise
    def builder():
        with open(Path(root)/"build_count","a") as stream:
            stream.write("build\n")
        if not blocked.wait(timeout=30):
            raise TimeoutError("No competing worker reached the preparation lock")
        return build()
    ready.wait(timeout=30)
    with patch("ima.research_preparation.fcntl.flock", side_effect=observed_flock):
        artifact = PreparationCache(root).prepare_fold(key(),builder)
    return artifact.cache_status,artifact.load_arrays()["train_x"].tolist()


class PreparationTests(unittest.TestCase):
    def test_reuse_readonly_pin_and_eviction(self):
        with tempfile.TemporaryDirectory() as root:
            cache = PreparationCache(root)
            first = cache.prepare_fold(key(),build)
            second = cache.prepare_fold(key(),lambda:self.fail("Recomputed warm fold"))
            self.assertEqual("miss",first.cache_status)
            self.assertEqual("hit",second.cache_status)
            arrays = second.load_arrays()
            self.assertIsInstance(arrays["train_x"],np.memmap)
            with self.assertRaises(ValueError):
                arrays["train_x"][0,0] = 9
            self.assertEqual({"value":1},second.load_fitted_state())
            with cache.pin(first):
                self.assertFalse(cache.evict(first.artifact_id))
                self.assertEqual("hit",cache.prepare_fold(key(),build).cache_status)
            self.assertTrue(cache.evict(first.artifact_id))

    def test_every_dependency_invalidates(self):
        original = key()
        changes = {"source_content_hash":"new","training_row_keys":("b","a"),"calibration_row_keys":("e",),"score_row_keys":("f",),"target_labels_hash":"newlabels","availability_policy":{"strict":False},"feature_definitions":({"name":"y"},),"selection_settings":{"method":"MI"},"fitted_transform_settings":{"imputer":"mean"},"fold_dates":("2021-01-01",),"seed":18,"implementation_revision":"v2","dependency_versions":{"numpy":"other"}}
        for field,value in changes.items():
            self.assertNotEqual(original.cache_id(),key(**{field:value}).cache_id(),field)
        with tempfile.TemporaryDirectory() as root:
            cache = PreparationCache(root)
            cache.prepare_fold(original,build)
            changed = cache.prepare_fold(key(target_labels_hash="newlabels"),build)
            self.assertEqual("miss",changed.cache_status)

    def test_concurrent_miss_only_builds_once(self):
        with tempfile.TemporaryDirectory() as root:
            context = multiprocessing.get_context("spawn")
            with context.Manager() as manager:
                ready, blocked = manager.Barrier(3), manager.Event()
                with ProcessPoolExecutor(max_workers=3,mp_context=context) as pool:
                    outcomes = list(pool.map(concurrent_prepare,[(root,ready,blocked)]*3))
            self.assertEqual(1,(Path(root)/"build_count").read_text().count("build"))
            self.assertEqual(1,sum(status == "miss" for status,_ in outcomes))
            self.assertTrue(any(status == "wait" for status,_ in outcomes))

    def test_corrupt_and_failed_publish_recovery(self):
        with tempfile.TemporaryDirectory() as root:
            cache = PreparationCache(root)
            artifact = cache.prepare_fold(key(),build)
            (artifact.path/"arrays.joblib").write_bytes(b"corrupt")
            self.assertEqual("miss",cache.prepare_fold(key(),build).cache_status)
            bad = key(target_labels_hash="bad")
            with self.assertRaises(ValueError):
                cache.prepare_fold(bad,lambda:PreparedFold({"train_x":np.array([[object()]],dtype=object)},("x",),build().row_keys))
            self.assertFalse((Path(root)/bad.cache_id()).exists())
            self.assertEqual("miss",cache.prepare_fold(bad,build).cache_status)

    def test_train_only_imputation_replayed(self):
        train = pd.DataFrame({"x":[1.,3.,np.nan],"y":[0.,1.,0.]})
        score = pd.DataFrame({"x":[np.nan],"y":[1.]})
        rows = {"train":("1","2","3"),"calibration":("4",),"score":("5",)}
        result = build_numeric_fold(train,score,score,base_features=("x",),label="y",row_keys=rows)
        self.assertEqual(2.,result.arrays["score_x"][0,0])
        np.testing.assert_equal(result.fitted_state["imputer"].transform([[np.nan]]),[[2.]])
