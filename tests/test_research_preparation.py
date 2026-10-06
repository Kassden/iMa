import multiprocessing
import os
import pickle
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ima.research_preparation import (
    PreparationKey, PreparationCache, PreparationCacheWaitTimeout, PreparedFold, build_numeric_fold,
)


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


def crash_prepare(root):
    def crash():
        os._exit(23)
    PreparationCache(root).prepare_fold(key(), crash, owner={"attempt_id": "crashed-producer"})


class PreparationTests(unittest.TestCase):
    def test_one_producer_and_26_followers_with_bounded_pool(self):
        with tempfile.TemporaryDirectory() as root:
            started, blocked, release = threading.Event(), threading.Event(), threading.Event()
            import fcntl
            original_flock = fcntl.flock
            def observed_flock(*args):
                try:
                    return original_flock(*args)
                except BlockingIOError:
                    blocked.set()
                    raise
            def builder():
                started.set()
                if not release.wait(5):
                    raise TimeoutError("Test producer was not released")
                return build()
            cache = PreparationCache(root)
            with patch("ima.research_preparation.fcntl.flock", side_effect=observed_flock), \
                 ThreadPoolExecutor(max_workers=4) as pool:
                producer = pool.submit(cache.prepare_fold, key(), builder)
                try:
                    self.assertTrue(started.wait(5))
                    followers = [pool.submit(PreparationCache(root).prepare_fold, key(),
                        lambda: self.fail("Follower recomputed artifact")) for _ in range(26)]
                    self.assertTrue(blocked.wait(5))
                finally:
                    release.set()
                outcomes = [producer.result(timeout=5), *(future.result(timeout=5) for future in followers)]
            self.assertEqual(1, cache.stats["builds"])
            self.assertEqual(1, sum(artifact.cache_status == "miss" for artifact in outcomes))
            self.assertEqual(1, len({artifact.artifact_id for artifact in outcomes}))
            self.assertTrue(any(artifact.cache_status == "wait" for artifact in outcomes))
            for artifact in outcomes:
                np.testing.assert_array_equal(artifact.load_arrays()["train_x"], [[1.], [2.]])

    def test_prewarmed_producer_can_exceed_120_seconds_without_follower_wait(self):
        with tempfile.TemporaryDirectory() as root:
            clock = [0.]
            def slow_builder():
                clock[0] = 121.
                return build()
            cache = PreparationCache(root)
            with patch("ima.research_preparation.time.monotonic", side_effect=lambda: clock[0]):
                producer = cache.prepare_fold(key(), slow_builder)
                follower = cache.prepare_fold(key(), lambda: self.fail("Follower rebuilt"))
            self.assertEqual("miss", producer.cache_status)
            self.assertEqual("hit", follower.cache_status)
            self.assertEqual(0, follower.wait_seconds)
            self.assertEqual(1, cache.stats["builds"])

    def test_wait_timeout_is_typed_bounded_and_keeps_live_lock(self):
        with tempfile.TemporaryDirectory() as root:
            started, release = threading.Event(), threading.Event()
            def builder():
                started.set()
                if not release.wait(5):
                    raise TimeoutError("Test producer was not released")
                return build()
            with ThreadPoolExecutor(max_workers=1) as pool:
                producer = pool.submit(PreparationCache(root).prepare_fold, key(), builder,
                                       owner={"attempt_id": "producer", "fold_id": "fold-001"})
                try:
                    self.assertTrue(started.wait(5))
                    lock = Path(root)/(key().cache_id()+".lock")
                    inode = lock.stat().st_ino
                    before = time.monotonic()
                    with self.assertRaises(PreparationCacheWaitTimeout) as caught:
                        PreparationCache(root, lock_timeout=.04).prepare_fold(key(), lambda: self.fail("Follower built"))
                    self.assertLess(time.monotonic()-before, 1)
                    error = caught.exception
                    self.assertEqual(key().cache_id(), error.cache_key)
                    self.assertEqual("producer", error.owner["attempt_id"])
                    self.assertEqual("building", error.stage)
                    self.assertTrue(error.diagnostic["retryable"])
                    self.assertEqual("infrastructure_cache_wait", error.diagnostic["failure_kind"])
                    self.assertIsNotNone(error.diagnostic["heartbeat_at_epoch"])
                    self.assertEqual(error.diagnostic, pickle.loads(pickle.dumps(error)).diagnostic)
                    self.assertEqual(inode, lock.stat().st_ino)
                    self.assertFalse(PreparationCache(root).evict(key().cache_id()))
                    heartbeat = error.diagnostic["heartbeat_at_epoch"]
                    until = time.monotonic()+2
                    while time.monotonic() < until:
                        observed = PreparationCache(root).owner_status(key().cache_id())
                        if observed["heartbeat_at_epoch"] > heartbeat:
                            break
                        time.sleep(.02)
                    self.assertGreater(observed["heartbeat_at_epoch"], heartbeat)
                    self.assertEqual("building", observed["stage"])
                finally:
                    release.set()
                artifact = producer.result(timeout=5)
                self.assertEqual("hit", PreparationCache(root).prepare_fold(key(), build).cache_status)
                self.assertEqual("released", PreparationCache(root).owner_status(artifact.artifact_id)["stage"])

    def test_unbounded_wait_configuration_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            for timeout in (None, float("inf"), float("nan"), -1):
                with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                    PreparationCache(root, lock_timeout=timeout)

    def test_producer_death_replays_without_deleting_lock(self):
        with tempfile.TemporaryDirectory() as root:
            context = multiprocessing.get_context("spawn")
            producer = context.Process(target=crash_prepare, args=(root,))
            producer.start()
            producer.join(timeout=15)
            if producer.is_alive():
                producer.terminate()
                producer.join(timeout=5)
                self.fail("Crash fixture did not exit")
            self.assertEqual(23, producer.exitcode)
            cache = PreparationCache(root, lock_timeout=.5)
            lock = Path(root)/(key().cache_id()+".lock")
            inode = lock.stat().st_ino
            self.assertFalse((Path(root)/key().cache_id()).exists())
            artifact = cache.prepare_fold(key(), build)
            self.assertEqual("miss", artifact.cache_status)
            self.assertEqual(inode, lock.stat().st_ino)
            self.assertEqual([[1.], [2.]], artifact.load_arrays()["train_x"].tolist())

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
