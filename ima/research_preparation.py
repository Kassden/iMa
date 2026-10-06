"""Pure fold preparation cache with trusted, read-only Joblib descriptors."""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator


class PreparationKey(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    source_content_hash: str
    training_row_keys: tuple[str, ...]
    calibration_row_keys: tuple[str, ...]
    score_row_keys: tuple[str, ...]
    target_labels_hash: str
    availability_policy: dict[str, Any]
    feature_definitions: tuple[dict[str, Any], ...]
    selection_settings: dict[str, Any]
    fitted_transform_settings: dict[str, Any]
    fold_dates: tuple[str, ...]
    seed: int
    implementation_revision: str
    dependency_versions: dict[str, str]

    @model_validator(mode="after")
    def complete(self):
        if not all((self.source_content_hash,self.target_labels_hash,self.implementation_revision,self.dependency_versions)):
            raise ValueError("Source, labels, implementation and dependency identity required")
        groups = [self.training_row_keys,self.calibration_row_keys,self.score_row_keys]
        if not self.training_row_keys or any(len(g) != len(set(g)) for g in groups):
            raise ValueError("Training keys required; each fold population must be unique")
        if any(set(a) & set(b) for i,a in enumerate(groups) for b in groups[i+1:]):
            raise ValueError("Fold populations must be disjoint")
        return self

    def cache_id(self):
        payload = json.dumps(self.model_dump(mode="json"),sort_keys=True,separators=(",",":"),allow_nan=False)
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class PreparedFold:
    arrays: dict[str, np.ndarray]
    feature_names: tuple[str, ...]
    row_keys: dict[str, tuple[str, ...]]
    category_vocabulary: dict = field(default_factory=dict)
    fitted_state: Any = None


def _checksum(path):
    digest = hashlib.sha256()
    with open(path,"rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class PreparationArtifact:
    root: str
    artifact_id: str
    cache_status: str
    wait_seconds: float = 0

    @property
    def path(self):
        if len(self.artifact_id) != 64 or any(c not in "0123456789abcdef" for c in self.artifact_id):
            raise ValueError("Invalid preparation artifact ID")
        return Path(self.root) / self.artifact_id

    def manifest(self):
        manifest = json.loads((self.path/"manifest.json").read_text())
        if manifest["artifact_id"] != self.artifact_id:
            raise ValueError("Preparation identity mismatch")
        for name,digest in manifest["checksums"].items():
            if name not in {"arrays.joblib","fitted.joblib"} or _checksum(self.path/name) != digest:
                raise ValueError("Corrupt preparation artifact")
        return manifest

    def load_arrays(self):
        manifest = self.manifest()
        arrays = joblib.load(self.path/"arrays.joblib",mmap_mode="r")
        for name, array in arrays.items():
            if array.size == 0:
                array.setflags(write=False)
            expected = manifest["arrays"][name]
            if list(array.shape) != expected["shape"] or str(array.dtype) != expected["dtype"] or array.flags.writeable:
                raise ValueError("Preparation array metadata/read-only mismatch")
        if set(arrays) != set(manifest["arrays"]):
            raise ValueError("Preparation arrays missing")
        return arrays

    def load_fitted_state(self):
        manifest = self.manifest()
        return joblib.load(self.path/"fitted.joblib",mmap_mode="r") if "fitted.joblib" in manifest["checksums"] else None


class PreparationCacheWaitTimeout(TimeoutError):
    """Retryable infrastructure contention, not a model/scientific failure."""

    def __init__(self, diagnostic):
        self.diagnostic = diagnostic
        self.cache_key = diagnostic["cache_key"]
        self.owner = diagnostic["owner"]
        self.stage = diagnostic["stage"]
        super().__init__("Preparation cache wait timeout: " + json.dumps(diagnostic, sort_keys=True))

    def __reduce__(self):
        return type(self), (self.diagnostic,)


class PreparationCache:
    """Only use an own-user cache root; Joblib objects are trusted local artifacts.

    Keys must describe every preparation dependency. Estimator settings may be omitted
    only when no selector/transform depends on that estimator. No implicit eviction.
    """
    def __init__(self, root, *, lock_timeout=120):
        self.root = Path(root).absolute()
        self.root.mkdir(parents=True,exist_ok=True,mode=0o700)
        if self.root.stat().st_uid != os.getuid() or self.root.stat().st_mode & 0o022:
            raise ValueError("Preparation cache must be owned by this user and not writable by others")
        if not isinstance(lock_timeout, (int, float)) or not math.isfinite(lock_timeout) or lock_timeout < 0:
            raise ValueError("Preparation lock timeout must be finite and nonnegative")
        self.lock_timeout = float(lock_timeout)
        self.stats = {"hit":0,"miss":0,"wait":0,"builds":0}

    def owner_status(self, identity):
        """Last observed owner; flock, never this diagnostic, grants ownership."""
        PreparationArtifact(str(self.root), identity, "hit").path
        try:
            status = json.loads((self.root/(identity+".owner.json")).read_text())
            return status if isinstance(status, dict) else {}
        except (OSError, ValueError):
            return {}

    @contextmanager
    def _producer_status(self, identity, owner):
        status = {"cache_key": identity, "owner": {**(owner or {}), "pid": os.getpid()},
                  "stage": "validating", "acquired_at_epoch": time.time()}
        stopped = threading.Event()
        guard = threading.Lock()

        def update(stage=None):
            with guard:
                if stage is not None:
                    status["stage"] = stage
                status["heartbeat_at_epoch"] = time.time()
                path = self.root/(identity+".owner.json")
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(status, sort_keys=True))
                os.replace(temporary, path)

        def heartbeat():
            while not stopped.wait(1):
                update()

        update()
        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            yield update
        finally:
            stopped.set()
            thread.join()
            update("released")

    @contextmanager
    def _lock(self, identity, *, shared=False, owner=None, stage="preparing"):
        with open(self.root/(identity+".lock"),"a+b") as stream:
            start = time.monotonic()
            deadline_epoch = time.time() + self.lock_timeout
            waited = False
            while True:
                try:
                    fcntl.flock(stream, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    waited = True
                    if time.monotonic()-start >= self.lock_timeout:
                        observed = self.owner_status(identity)
                        raise PreparationCacheWaitTimeout({
                            "failure_kind": "infrastructure_cache_wait", "retryable": True,
                            "cache_key": identity, "owner": observed.get("owner"),
                            "stage": observed.get("stage", "unknown"), "waiting_stage": stage,
                            "heartbeat_at_epoch": observed.get("heartbeat_at_epoch"),
                            "wait_seconds": time.monotonic()-start,
                            "lock_timeout_seconds": self.lock_timeout,
                            "deadline_at_epoch": deadline_epoch,
                        })
                    time.sleep(min(.02, max(0, self.lock_timeout-(time.monotonic()-start))))
            try:
                if shared:
                    yield waited, time.monotonic()-start
                else:
                    with self._producer_status(identity, owner) as update:
                        yield waited, time.monotonic()-start, update
            finally:
                fcntl.flock(stream,fcntl.LOCK_UN)

    def prepare_fold(self, key: PreparationKey, builder, *, owner=None) -> PreparationArtifact:
        key = key if isinstance(key,PreparationKey) else PreparationKey.model_validate(key)
        identity = key.cache_id()
        destination = self.root/identity
        if destination.exists():
            with self._lock(identity,shared=True,stage="reading") as (waited,seconds):
                artifact = PreparationArtifact(str(self.root),identity,"wait" if waited else "hit",seconds)
                try:
                    if artifact.manifest()["key"] != key.model_dump(mode="json"):
                        raise ValueError("Preparation key mismatch")
                    artifact.load_arrays()
                    self.stats[artifact.cache_status] += 1
                    return artifact
                except (OSError,ValueError,KeyError,EOFError):
                    pass
        with self._lock(identity,owner=owner) as (waited, seconds, update):
            artifact = PreparationArtifact(str(self.root),identity,"wait" if waited else "hit",seconds)
            try:
                manifest = artifact.manifest()
                if manifest["key"] != key.model_dump(mode="json"):
                    raise ValueError("Preparation key mismatch")
                artifact.load_arrays()
                self.stats[artifact.cache_status] += 1
                return artifact
            except (OSError,ValueError,KeyError,EOFError):
                pass
            if destination.exists():
                shutil.rmtree(destination)
            temporary = Path(tempfile.mkdtemp(prefix=identity+"-",dir=self.root))
            try:
                update("building")
                prepared = builder()
                if not isinstance(prepared,PreparedFold):
                    raise TypeError("Preparation builder must return PreparedFold")
                self._validate(prepared,key)
                update("writing")
                arrays = {name:np.ascontiguousarray(array) for name,array in prepared.arrays.items()}
                joblib.dump(arrays,temporary/"arrays.joblib",compress=0)
                checksums = {"arrays.joblib":_checksum(temporary/"arrays.joblib")}
                if prepared.fitted_state is not None:
                    joblib.dump(prepared.fitted_state,temporary/"fitted.joblib",compress=0)
                    checksums["fitted.joblib"] = _checksum(temporary/"fitted.joblib")
                manifest = {"schema_version":1,"artifact_id":identity,"key":key.model_dump(mode="json"),"feature_names":prepared.feature_names,"row_keys":prepared.row_keys,"category_vocabulary":prepared.category_vocabulary,"arrays":{name:{"dtype":str(a.dtype),"shape":list(a.shape),"nbytes":a.nbytes} for name,a in arrays.items()},"checksums":checksums}
                with open(temporary/"manifest.json","w") as stream:
                    json.dump(manifest,stream,sort_keys=True,allow_nan=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                for name in checksums:
                    with open(temporary/name,"rb") as stream:
                        os.fsync(stream.fileno())
                update("publishing")
                os.replace(temporary,destination)
                directory_fd = os.open(self.root,os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
                self.stats["miss"] += 1
                self.stats["builds"] += 1
                result = PreparationArtifact(str(self.root),identity,"miss",seconds)
                result.load_arrays()
                return result
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)

    prepare = prepare_fold

    @staticmethod
    def _validate(prepared, key):
        expected = {"train":key.training_row_keys,"calibration":key.calibration_row_keys,"score":key.score_row_keys}
        if {name:tuple(rows) for name,rows in prepared.row_keys.items()} != expected:
            raise ValueError("Prepared row keys must match every ordered fold population")
        if len(prepared.feature_names) != len(set(prepared.feature_names)) or not prepared.feature_names:
            raise ValueError("Unique ordered feature names required")
        if not prepared.arrays:
            raise ValueError("No prepared arrays")
        for name,array in prepared.arrays.items():
            if not name.isidentifier() or not isinstance(array,np.ndarray) or array.dtype.kind not in "biuf" or array.ndim not in {1,2}:
                raise ValueError("Prepared arrays must be numeric one/two dimensional buffers")
            population = name.split("_",1)[0]
            if population not in expected or len(array) != len(expected[population]):
                raise ValueError("Array population/shape mismatch")
            if array.ndim == 2 and array.shape[1] != len(prepared.feature_names):
                raise ValueError("Feature order/shape mismatch")

    @contextmanager
    def pin(self, artifact):
        if Path(artifact.root) != self.root:
            raise ValueError("Foreign preparation artifact")
        with self._lock(artifact.artifact_id,shared=True,stage="pinning"):
            artifact.manifest()
            yield artifact

    def evict(self, artifact_id):
        artifact = PreparationArtifact(str(self.root),artifact_id,"hit")
        artifact.path  # Validate identifier before forming the lock path.
        with open(self.root/(artifact_id+".lock"),"a+b") as stream:
            try:
                fcntl.flock(stream,fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            try:
                if artifact.path.exists():
                    shutil.rmtree(artifact.path)
                return True
            finally:
                fcntl.flock(stream,fcntl.LOCK_UN)


def build_numeric_fold(train, calibration, score, *, base_features, label, row_keys, discovery_spec=None, target_kind="win_probability", dtype="float64"):
    """Fit screening/imputation exactly once on training rows; transform other folds.

    Use as the prepare_fold builder. row_keys has train/calibration/score tuples.
    The caller binds this data/spec/target to the complete PreparationKey.
    """
    from sklearn.impute import SimpleImputer
    from .feature_screening import DiscoverySelection
    selection = DiscoverySelection.fit(train,discovery_spec,target_kind,label) if discovery_spec is not None else None
    names = tuple(dict.fromkeys([*base_features,*(selection.columns if selection else ())]))
    if not names or label in names:
        raise ValueError("Nonempty predictor columns required; target cannot be a predictor")
    transform = SimpleImputer(strategy="median",keep_empty_features=True)
    arrays = {}
    for population,frame in (("train",train),("calibration",calibration),("score",score)):
        values = frame.loc[:,names].replace([np.inf,-np.inf],np.nan).to_numpy(dtype=dtype)
        if population == "train":
            values = transform.fit_transform(values)
        elif len(frame):
            values = transform.transform(values)
        arrays[population+"_x"] = np.asarray(values,dtype=dtype)
        if label in frame:
            arrays[population+"_y"] = frame[label].to_numpy(dtype=dtype)
    return PreparedFold(arrays,names,row_keys,fitted_state={"imputer":transform,"selection":selection,"label":label,"target_kind":target_kind})
