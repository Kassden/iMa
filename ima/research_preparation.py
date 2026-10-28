"""Pure fold preparation cache with trusted, read-only Joblib descriptors."""
from __future__ import annotations

import fcntl
import hashlib
import io
import json
import math
import os
import stat
import shutil
import tempfile
import time
import threading
from contextlib import contextmanager, nullcontext
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


def _open_cache_directory(root, identity):
    """Open and bind the cache root and one child directory without following links."""
    root_fd = directory_fd = -1
    try:
        root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        root_info = os.fstat(root_fd)
        path_info = os.stat(root, follow_symlinks=False)
        if (not stat.S_ISDIR(root_info.st_mode) or root_info.st_uid != os.getuid()
                or (root_info.st_dev, root_info.st_ino) != (path_info.st_dev, path_info.st_ino)):
            raise ValueError("Preparation cache root changed or is not an owned directory")
        directory_fd = os.open(identity, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                               dir_fd=root_fd)
        directory_info = os.fstat(directory_fd)
        path_info = os.stat(identity, dir_fd=root_fd, follow_symlinks=False)
        if (not stat.S_ISDIR(directory_info.st_mode) or directory_info.st_uid != os.getuid()
                or directory_info.st_dev != root_info.st_dev
                or (directory_info.st_dev, directory_info.st_ino) !=
                   (path_info.st_dev, path_info.st_ino)):
            raise ValueError("Preparation cache artifact changed or is not an owned directory")
        return root_fd, directory_fd, (root_info.st_dev, root_info.st_ino), (
            directory_info.st_dev, directory_info.st_ino)
    except BaseException:
        if directory_fd >= 0:
            os.close(directory_fd)
        if root_fd >= 0:
            os.close(root_fd)
        raise


def _qualify_cache_directory(root, root_fd, directory_fd, identity, root_identity,
                             directory_identity):
    """Validate the exact manifest and entries through already-bound descriptors."""
    try:
        root_info = os.fstat(root_fd)
        directory_info = os.fstat(directory_fd)
        root_path_info = os.stat(root, follow_symlinks=False)
        directory_path_info = os.stat(identity, dir_fd=root_fd, follow_symlinks=False)
        if ((root_info.st_dev, root_info.st_ino) != root_identity
                or (root_path_info.st_dev, root_path_info.st_ino) != root_identity
                or (directory_info.st_dev, directory_info.st_ino) != directory_identity
                or (directory_path_info.st_dev, directory_path_info.st_ino) != directory_identity):
            return None

        entries = {}
        for name in os.listdir(directory_fd):
            info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                    or info.st_uid != os.getuid() or info.st_dev != root_info.st_dev):
                return None
            entries[name] = info

        manifest_info = entries.get("manifest.json")
        if manifest_info is None:
            return None
        descriptor = os.open("manifest.json", os.O_RDONLY | os.O_NOFOLLOW,
                             dir_fd=directory_fd)
        try:
            opened_info = os.fstat(descriptor)
            if (not stat.S_ISREG(opened_info.st_mode) or opened_info.st_nlink != 1
                    or opened_info.st_uid != os.getuid()
                    or opened_info.st_dev != root_info.st_dev
                    or (opened_info.st_dev, opened_info.st_ino) !=
                       (manifest_info.st_dev, manifest_info.st_ino)):
                return None
            with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
                descriptor = -1
                manifest = json.load(stream)
        finally:
            if descriptor >= 0:
                os.close(descriptor)

        if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1
                or manifest.get("artifact_id") != identity):
            return None
        checksums = manifest.get("checksums")
        if not isinstance(checksums, dict) or "arrays.joblib" not in checksums:
            return None
        payload_names = set(checksums)
        if not payload_names <= {"arrays.joblib", "fitted.joblib"}:
            return None
        if any(not isinstance(value, str) or len(value) != 64
               or set(value) - set("0123456789abcdef") for value in checksums.values()):
            return None
        if set(entries) != payload_names | {"manifest.json"}:
            return None
        key = PreparationKey.model_validate(manifest.get("key"))
        if key.cache_id() != identity:
            return None
        return {name: (info.st_dev, info.st_ino) for name, info in entries.items()}
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def _remove_known_cache_payload(root, identity):
    """Remove only a structurally proven cache payload, using bound directory FDs."""
    from .research_cache_policy import cache_is_held
    if cache_is_held(root/identity):
        from .research_storage import StorageBlocked
        raise StorageBlocked('Preparation rebuild is held by the pinned retention policy')
    try:
        root_fd, directory_fd, root_identity, directory_identity = _open_cache_directory(
            root, identity)
    except (OSError, ValueError) as exc:
        raise ValueError("Refusing to rebuild preparation cache with unknown or linked contents") from exc
    try:
        entries = _qualify_cache_directory(root, root_fd, directory_fd, identity, root_identity,
                                           directory_identity)
        if entries is None:
            raise ValueError("Refusing to rebuild preparation cache with unknown or linked contents")
        for name, inode in sorted(entries.items()):
            current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if (current.st_dev, current.st_ino) != inode:
                raise ValueError("Preparation cache payload changed during safe cleanup")
            os.unlink(name, dir_fd=directory_fd)
        current = os.stat(identity, dir_fd=root_fd, follow_symlinks=False)
        if (current.st_dev, current.st_ino) != directory_identity:
            raise ValueError("Preparation cache directory changed during safe cleanup")
        os.rmdir(identity, dir_fd=root_fd)
    finally:
        os.close(directory_fd)
        os.close(root_fd)


def _open_eviction_locks(root, identity, *, allow_create=None):
    """Open stable locks; configured eviction never creates them under authority locks."""
    if allow_create is None:
        allow_create = not bool(os.environ.get("IMA_STORAGE_ROOT"))
    paths = (root / (identity + ".pin.lock"), root / (identity + ".lock"))
    root_stat = root.stat()
    missing = 0
    for path in paths:
        try:
            info = path.lstat()
        except FileNotFoundError:
            missing += 1
            continue
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or info.st_uid != os.getuid() or info.st_dev != root_stat.st_dev):
            return None

    if missing and not allow_create:
        return None
    if missing:
        from .research_storage import admitted_output
        admission = admitted_output(root, estimated_bytes=missing * 4096,
                                    stage="preparation_cache_eviction_locks", inodes=missing)
    else:
        admission = nullcontext()
    with admission:
        descriptors = []
        try:
            for path in paths:
                try:
                    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL |
                                         os.O_NOFOLLOW, 0o600)
                except FileExistsError:
                    descriptor = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
                info = os.fstat(descriptor)
                if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                        or info.st_uid != os.getuid() or info.st_dev != root_stat.st_dev):
                    os.close(descriptor)
                    for opened in descriptors:
                        os.close(opened)
                    return None
                descriptors.append(descriptor)
            return descriptors
        except BaseException:
            for descriptor in descriptors:
                os.close(descriptor)
            raise


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
        if os.environ.get('IMA_STORAGE_ROOT') and not self.root.exists():
            from .research_storage import admitted_output, as_storage_blocked
            missing, parent = 0, self.root
            while not parent.exists():
                missing += 1
                parent = parent.parent
            try:
                with admitted_output(self.root, estimated_bytes=missing*65536,
                                     stage='preparation_directory', inodes=missing):
                    self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
            except Exception as exc:
                converted = as_storage_blocked(exc, stage='preparation_directory', output=self.root)
                if converted is None or converted is exc:
                    raise
                raise converted from exc
        else:
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
        heartbeat_errors = []

        def update(stage=None):
            with guard:
                if heartbeat_errors:
                    raise heartbeat_errors[0]
                if stage is not None:
                    status["stage"] = stage
                status["heartbeat_at_epoch"] = time.time()
                path = self.root/(identity+".owner.json")
                if os.environ.get('IMA_STORAGE_ROOT'):
                    from .research_controller import _write_json_atomic
                    _write_json_atomic(path, status)
                else:
                    temporary = path.with_suffix(".tmp")
                    temporary.write_text(json.dumps(status, sort_keys=True))
                    os.replace(temporary, path)

        def heartbeat():
            while not stopped.wait(1):
                try:
                    update()
                except BaseException as exc:
                    heartbeat_errors.append(exc)
                    stopped.set()

        update()
        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        primary = None
        try:
            yield update
        except BaseException as exc:
            primary = exc
            raise
        finally:
            stopped.set()
            thread.join()
            try:
                update("released")
            except BaseException as cleanup:
                if primary is None:
                    raise
                primary.add_note(f"Preparation owner finalization also failed: {cleanup!r}")

    @contextmanager
    def _lock(self, identity, *, shared=False, owner=None, stage="preparing", pins=None):
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
            transferred=False
            def retain():
                nonlocal transferred
                if pins is None or transferred: return
                pin=open(self.root/(identity+'.pin.lock'),'a+b')
                descriptor=None
                try:
                    descriptor=os.dup(stream.fileno())
                    fcntl.flock(pin,fcntl.LOCK_SH)
                    fcntl.flock(stream,fcntl.LOCK_SH)
                    pins.callback(pin.close)
                    pins.callback(os.close,descriptor)
                    transferred=True
                except BaseException as primary:
                    for close in (pin.close, lambda: os.close(descriptor) if descriptor is not None else None):
                        try:
                            close()
                        except BaseException as cleanup:
                            primary.add_note(f'Preparation pin setup cleanup also failed: {cleanup!r}')
                    raise
            try:
                if shared:
                    yield waited, time.monotonic()-start, retain
                else:
                    with self._producer_status(identity, owner) as update:
                        yield waited, time.monotonic()-start, update, retain
            finally:
                if not transferred: fcntl.flock(stream,fcntl.LOCK_UN)

    def prepare_fold(self, key: PreparationKey, builder, *, owner=None, pins=None) -> PreparationArtifact:
        control = os.environ.get('IMA_STORAGE_ROOT')
        if not control:
            return self._prepare_fold(key,builder,owner=owner,pins=pins)
        from .research_storage import StorageAuthority
        from .research_cache_capacity import trim_for_capacity
        authority=StorageAuthority(control)
        key=key if isinstance(key,PreparationKey) else PreparationKey.model_validate(key)
        columns=max((len(d.get('numeric',())) for d in key.feature_definitions),default=32)
        rows=sum(map(len,(key.training_row_keys,key.calibration_row_keys,key.score_row_keys)))
        path=self.root/key.cache_id()
        if path.exists():
            try:
                allocated=sum(p.stat().st_blocks*512 for p in path.iterdir())
                authority.record_cache(self.root,key.cache_id(),allocated,enforce_limit=False)
            except FileNotFoundError:
                pass
            trim_for_capacity(authority, required_bytes=1024**2, required_inodes=8,
                              cache_growth_bytes=0)
        exists=path.exists()
        estimate=1024**2 if exists else max(16*1024**2,rows*max(columns,1)*16+16*1024**2)
        if not exists:
            trim_for_capacity(authority, required_bytes=estimate)
        with authority.admitted(bytes_required=estimate,inodes_required=8,
                                stage='fold_preparation',output=self.root,
                                cache_bytes=0 if exists else estimate) as token:
            budget={'authority':authority,'token':token,'limit':estimate,'written':0,'cache':True,'consume':True}
            def bounded_builder():
                prepared=builder()
                if not isinstance(prepared,PreparedFold):
                    return prepared
                required=sum(a.nbytes for a in prepared.arrays.values())*2+16*1024**2
                if required>estimate:
                    authority.extend(token,extra_bytes=required-estimate,
                                     extra_cache_bytes=required-estimate)
                    budget['limit']=required
                return prepared
            def ready(artifact):
                allocated=sum(p.stat().st_blocks*512 for p in artifact.path.iterdir())
                authority.commit_cache(token,self.root,artifact.artifact_id,allocated)
            try:
                return self._prepare_fold(key,bounded_builder,owner=owner,write_budget=budget,
                                          pins=pins,on_ready=ready)
            except Exception as exc:
                from .research_storage import as_storage_blocked
                converted = as_storage_blocked(exc, stage='fold_preparation', output=self.root)
                if converted is None or converted is exc:
                    raise
                raise converted from exc

    def _prepare_fold(self, key: PreparationKey, builder, *, owner=None,write_budget=None,
                      pins=None,on_ready=None) -> PreparationArtifact:
        key = key if isinstance(key,PreparationKey) else PreparationKey.model_validate(key)
        identity = key.cache_id()
        destination = self.root/identity
        if destination.exists():
            with self._lock(identity,shared=True,stage="reading",pins=pins) as (waited,seconds,retain):
                artifact = PreparationArtifact(str(self.root),identity,"wait" if waited else "hit",seconds)
                try:
                    if artifact.manifest()["key"] != key.model_dump(mode="json"):
                        raise ValueError("Preparation key mismatch")
                    artifact.load_arrays()
                    if on_ready: on_ready(artifact)
                    retain()
                    self.stats[artifact.cache_status] += 1
                    return artifact
                except (OSError,ValueError,KeyError,EOFError):
                    pass
        with self._lock(identity,owner=owner,pins=pins) as (waited, seconds, update,retain):
            artifact = PreparationArtifact(str(self.root),identity,"wait" if waited else "hit",seconds)
            try:
                manifest = artifact.manifest()
                if manifest["key"] != key.model_dump(mode="json"):
                    raise ValueError("Preparation key mismatch")
                artifact.load_arrays()
                if on_ready: on_ready(artifact)
                retain()
                self.stats[artifact.cache_status] += 1
                return artifact
            except (OSError,ValueError,KeyError,EOFError):
                pass
            if destination.exists() or destination.is_symlink():
                _remove_known_cache_payload(self.root, identity)
            temporary = Path(tempfile.mkdtemp(prefix=identity+"-",dir=self.root))
            primary = None
            try:
                update("building")
                prepared = builder()
                if not isinstance(prepared,PreparedFold):
                    raise TypeError("Preparation builder must return PreparedFold")
                self._validate(prepared,key)
                update("writing")
                arrays = {name:np.ascontiguousarray(array) for name,array in prepared.arrays.items()}
                def dump(value,path):
                    if write_budget is None:
                        return joblib.dump(value,path,compress=0)
                    from .research_storage import ReservedFile
                    with open(path,'wb') as stream:
                        bounded=ReservedFile(stream,write_budget)
                        result=joblib.dump(value,bounded,compress=0)
                        bounded.flush()
                        return result
                dump(arrays,temporary/"arrays.joblib")
                checksums = {"arrays.joblib":_checksum(temporary/"arrays.joblib")}
                if prepared.fitted_state is not None:
                    dump(prepared.fitted_state,temporary/"fitted.joblib")
                    checksums["fitted.joblib"] = _checksum(temporary/"fitted.joblib")
                manifest = {"schema_version":1,"artifact_id":identity,"key":key.model_dump(mode="json"),"feature_names":prepared.feature_names,"row_keys":prepared.row_keys,"category_vocabulary":prepared.category_vocabulary,"arrays":{name:{"dtype":str(a.dtype),"shape":list(a.shape),"nbytes":a.nbytes} for name,a in arrays.items()},"checksums":checksums}
                with open(temporary/'manifest.json','wb') as raw:
                    if write_budget is not None:
                        from .research_storage import ReservedFile
                    buffer=raw if write_budget is None else ReservedFile(raw,write_budget)
                    with io.TextIOWrapper(buffer,encoding='utf-8') as stream:
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
                if on_ready: on_ready(result)
                retain()
                return result
            except BaseException as exc:
                primary = exc
                raise
            finally:
                try:
                    if temporary.exists():
                        shutil.rmtree(temporary)
                except BaseException as cleanup:
                    if primary is None:
                        raise
                    primary.add_note(f"Preparation staging cleanup also failed: {cleanup!r}")

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

    def prepare_eviction(self, artifact_id):
        """Provision missing stable metadata before the caller takes authority.lock.

        This never removes payloads. Admission failure leaves the cache held.
        """
        artifact = PreparationArtifact(str(self.root), artifact_id, "hit")
        artifact.path
        from .research_cache_policy import cache_is_held
        if cache_is_held(artifact.path):
            return False
        root_fd = directory_fd = -1
        try:
            root_fd, directory_fd, root_identity, directory_identity = _open_cache_directory(self.root, artifact_id)
            if _qualify_cache_directory(self.root, root_fd, directory_fd, artifact_id,
                                        root_identity, directory_identity) is None:
                return False
            descriptors = _open_eviction_locks(self.root, artifact_id, allow_create=True)
            if descriptors is None:
                return False
            for descriptor in descriptors:
                os.close(descriptor)
            return True
        except (OSError, ValueError):
            return False
        finally:
            if directory_fd >= 0:
                os.close(directory_fd)
            if root_fd >= 0:
                os.close(root_fd)

    def evict(self, artifact_id):
        artifact = PreparationArtifact(str(self.root),artifact_id,"hit")
        artifact.path  # Validate identifier before forming the lock path.
        from .research_cache_policy import cache_is_held
        if cache_is_held(artifact.path):return False
        root_fd = directory_fd = -1
        descriptors = []
        locked = []
        try:
            try:
                root_fd, directory_fd, root_identity, directory_identity = _open_cache_directory(
                    self.root, artifact_id)
            except (OSError, ValueError):
                return False
            if _qualify_cache_directory(self.root, root_fd, directory_fd, artifact_id, root_identity,
                                        directory_identity) is None:
                return False
            opened = _open_eviction_locks(self.root, artifact_id)
            if opened is None:
                return False
            descriptors = opened
            for descriptor in descriptors:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return False
                locked.append(descriptor)

            # The candidate may have changed while lock files were opened. Do a
            # complete second check while both stable locks exclude cooperators.
            entries = _qualify_cache_directory(self.root, root_fd, directory_fd, artifact_id,
                                                root_identity, directory_identity)
            if entries is None:
                return False
            for name, inode in sorted(entries.items()):
                current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if (current.st_dev, current.st_ino) != inode:
                    return False
                os.unlink(name, dir_fd=directory_fd)
            current = os.stat(artifact_id, dir_fd=root_fd, follow_symlinks=False)
            if (current.st_dev, current.st_ino) != directory_identity:
                return False
            os.rmdir(artifact_id, dir_fd=root_fd)
            return True
        finally:
            for descriptor in reversed(locked):
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            for descriptor in descriptors:
                os.close(descriptor)
            if directory_fd >= 0:
                os.close(directory_fd)
            if root_fd >= 0:
                os.close(root_fd)


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
