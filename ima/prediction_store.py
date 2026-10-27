"""Immutable, content-addressed forward prediction artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from .research_storage import admitted_output, as_storage_blocked, storage_writer, reserved_output_file


def frame_fingerprint(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps([(str(c), str(t)) for c, t in zip(frame.columns, frame.dtypes)]).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def prediction_key(metadata: dict) -> str:
    return hashlib.sha256(json.dumps(metadata, sort_keys=True, allow_nan=False).encode()).hexdigest()


@dataclass
class PredictionArtifact:
    values: np.ndarray
    row_keys: tuple[tuple[str, str], ...]
    metadata: dict

    def validate(self) -> None:
        values = np.asarray(self.values)
        if values.ndim not in (1, 2) or len(values) != len(self.row_keys):
            raise ValueError("Prediction row mismatch")
        if not np.isfinite(values).all() or len(set(self.row_keys)) != len(self.row_keys):
            raise ValueError("Nonfinite predictions or duplicate race/horse keys")
        if self.metadata.get("fit_scope") != "forward_oof":
            raise ValueError("Reusable training predictions must be forward OOF")
        cutoff = pd.Timestamp(self.metadata["training_cutoff"])
        start = pd.Timestamp(self.metadata["prediction_start"])
        if pd.isna(cutoff) or pd.isna(start) or cutoff >= start:
            raise ValueError("Future-trained prediction reference")


class PredictionStore:
    """One atomic NPZ per complete key; no mutable champion references."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        missing, parent = 0, self.root
        while not parent.exists():
            missing += 1
            parent = parent.parent
        try:
            with admitted_output(self.root, estimated_bytes=max(4096, missing * 65536),
                                 stage='forward_oof_directory', inodes=missing):
                self.root.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            converted = as_storage_blocked(exc, stage='forward_oof_directory', output=self.root)
            if converted is None or converted is exc:
                raise
            raise converted from exc

    @storage_writer(stage='forward_oof',output=lambda a:a['self'].root,
        estimate=lambda a:max(1024**2,2*np.asarray(a['artifact'].values).nbytes+
            8*len(json.dumps(a['artifact'].row_keys).encode())+
            4*len(json.dumps(a['artifact'].metadata,allow_nan=False).encode())))
    def put(self, key: str, artifact: PredictionArtifact) -> None:
        artifact.validate()
        path = self._path(key)
        fd, temporary = tempfile.mkstemp(dir=self.root, suffix=".npz")
        primary = None
        try:
            os.close(fd)
            with reserved_output_file(temporary, "wb") as handle:
                np.savez(handle, values=np.asarray(artifact.values),
                         row_keys=np.asarray(artifact.row_keys, dtype=str),
                         metadata=json.dumps(artifact.metadata, sort_keys=True, allow_nan=False))
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                existing = self.get(key, artifact.row_keys)
                canonical = json.loads(json.dumps(artifact.metadata, sort_keys=True, allow_nan=False))
                if existing.metadata != canonical or not np.array_equal(existing.values, artifact.values):
                    raise ValueError("Immutable prediction key collision")
            directory_fd=os.open(self.root,os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except BaseException as exc:
            primary = exc
            raise
        finally:
            try:
                os.unlink(temporary)
            except BaseException as cleanup:
                if primary is None:
                    raise
                primary.add_note(f"Prediction staging cleanup also failed: {cleanup!r}")

    def get(self, key: str, row_keys=None) -> PredictionArtifact | None:
        path = self._path(key)
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as stored:
            artifact = PredictionArtifact(stored["values"].copy(),
                tuple(tuple(row) for row in stored["row_keys"].tolist()),
                json.loads(str(stored["metadata"])))
        artifact.validate()
        if row_keys is not None and tuple(row_keys) != artifact.row_keys:
            raise ValueError("Prediction artifact row mismatch")
        return artifact

    def _path(self, key: str) -> Path:
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("Prediction key must be a SHA256 hex digest")
        return self.root / (key + ".npz")
