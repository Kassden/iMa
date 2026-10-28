"""Additive immutable Parquet objects. No deletion/GC authority is provided."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile

import pandas as pd
import pyarrow.parquet as pq

from .research_storage import (
    StorageBlocked,
    admitted_output,
    as_storage_blocked,
    reserved_output_file,
    storage_writer,
)


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _storage_error_boundary(stage, output):
    try:
        yield
    except BaseException as error:
        converted = as_storage_blocked(error, stage=stage, output=output)
        if converted is None or converted is error:
            raise
        raise converted from error


class ObjectStore:
    def __init__(self, root):
        self.root = Path(root).absolute()
        if self.root.resolve() != self.root:
            raise ValueError("Noncanonical object root")
        if not self.root.exists():
            missing=0
            created=[]
            current=self.root
            while not current.exists():
                missing+=1
                created.append(current)
                current=current.parent
            with _storage_error_boundary("snapshot_object_root", self.root):
                with admitted_output(self.root, estimated_bytes=missing*65536,
                                     stage="snapshot_object_root", inodes=missing):
                    self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
                    for directory in created:
                        _fsync_directory(directory)
                        _fsync_directory(directory.parent)
        self._validate_root()

    def _validate_root(self):
        if (not self.root.is_dir() or self.root.stat().st_uid != os.getuid()
                or self.root.stat().st_mode & 0o022):
            raise ValueError("Object store must be an owned, non-shared directory")

    def path(self, reference):
        digest = reference["sha256"]
        if (not isinstance(digest, str) or len(digest) != 64
                or set(digest) - set("0123456789abcdef")):
            raise ValueError("Invalid object content identity")
        path = self.root / digest[:2] / digest
        if path.resolve() != path or path.stat().st_uid != os.getuid():
            raise ValueError("Escaped or foreign object")
        if path.stat().st_size != reference["size"] or _hash(path) != digest:
            raise ValueError("Missing/corrupt immutable object")
        return path

    @storage_writer(
        stage="snapshot_object",
        output=lambda arguments: arguments["self"].root,
        estimate=lambda arguments: max(
            16 * 1024**2,
            2 * int(arguments["frame"].memory_usage(index=False, deep=False).sum()),
        ),
    )
    def _put_table_admitted(self, frame):
        temporary = None
        destination = None
        primary_error = None
        try:
            # The decorator's reservation precedes mkstemp and covers its inode
            # and bounded native Parquet writes.
            descriptor, name = tempfile.mkstemp(prefix=".parquet-", dir=self.root)
            os.close(descriptor)
            temporary = Path(name)
            with reserved_output_file(temporary) as stream:
                frame.to_parquet(stream, index=False, compression="zstd")
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())

            digest = _hash(temporary)
            destination = self.root / digest[:2] / digest
            if not destination.parent.exists():
                with _storage_error_boundary("snapshot_object_shard", self.root):
                    with admitted_output(self.root, estimated_bytes=65536,
                                         stage="snapshot_object_shard", inodes=1):
                        try:
                            destination.parent.mkdir(mode=0o700)
                        except FileExistsError:
                            pass
            shard_stat = destination.parent.lstat()
            if (destination.parent.resolve() != destination.parent
                    or not destination.parent.is_dir()
                    or shard_stat.st_uid != os.getuid()
                    or shard_stat.st_mode & 0o022):
                raise ValueError("Unsafe object shard")

            os.chmod(temporary, 0o400)
            try:
                os.link(temporary, destination)
            except FileExistsError:
                # A prior writer may have published this content identity.
                # Validate it below without changing its mode or bytes.
                pass
            # Both the original publisher and a collision-reusing writer must
            # persist the shard and its parent before exposing a reference.
            _fsync_directory(destination.parent)
            _fsync_directory(self.root)

            reference = {
                "sha256": digest,
                "size": temporary.stat().st_size,
                "format": "parquet-zstd-v1",
            }
            self.path(reference)
            return reference
        except BaseException as error:
            primary_error = error
            # A concurrent writer may already have accepted this immutable
            # content identity. Only reference-aware GC may remove it later.
            raise
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except BaseException as cleanup_error:
                    if primary_error is None:
                        raise
                    primary_error.add_note(f"Object temporary retained at {temporary}: {cleanup_error!r}")

    def put_table(self, frame):
        # All direct object publishers participate in the marker. The snapshot
        # builder nests this lease through durable manifest publication.
        from .research_object_lifecycle import object_publication_lease
        with object_publication_lease(self.root):
            return self._put_table_admitted(frame)

    def publish_table(self, frame):
        ordinal = "__ima_object_ordinal__"
        if ordinal in frame:
            raise ValueError("Reserved internal column name")
        work = frame.copy(deep=False).assign(**{ordinal: range(len(frame))})
        if "date" in frame and not frame.empty:
            years = pd.to_datetime(frame.date, errors="raise").dt.year
            if years.isna().any():
                raise ValueError("Missing partition date")
            chunks = [work.loc[years.eq(year)] for year in sorted(years.unique())]
        else:
            chunks = [work]
        return {
            "schema_version": 1,
            "object_root": str(self.root),
            "rows": len(frame),
            "columns": list(frame.columns),
            "ordinal": ordinal,
            "partitions": [self.put_table(chunk) for chunk in chunks],
        }

    def read_table(self, manifest, *, columns=None, positions=None):
        if (manifest.get("schema_version") != 1
                or Path(manifest["object_root"]) != self.root):
            raise ValueError("Unknown table format/object root")
        requested = list(manifest["columns"] if columns is None else columns)
        if len(requested) != len(set(requested)) or set(requested) - set(manifest["columns"]):
            raise ValueError("Unknown/duplicate projected columns")
        ordinal = manifest["ordinal"]
        chunks = [
            pq.read_table(self.path(reference), columns=requested + [ordinal]).to_pandas()
            for reference in manifest["partitions"]
        ]
        frame = pd.concat(chunks, ignore_index=True).sort_values(ordinal, kind="stable")
        if len(frame) != manifest["rows"] or frame[ordinal].tolist() != list(range(len(frame))):
            raise ValueError("Object table row identity changed")
        frame = frame.drop(columns=[ordinal]).reset_index(drop=True)
        if positions is not None:
            frame = frame.iloc[list(positions)].reset_index(drop=True)
        return frame


def read_snapshot_table(snapshot, name, *, columns=None):
    snapshot = Path(snapshot)
    if Path(name).name != name:
        raise ValueError("Invalid snapshot table name")
    manifest = json.loads((snapshot / "manifest.json").read_text())
    if manifest.get("storage_format") == "ima.snapshot.objects.v2":
        table = manifest["tables"][name]
        store = Path(table["object_root"])
        # Shared objects are a sibling of immutable snapshots, not arbitrary paths.
        if store.resolve() != snapshot.parent.resolve() / "objects":
            raise ValueError("Snapshot object root outside qualified acquisition output")
        if not store.is_dir():
            raise FileNotFoundError("Missing immutable snapshot object store")
        # Keep the shared lifecycle lease until the Parquet bytes have actually
        # been consumed; a manifest-only pin would race a prospective collector.
        from .research_object_lifecycle import object_read_lease
        with object_read_lease(store):
            # Re-read the reference manifest after acquiring the lease so the
            # pin covers both root resolution and the Parquet consumer read.
            manifest = json.loads((snapshot / "manifest.json").read_text())
            if manifest.get("storage_format") != "ima.snapshot.objects.v2":
                raise ValueError("Snapshot format changed while acquiring object read lease")
            table = manifest["tables"][name]
            if Path(table["object_root"]) != store:
                raise ValueError("Snapshot object root changed while acquiring read lease")
            return ObjectStore(store).read_table(table, columns=columns)
    return pd.read_parquet(snapshot / name, columns=columns)


def snapshot_storage_estimate(snapshot):
    """Use the committed metadata-only capacity estimator."""
    from .research_dataset_capacity import snapshot_storage_estimate as estimate

    return estimate(snapshot)
