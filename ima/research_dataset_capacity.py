"""Metadata-only conservative storage estimates for research input datasets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


DELTA_FORMAT = "ima.dataset.delta.v2"


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_storage_estimate(path, *, _seen=None, parquet=False):
    """Estimate decoded CSV/Parquet or checksum-bound delta storage without loading tables."""
    path = Path(path).absolute()
    if path.resolve() != path:
        raise ValueError("Noncanonical dataset input")
    seen = set() if _seen is None else set(_seen)
    if path in seen:
        raise ValueError("Dataset parent cycle")
    seen.add(path)

    if path.suffix == ".parquet" or parquet:
        import pyarrow as pa
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(path)
        metadata = parquet_file.metadata
        decoded_width = 0
        for field in parquet_file.schema_arrow:
            dtype = field.type.value_type if pa.types.is_dictionary(field.type) else field.type
            if pa.types.is_integer(dtype) or pa.types.is_floating(dtype) or pa.types.is_temporal(dtype):
                decoded_width += max(8, dtype.bit_width // 8) + 1
            elif pa.types.is_boolean(dtype):
                decoded_width += 2
            else:
                decoded_width += 64
        encoded = sum(metadata.row_group(i).total_byte_size for i in range(metadata.num_row_groups))
        return max(path.stat().st_size, encoded, metadata.num_rows * decoded_width)

    if path.suffix == ".csv":
        # CSV has no row-group/schema metadata. Its encoded size is the available
        # metadata-only floor; the FIT/FOLD wrapper applies the established 6x bound.
        return path.stat().st_size

    if path.name != "features.delta.json":
        raise ValueError("Unsupported dataset representation")
    descriptor = json.loads(path.read_text())
    if descriptor.get("storage_format") != DELTA_FORMAT:
        raise ValueError("Unknown feature descriptor format")
    parent = Path(descriptor["parent_manifest_path"])
    if _digest(parent) != descriptor["parent_manifest_sha256"]:
        raise ValueError("Parent manifest changed")
    manifest = json.loads(parent.read_text())
    table = parent.parent / Path(manifest["features_path"]).name
    if _digest(table) != manifest["files"][table.name]:
        raise ValueError("Parent table changed")
    delta = path.parent / "feature-delta.parquet"
    if _digest(delta) != descriptor["delta_sha256"]:
        raise ValueError("Delta content changed")
    return dataset_storage_estimate(table, _seen=seen) + dataset_storage_estimate(delta, _seen=seen)


def snapshot_storage_estimate(snapshot):
    """Estimate a source snapshot from metadata without creating storage paths.

    Object snapshots are qualified from their declared sibling object root and
    every referenced object's digest and byte count. This intentionally avoids
    constructing ObjectStore, whose constructor creates its root.
    """
    snapshot = Path(snapshot).resolve()
    manifest_path = snapshot / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("storage_format") == "ima.snapshot.objects.v2":
        table = manifest["tables"]["runners.parquet"]
        root = Path(table["object_root"]).absolute()
        expected_root = snapshot.parent / "objects"
        if root.resolve() != root or root != expected_root or not root.is_dir():
            raise ValueError("Snapshot object root outside qualified acquisition output")
        numeric = 0
        for reference in table["partitions"]:
            digest = reference.get("sha256")
            if (not isinstance(digest, str) or len(digest) != 64
                    or set(digest) - set("0123456789abcdef")):
                raise ValueError("Invalid snapshot object identity")
            path = root / digest[:2] / digest
            if path.resolve() != path or not path.is_file():
                raise ValueError("Missing or escaped snapshot object")
            if path.stat().st_size != reference.get("size") or _digest(path) != digest:
                raise ValueError("Snapshot object checksum or size mismatch")
            numeric += dataset_storage_estimate(path, parquet=True)
    else:
        numeric = dataset_storage_estimate(snapshot / "runners.parquet")
    return numeric + (snapshot / "events.jsonl").stat().st_size
