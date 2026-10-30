"""Inventory compressed snapshot inputs and durably hold unreferenced bytes.

The caller supplies raw roots and capture-file hashes from verified lineage.
This module never parses or decompresses raw input and exposes no source or
held-payload deletion path; it only removes its own failed publication temps.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import fcntl
import json
import os
from pathlib import Path
import re
import stat
import uuid

from .research_storage import admitted_output, reserved_output_file


SCHEMA = "ima.snapshot-input-inventory.v1"
STREAM_CHUNK_BYTES = 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ROOT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _canonical(path: str | os.PathLike[str], *, exists: bool = True) -> Path:
    value = Path(path)
    if not value.is_absolute() or ".." in value.parts:
        raise ValueError(f"Path must be absolute and normalized: {value}")
    if value.resolve(strict=exists) != value:
        raise ValueError(f"Noncanonical path or symlink rejected: {value}")
    return value


def _owned_directory(path: Path, *, label: str) -> os.stat_result:
    path = _canonical(path)
    info = path.lstat()
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
            or info.st_mode & 0o022):
        raise ValueError(f"{label} must be an owned, non-shared real directory: {path}")
    return info


def _source_signature(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_uid, info.st_mode, info.st_nlink,
            info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _generation(path: Path, descriptor: int, expected: tuple[int, ...]) -> None:
    opened = os.fstat(descriptor)
    try:
        named = path.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"Source disappeared during inventory: {path}") from exc
    if (not stat.S_ISREG(opened.st_mode) or stat.S_ISLNK(named.st_mode)
            or _source_signature(opened) != expected
            or _source_signature(named) != expected):
        raise ValueError(f"Source generation changed during inventory: {path}")


def _open_source(path: Path, expected: tuple[int, ...]) -> int:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or _source_signature(info) != expected:
            raise ValueError(f"Source changed before read: {path}")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _hash_source(path: Path, expected: tuple[int, ...]) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    descriptor = _open_source(path, expected)
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            while True:
                block = source.read(STREAM_CHUNK_BYTES)
                if not block:
                    break
                digest.update(block)
                size += len(block)
                _generation(path, descriptor, expected)
        _generation(path, descriptor, expected)
    finally:
        os.close(descriptor)
    if size != expected[5]:
        raise ValueError(f"Source size changed during hashing: {path}")
    return digest.hexdigest(), size


def _scan_roots(raw_roots: dict[str, str | os.PathLike[str]]) -> tuple[list[dict], list[dict]]:
    if not isinstance(raw_roots, dict) or not raw_roots:
        raise ValueError("raw_roots must be a nonempty mapping of explicit root IDs to paths")
    roots: list[dict] = []
    files: list[dict] = []
    seen_roots: list[Path] = []
    for root_id, raw_path in sorted(raw_roots.items()):
        if not isinstance(root_id, str) or not _ROOT_ID.fullmatch(root_id):
            raise ValueError(f"Invalid raw root ID: {root_id!r}")
        root = _canonical(raw_path, exists=False)
        present = root.exists()
        _owned_directory(root if present else root.parent, label=f"Raw root {root_id}" if present else f"Parent of absent raw root {root_id}")
        if any(root == other or root.is_relative_to(other) or other.is_relative_to(root)
               for other in seen_roots):
            raise ValueError("Explicit raw roots must not overlap")
        seen_roots.append(root)
        roots.append({"root_id": root_id, "path": str(root),
                      "status": "present" if present else "absent_at_inventory"})
        stack = [root] if present else []
        while stack:
            directory = stack.pop()
            _owned_directory(directory, label=f"Raw directory for {root_id}")
            with os.scandir(directory) as entries:
                ordered = sorted(entries, key=lambda entry: entry.name)
            child_dirs = []
            for entry in ordered:
                path = directory / entry.name
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(info.st_mode):
                    raise ValueError(f"Symlink encountered in explicit raw root: {path}")
                if info.st_uid != os.getuid():
                    raise ValueError(f"Foreign-owned entry in explicit raw root: {path}")
                if stat.S_ISDIR(info.st_mode):
                    if info.st_mode & 0o022:
                        raise ValueError(f"Shared-writable raw directory rejected: {path}")
                    child_dirs.append(path)
                    continue
                if not stat.S_ISREG(info.st_mode):
                    raise ValueError(f"Non-regular entry in explicit raw root: {path}")
                if path.suffix.lower() != ".gz":
                    continue
                if path.resolve() != path:
                    raise ValueError(f"Noncanonical raw source rejected: {path}")
                files.append({
                    "root_id": root_id,
                    "path": str(path),
                    "relative_path": path.relative_to(root).as_posix(),
                    "generation": _source_signature(info),
                    "size_bytes": info.st_size,
                })
            stack.extend(reversed(child_dirs))
    files.sort(key=lambda row: (row["root_id"], row["relative_path"]))
    return roots, files


def _validate_hold_root(output_parent: Path, *, create: bool) -> Path:
    _owned_directory(output_parent, label="Snapshot output parent")
    hold_root = output_parent / "unknown-raw"
    if hold_root.exists() or hold_root.is_symlink():
        _owned_directory(hold_root, label="Unknown raw hold root")
    elif create:
        _require_storage_authority()
        # Directory creation is admitted through the existing shared authority.
        with admitted_output(hold_root, estimated_bytes=0,
                             stage="snapshot_unknown_raw_root", inodes=1):
            try:
                hold_root.mkdir(mode=0o700)
            except FileExistsError:
                pass
            _owned_directory(hold_root, label="Unknown raw hold root")
            _fsync_directory(output_parent)
    return _canonical(hold_root, exists=create or hold_root.exists())


def _require_storage_authority() -> Path:
    configured = os.environ.get("IMA_STORAGE_ROOT")
    if not configured:
        raise RuntimeError("Unknown raw preservation requires configured shared storage admission")
    authority = _canonical(configured)
    _owned_directory(authority, label="Shared storage authority")
    return authority


@contextmanager
def _publication_lock(hold_root: Path):
    """Serialize cooperating publishers while keeping a stable lock inode."""
    path = hold_root / ".publication.lock"
    if not path.exists() and not path.is_symlink():
        with admitted_output(hold_root, estimated_bytes=0,
                             stage="snapshot_unknown_raw_publication_lock", inodes=1):
            try:
                with reserved_output_file(path, "xb") as stream:
                    stream.flush()
                    os.fsync(stream.fileno())
            except FileExistsError:
                pass
            _fsync_directory(hold_root)
    before = path.lstat()
    if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid()
            or before.st_nlink != 1 or before.st_mode & 0o077):
        raise ValueError(f"Unsafe unknown-raw publication lock: {path}")
    descriptor = os.open(path, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError("Unknown-raw publication lock changed while opening")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        current = path.lstat()
        if (not stat.S_ISREG(current.st_mode) or current.st_uid != os.getuid()
                or current.st_nlink != 1
                or (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino)):
            raise ValueError("Unknown-raw publication lock changed while acquiring")
        yield
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                         | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_hold_path(path: Path, *, allow_publication_link: bool = False) -> os.stat_result:
    if path.resolve() != path:
        raise ValueError(f"Noncanonical hold path rejected: {path}")
    info = path.lstat()
    valid_links = {1, 2} if allow_publication_link else {1}
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
            or info.st_nlink not in valid_links or info.st_mode & 0o222):
        raise ValueError(f"Unsafe or mutable content-addressed hold: {path}")
    return info


def _hash_regular_file(path: Path, *, expected_size: int | None = None,
                       allow_publication_link: bool = False) -> tuple[str, int]:
    named = path.lstat()
    valid_links = {1, 2} if allow_publication_link else {1}
    if (not stat.S_ISREG(named.st_mode) or named.st_uid != os.getuid()
            or named.st_nlink not in valid_links or named.st_mode & 0o222):
        raise ValueError(f"Unsafe content-addressed hold: {path}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    digest = hashlib.sha256()
    size = 0
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino):
            raise ValueError(f"Hold changed while opening: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            while True:
                block = stream.read(STREAM_CHUNK_BYTES)
                if not block:
                    break
                digest.update(block)
                size += len(block)
        after = os.fstat(descriptor)
        current = path.lstat()
        if ((after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
                != (named.st_dev, named.st_ino, named.st_size, named.st_mtime_ns, named.st_ctime_ns)
                or (current.st_dev, current.st_ino) != (named.st_dev, named.st_ino)):
            raise ValueError(f"Hold changed during verification: {path}")
    finally:
        os.close(descriptor)
    if size != named.st_size or (expected_size is not None and size != expected_size):
        raise ValueError(f"Content-addressed hold size mismatch: {path}")
    return digest.hexdigest(), size


def _verify_existing_hold(destination: Path, digest: str, size: int,
                          *, allow_publication_link: bool = False) -> None:
    _validate_hold_path(destination, allow_publication_link=allow_publication_link)
    actual, actual_size = _hash_regular_file(destination, expected_size=size,
                                              allow_publication_link=allow_publication_link)
    if actual != digest or actual_size != size:
        raise ValueError(f"Content-address collision or tampered hold: {destination}")


def _publish_unresolved(source: Path, expected_generation: tuple[int, ...],
                        digest: str, size: int, hold_root: Path) -> Path:
    destination = hold_root / f"{digest}.gz"
    _owned_directory(hold_root, label="Unknown raw hold root")
    if destination.exists() or destination.is_symlink():
        _verify_existing_hold(destination, digest, size)
        return destination

    _require_storage_authority()

    temporary = hold_root / f".{digest}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    identity: tuple[int, int] | None = None
    try:
        with admitted_output(hold_root, estimated_bytes=size + 4096,
                             stage="snapshot_unknown_raw_copy", inodes=1):
            _owned_directory(hold_root, label="Unknown raw hold root")
            source_fd = _open_source(source, expected_generation)
            digest_copy = hashlib.sha256()
            copied = 0
            try:
                with os.fdopen(source_fd, "rb", closefd=False) as input_stream:
                    with reserved_output_file(temporary, "xb") as output_stream:
                        created = os.fstat(output_stream.fileno())
                        identity = (created.st_dev, created.st_ino)
                        while True:
                            block = input_stream.read(STREAM_CHUNK_BYTES)
                            if not block:
                                break
                            output_stream.write(block)
                            digest_copy.update(block)
                            copied += len(block)
                            _generation(source, source_fd, expected_generation)
                        _generation(source, source_fd, expected_generation)
                        if copied != size or digest_copy.hexdigest() != digest:
                            raise ValueError(f"Source bytes changed while preserving raw input: {source}")
                        os.fchmod(output_stream.fileno(), 0o400)
                        output_stream.flush()
                        os.fsync(output_stream.fileno())
                _generation(source, source_fd, expected_generation)
            finally:
                os.close(source_fd)

            temporary_hash, temporary_size = _hash_regular_file(temporary, expected_size=size)
            if temporary_hash != digest or temporary_size != size:
                raise ValueError(f"Temporary hold verification failed: {temporary}")
            _owned_directory(hold_root, label="Unknown raw hold root")
            try:
                os.link(temporary, destination, follow_symlinks=False)
                _fsync_directory(hold_root)
                temporary_info = temporary.lstat()
                if (identity is None or not stat.S_ISREG(temporary_info.st_mode)
                        or (temporary_info.st_dev, temporary_info.st_ino) != identity
                        or temporary_info.st_uid != os.getuid()):
                    raise ValueError(f"Temporary hold identity changed before unlink: {temporary}")
                temporary.unlink()
                _fsync_directory(hold_root)
            except FileExistsError:
                _verify_existing_hold(destination, digest, size, allow_publication_link=True)
            _verify_existing_hold(destination, digest, size)
    finally:
        if temporary.exists() or temporary.is_symlink():
            info = temporary.lstat()
            if (identity is not None and stat.S_ISREG(info.st_mode)
                    and (info.st_dev, info.st_ino) == identity
                    and info.st_uid == os.getuid()):
                temporary.unlink()
                _fsync_directory(hold_root)
    return destination


def inventory_snapshot_inputs(*, raw_roots: dict[str, str | os.PathLike[str]],
                              output_parent: str | os.PathLike[str],
                              verified_capture_file_sha256: set[str] | frozenset[str]) -> dict:
    """Inventory explicit raw roots and hold every unreferenced compressed file.

    ``verified_capture_file_sha256`` must come from already verified lineage.
    Source and hold hashes cover the exact compressed bytes; gzip content is
    deliberately never decoded, including when the compressed bytes are bad.
    The returned deterministic manifest can be written by the caller into its
    staging snapshot and its byte hash added to that snapshot's manifest.
    """
    if not isinstance(verified_capture_file_sha256, (set, frozenset)):
        raise ValueError("Verified capture hashes must be supplied as a set")
    known_hashes = set(verified_capture_file_sha256)
    if any(not isinstance(value, str) or not _SHA256.fullmatch(value) for value in known_hashes):
        raise ValueError("Invalid verified capture_file_sha256 value")

    output = _canonical(output_parent)
    _owned_directory(output, label="Snapshot output parent")
    roots, sources = _scan_roots(raw_roots)
    hold_root = output / "unknown-raw"
    for root in (Path(row["path"]) for row in roots):
        if (hold_root == root or hold_root.is_relative_to(root)
                or root.is_relative_to(hold_root)):
            raise ValueError("Unknown raw destination must not overlap an input raw root")

    records = []
    pending_holds = []
    for source_row in sources:
        source = Path(source_row["path"])
        generation = tuple(source_row.pop("generation"))
        digest, size = _hash_source(source, generation)
        status = "content_referenced" if digest in known_hashes else "unresolved_protected_raw"
        hold_path = None
        reason = "capture_file_sha256_matches_verified_lineage"
        if status == "unresolved_protected_raw":
            pending_holds.append((len(records), source, generation, digest, size))
            reason = "descriptor_missing_or_not_resolved"
        records.append({
            "root_id": source_row["root_id"],
            "path": source_row["path"],
            "relative_path": source_row["relative_path"],
            "size_bytes": size,
            "sha256": digest,
            "status": status,
            "held_path": hold_path,
            "training_eligible": None if status == "content_referenced" else False,
            "deletion_eligible": False,
            "reason": reason,
            "source_generation": {
                "device": generation[0], "inode": generation[1], "uid": generation[2],
                "mode": generation[3], "link_count": generation[4],
                "size_bytes": generation[5], "mtime_ns": generation[6], "ctime_ns": generation[7]
            }
        })

    if pending_holds:
        hold_root = _validate_hold_root(output, create=True)
        with _publication_lock(hold_root):
            for index, source, generation, digest, size in pending_holds:
                held = _publish_unresolved(source, generation, digest, size, hold_root)
                records[index]["held_path"] = str(held)

    manifest = {"schema": SCHEMA, "hash_scope": "exact_compressed_source_bytes",
                "reference_source": "caller_supplied_verified_lineage_capture_file_sha256",
                "roots": roots, "files": records}
    encoded = inventory_manifest_bytes(manifest)
    manifest["inventory_sha256"] = hashlib.sha256(encoded).hexdigest()
    return manifest


def inventory_manifest_bytes(manifest: dict) -> bytes:
    """Serialize the inventory deterministically for immutable snapshot storage."""
    payload = {key: value for key, value in manifest.items() if key != "inventory_sha256"}
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
