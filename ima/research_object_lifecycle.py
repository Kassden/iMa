"""Fail-closed lifecycle planning and verified copies for immutable objects.

No deletion operation is exposed. Plans are read-only review material.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl, hashlib, json, os, shutil, stat, time
import threading
from pathlib import Path
from .research_objects import ObjectStore, read_snapshot_table

GRACE_SECONDS = 7 * 86400
SNAPSHOT_FORMAT = "ima.snapshot.objects.v2"
TABLE_FORMAT = "parquet-zstd-v1"
OPAQUE_ROOT_KINDS = frozenset({"mlflow", "hypothesis", "confirmation", "pending",
                               "models", "raw", "unique_history"})
_LEASES = threading.local()

class LifecycleHold(RuntimeError):
    """Lifecycle evidence is incomplete, unreadable, or unsupported."""

def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()

def _fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try: os.fsync(fd)
    finally: os.close(fd)

def _lock_path(object_root):
    root = Path(object_root).absolute()
    if root.resolve() != root: raise ValueError("Object root must be canonical")
    return root.parent / ".ima-object-lifecycle.lock"

@contextmanager
def lifecycle_lease(object_root, *, exclusive=False):
    """Stable flock lease; process death releases it without stale lease records."""
    path = _lock_path(object_root)
    key = str(path)
    held = getattr(_LEASES, "held", {})
    if key in held:
        if exclusive and not held[key]["exclusive"]:
            raise RuntimeError("Cannot upgrade a shared lifecycle lease")
        held[key]["depth"] += 1
        try: yield
        finally: held[key]["depth"] -= 1
        return
    if not path.parent.is_dir() or path.parent.resolve() != path.parent:
        raise ValueError("Lifecycle lock parent must be a canonical existing directory")
    fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(fd, "a+b") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError("Unsafe lifecycle lock inode")
        if path.stat().st_ino != info.st_ino: raise ValueError("Lifecycle lock inode changed")
        fcntl.flock(stream, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        held = dict(getattr(_LEASES, "held", {}))
        held[key] = {"exclusive": exclusive, "depth": 1}
        _LEASES.held = held
        try:
            if path.stat().st_ino != info.st_ino: raise ValueError("Lifecycle lock inode replaced")
            yield
        finally:
            held = dict(getattr(_LEASES, "held", {}))
            held.pop(key, None)
            _LEASES.held = held
            fcntl.flock(stream, fcntl.LOCK_UN)

def object_publication_lease(object_root):
    """Hold shared lease from first object link through durable snapshot publish."""
    return lifecycle_lease(object_root)

def object_read_lease(object_root):
    """Hold shared lease for the actual Parquet read."""
    return lifecycle_lease(object_root)

def object_pin_lease(object_root):
    return lifecycle_lease(object_root)

@dataclass(frozen=True)
class ProtectedRoot:
    kind: str
    path: Path
    def __post_init__(self):
        path = Path(self.path).absolute()
        if self.kind not in OPAQUE_ROOT_KINDS: raise ValueError(f"Unsupported protected-root kind: {self.kind}")
        if path != Path(self.path) or path.resolve() != path: raise ValueError(f"Protected root is not canonical: {path}")

def _validate_ref(ref, object_root):
    if not isinstance(ref, dict) or set(ref) != {"sha256", "size", "format"}:
        raise LifecycleHold("Unknown object-reference shape")
    digest, size, fmt = ref["sha256"], ref["size"], ref["format"]
    if (not isinstance(digest, str) or len(digest) != 64 or set(digest) - set("0123456789abcdef")
            or type(size) is not int or size < 0 or fmt != TABLE_FORMAT):
        raise LifecycleHold("Unsupported object reference")
    shard = Path(object_root) / digest[:2]
    path = shard / digest
    try:
        if shard.is_symlink() or shard.resolve() != shard:
            raise LifecycleHold(f"Noncanonical object shard: {shard}")
        info = path.lstat()
        valid = stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid() and info.st_size == size and _sha256(path) == digest
    except OSError as exc: raise LifecycleHold(f"Referenced object unreadable: {digest}") from exc
    if not valid: raise LifecycleHold(f"Referenced object identity mismatch: {digest}")
    return digest

def _snapshot_refs(snapshot, object_root):
    try: manifest = json.loads((snapshot / "manifest.json").read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc: raise LifecycleHold(f"Unreadable snapshot manifest: {snapshot}") from exc
    if not isinstance(manifest, dict): raise LifecycleHold(f"Invalid snapshot manifest: {snapshot}")
    fmt = manifest.get("storage_format")
    if fmt is None: return set()  # legacy snapshots own local Parquet
    if fmt != SNAPSHOT_FORMAT: raise LifecycleHold(f"Unsupported snapshot format: {fmt!r}")
    tables = manifest.get("tables")
    if not isinstance(tables, dict) or not tables: raise LifecycleHold("Invalid objects-v2 tables")
    refs = set()
    for name, table in tables.items():
        if (not isinstance(name, str) or Path(name).name != name or not isinstance(table, dict)
                or set(table) != {"schema_version", "object_root", "rows", "columns", "ordinal", "partitions"}
                or table.get("schema_version") != 1 or Path(table.get("object_root", "")).absolute() != object_root
                or not isinstance(table.get("columns"), list) or type(table.get("rows")) is not int
                or not isinstance(table.get("ordinal"), str) or not isinstance(table.get("partitions"), list)):
            raise LifecycleHold(f"Unsupported objects-v2 table: {name!r}")
        if table["rows"] > 0 and not table["partitions"]: raise LifecycleHold("Nonempty table has no partitions")
        for ref in table["partitions"]: refs.add(_validate_ref(ref, object_root))
    return refs

def _snapshot_dirs(root):
    if (root / "manifest.json").is_file():
        yield root
        return
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "manifest.json").exists(): yield child
        elif child.is_dir() and child.name.endswith(".building"): raise LifecycleHold(f"Incomplete snapshot publication: {child}")
        else: raise LifecycleHold(f"Unrecognized snapshot-root entry: {child}")

def _registry_refs(registry, object_root):
    datasets = registry / "datasets"
    if not datasets.is_dir(): raise LifecycleHold(f"DatasetRegistry datasets missing: {datasets}")
    manifests, refs = {}, set()
    for child in sorted(datasets.iterdir()):
        if not child.is_dir(): raise LifecycleHold(f"Unexpected DatasetRegistry entry: {child}")
        try: body = json.loads((child / "manifest.json").read_text())
        except (OSError, UnicodeError, json.JSONDecodeError) as exc: raise LifecycleHold(f"Unreadable DatasetRegistry manifest: {child}") from exc
        identity = body.get("dataset_id") if isinstance(body, dict) else None
        if not isinstance(identity, str) or identity in manifests: raise LifecycleHold(f"Invalid DatasetRegistry identity: {child}")
        manifests[identity] = body
        source = body.get("source_snapshot")
        if not isinstance(source, str): raise LifecycleHold(f"Missing source_snapshot for {identity}")
        snapshot = Path(source).absolute()
        if snapshot.resolve() != snapshot or not snapshot.is_dir(): raise LifecycleHold(f"Unavailable source_snapshot for {identity}")
        refs.update(_snapshot_refs(snapshot, object_root))
    for identity, body in manifests.items():
        seen, parent = {identity}, body.get("parent_dataset_id")
        while parent is not None:
            if not isinstance(parent, str) or parent in seen or parent not in manifests: raise LifecycleHold(f"Broken DatasetRegistry parent chain: {identity}")
            seen.add(parent)
            parent = manifests[parent].get("parent_dataset_id")
    return refs

def plan_retention(*, object_root, snapshot_roots=(), dataset_registry_roots=(), protected_roots=(), now=None, grace_seconds=GRACE_SECONDS):
    """Typed read-only mark plan. Opaque research roots hold all shared-object sweep."""
    root = Path(object_root).absolute()
    if root.resolve() != root or not root.is_dir(): raise LifecycleHold("Existing canonical object root required")
    if type(grace_seconds) is not int or grace_seconds < 0: raise ValueError("Invalid grace period")
    roots = list(protected_roots)
    by_kind = {entry.kind: entry for entry in roots}
    if len(by_kind) != len(roots) or set(by_kind) != set(OPAQUE_ROOT_KINDS):
        raise LifecycleHold("Explicit MLflow, hypothesis, confirmation, pending, model, raw, and unique-history roots required")
    for entry in roots:
        if not entry.path.exists() or entry.path.is_symlink(): raise LifecycleHold(f"Protected {entry.kind} root missing or symlinked")
    now = time.time() if now is None else now
    if type(now) not in (int, float): raise ValueError("Invalid planner clock")
    with lifecycle_lease(root, exclusive=True):
        marked = set()
        for location in snapshot_roots:
            snapshots = Path(location).absolute()
            if not snapshots.is_dir() or snapshots.resolve() != snapshots: raise LifecycleHold(f"Snapshot root unavailable: {snapshots}")
            for snapshot in _snapshot_dirs(snapshots): marked.update(_snapshot_refs(snapshot, root))
        for location in dataset_registry_roots:
            registry = Path(location).absolute()
            if not registry.is_dir() or registry.resolve() != registry: raise LifecycleHold(f"DatasetRegistry root unavailable: {registry}")
            marked.update(_registry_refs(registry, root))
        rows = []
        for shard in sorted(root.iterdir()):
            if shard.name.startswith("."):
                if shard.name.endswith(".building") or shard.name.startswith(".parquet-"): raise LifecycleHold(f"Incomplete object publication: {shard}")
                raise LifecycleHold(f"Unknown object-store entry: {shard}")
            if (len(shard.name) != 2 or set(shard.name) - set("0123456789abcdef")
                    or shard.is_symlink() or shard.resolve() != shard or not shard.is_dir()):
                raise LifecycleHold(f"Unknown object-store entry: {shard}")
            for path in sorted(shard.iterdir()):
                digest = path.name
                if (len(digest) != 64 or digest[:2] != shard.name or set(digest) - set("0123456789abcdef")
                        or path.is_symlink() or not path.is_file() or path.stat().st_uid != os.getuid()):
                    raise LifecycleHold(f"Unknown object payload: {path}")
                size, verified = path.stat().st_size, _sha256(path) == digest
                age = max(0.0, now - path.stat().st_mtime)
                reasons = []
                if not verified: reasons.append("corrupt_identity")
                if digest in marked: reasons.append("typed_reference")
                if age < grace_seconds: reasons.append("initial_grace")
                if roots: reasons.append("opaque_protected_roots")
                rows.append({"sha256": digest, "size": size, "verified": verified, "referenced": digest in marked,
                             "age_seconds": age, "hold_reasons": reasons, "status": "held"})
        return {"schema_version": 1, "mode": "read_only_dry_run", "approved": False, "object_root": str(root),
                "reference_count": len(marked), "objects": rows, "review_eligible": [],
                "holds": ([f"{kind}: protected root is opaque; shared reference adapter not qualified" for kind in sorted(by_kind)]
                          + ["archive: no explicit external destination supplied"]),
                "deletion_performed": False, "phase_complete": False}

def _checked_ref(ref):
    if (not isinstance(ref, dict) or set(ref) != {"sha256", "size", "format"} or not isinstance(ref["sha256"], str)
            or len(ref["sha256"]) != 64 or set(ref["sha256"]) - set("0123456789abcdef")
            or type(ref["size"]) is not int or ref["size"] < 0 or ref["format"] != TABLE_FORMAT):
        raise ValueError("Unsupported immutable object reference")
    return dict(ref)

def archive_object(*, object_root, reference, destination, authority):
    """Checksum-verified no-replace copy; originals are never removed."""
    ref, store = _checked_ref(reference), ObjectStore(object_root)
    dest = Path(destination).absolute()
    if dest.resolve() != dest or not dest.parent.is_dir(): raise ValueError("Explicit destination with existing canonical parent required")
    with lifecycle_lease(store.root):
        source = store.path(ref)
        with authority.admitted(bytes_required=source.stat().st_size + 128 * 1024, inodes_required=1,
                                stage="research_object_archive", output=dest):
            temporary = dest.with_name(f".{dest.name}.{os.urandom(8).hex()}.tmp")
            try:
                with source.open("rb") as src, temporary.open("xb") as dst:
                    shutil.copyfileobj(src, dst, 1024 * 1024); dst.flush(); os.fsync(dst.fileno())
                if temporary.stat().st_size != ref["size"] or _sha256(temporary) != ref["sha256"]: raise ValueError("Archive copy checksum mismatch")
                os.chmod(temporary, 0o400)
                try: os.link(temporary, dest)
                except FileExistsError as exc: raise ValueError("Archive destination already exists") from exc
                _fsync_dir(dest.parent)
            finally: temporary.unlink(missing_ok=True)
    same_fs = source.stat().st_dev == dest.stat().st_dev
    return {"reference": ref, "archive_path": str(dest), "sha256": _sha256(dest), "size": dest.stat().st_size,
            "same_filesystem": same_fs, "reclaimed_bytes": 0, "source_deleted": False, "status": "verified_copy_only"}

def restore_object(*, archive_path, object_root, reference, authority):
    """Restore a verified object without overwriting an existing identity."""
    ref, archive = _checked_ref(reference), Path(archive_path).absolute()
    if archive.resolve() != archive or not archive.is_file() or archive.is_symlink(): raise ValueError("Explicit canonical archive file required")
    if archive.stat().st_size != ref["size"] or _sha256(archive) != ref["sha256"]: raise ValueError("Archive checksum mismatch")
    root, store = Path(object_root).absolute(), ObjectStore(object_root)
    dest = root / ref["sha256"][:2] / ref["sha256"]
    if (not dest.parent.is_dir() or dest.parent.is_symlink()
            or dest.parent.resolve() != dest.parent):
        raise ValueError("Restore shard must already exist at its canonical path")
    with lifecycle_lease(root):
        # Pin the shard before admission or writes. All writes and cleanup use
        # this descriptor, so swapping a path for a symlink cannot redirect them.
        shard_fd = os.open(dest.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            info = os.fstat(shard_fd)
            if info.st_uid != os.getuid() or info.st_mode & 0o022:
                raise ValueError("Restore shard must be owned and non-shared")
            with authority.admitted(bytes_required=ref["size"] + 128 * 1024, inodes_required=1,
                                    stage="research_object_restore", output=dest):
                temporary = f".{dest.name}.{os.urandom(8).hex()}.tmp"
                _restore_pinned(archive, ref, dest, temporary, shard_fd)
        finally:
            os.close(shard_fd)
        store.path(ref)
    return {"reference": ref, "object_path": str(dest), "sha256": _sha256(dest), "size": dest.stat().st_size,
            "status": "restored_verified", "consumer_readback": "required via read_snapshot_table"}

def _restore_pinned(archive, ref, dest, temporary, shard_fd):
    created = False
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o600, dir_fd=shard_fd)
        created = True
        with archive.open("rb") as src, os.fdopen(descriptor, "wb") as dst:
            digest = hashlib.sha256()
            size = 0
            for chunk in iter(lambda: src.read(1024 * 1024), b""):
                dst.write(chunk); digest.update(chunk); size += len(chunk)
            dst.flush()
            if size != ref["size"] or digest.hexdigest() != ref["sha256"]:
                raise ValueError("Restored object checksum mismatch")
            os.fchmod(dst.fileno(), 0o400)
            os.fsync(dst.fileno())
        try:
            os.link(temporary, dest.name, src_dir_fd=shard_fd, dst_dir_fd=shard_fd,
                    follow_symlinks=False)
        except FileExistsError as exc:
            raise ValueError("Restore destination already exists") from exc
        os.fsync(shard_fd)
    finally:
        if created:
            try:
                os.unlink(temporary, dir_fd=shard_fd)
            except FileNotFoundError:
                pass

def verify_restored_snapshot(snapshot, table_name="runners.parquet"):
    """Prove restore through the actual snapshot reader."""
    frame = read_snapshot_table(snapshot, table_name)
    return {"snapshot": str(Path(snapshot).absolute()), "table": table_name, "rows": len(frame),
            "columns": list(frame.columns), "consumer_readback": True}
