"""Read-only discovery and registration of native preparation-cache artifacts.

Callers must supply every cache root explicitly. This module never removes or
rewrites cache payloads; it only imports qualified measurements into the
existing storage authority's cache inventory.
"""
from __future__ import annotations

import fcntl
import math
import os
from pathlib import Path
import re
import stat

from . import research_preparation as preparation
from . import research_storage as storage


_ARTIFACT_ID = re.compile(r"[0-9a-f]{64}\Z")
_ROOT_METADATA = re.compile(r"([0-9a-f]{64})(\.pin\.lock|\.lock|\.owner\.json)\Z")
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_UNKNOWN_MAX_DEPTH = 16
_UNKNOWN_MAX_NODES = 4096


def _canonical_roots(authority, roots):
    if isinstance(roots, (str, bytes, os.PathLike)):
        raise TypeError("qualified_roots must be an explicit sequence of root paths")
    try:
        supplied = tuple(roots)
    except TypeError as exc:
        raise TypeError("qualified_roots must be an explicit sequence of root paths") from exc
    if not supplied:
        raise ValueError("At least one explicitly qualified cache root is required")

    authority_info = os.stat(authority.root, follow_symlinks=False)
    if not stat.S_ISDIR(authority_info.st_mode) or authority_info.st_uid != os.getuid():
        raise ValueError("Storage authority root must be an owned real directory")
    roots_out = []
    identities = set()
    for value in supplied:
        root = Path(value).absolute()
        if ".." in Path(value).parts or root.resolve(strict=True) != root:
            raise ValueError(f"Cache root is not canonical: {root}")
        info = os.stat(root, follow_symlinks=False)
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
                or info.st_mode & 0o022 or info.st_dev != authority_info.st_dev):
            raise ValueError(f"Cache root is foreign, shared, or on another filesystem: {root}")
        identity = (info.st_dev, info.st_ino)
        if identity in identities:
            raise ValueError(f"Duplicate qualified cache root: {root}")
        identities.add(identity)
        roots_out.append((root, identity))

    paths = [root for root, _ in roots_out]
    for index, root in enumerate(paths):
        for other in paths[index + 1:]:
            if root in other.parents or other in root.parents:
                raise ValueError("Nested qualified cache roots would double count artifacts")
    return roots_out


def _signature(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_uid,
            info.st_size, getattr(info, "st_blocks", 0), info.st_mtime_ns,
            info.st_ctime_ns)


def _measure_unknown(parent_fd, name, device, owner, *, max_depth=_UNKNOWN_MAX_DEPTH,
                     max_nodes=_UNKNOWN_MAX_NODES):
    """Conservatively measure an unqualified tree without following any link."""
    total_bytes = 0
    total_inodes = 0
    complete = True
    try:
        info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError:
        return {"allocated_bytes": None, "inode_count": None, "complete": False}

    visited = 0

    def add(info, depth):
        nonlocal visited
        nonlocal total_bytes, total_inodes, complete
        total_bytes += getattr(info, "st_blocks", 0) * 512
        total_inodes += 1
        visited += 1
        if info.st_uid != owner or info.st_dev != device:
            complete = False
        if depth > max_depth or visited > max_nodes:
            complete = False
            return False
        return info.st_uid == owner and info.st_dev == device

    if not add(info, 0):
        return {"allocated_bytes": total_bytes, "inode_count": total_inodes,
                "complete": False}
    if not stat.S_ISDIR(info.st_mode):
        return {"allocated_bytes": total_bytes, "inode_count": total_inodes,
                "complete": complete}
    try:
        directory_fd = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
    except OSError:
        return {"allocated_bytes": total_bytes, "inode_count": total_inodes,
                "complete": False}
    try:
        bound = os.fstat(directory_fd)
        if (bound.st_dev, bound.st_ino) != (info.st_dev, info.st_ino):
            return {"allocated_bytes": total_bytes, "inode_count": total_inodes,
                    "complete": False}
        # Keep directory descriptors bounded and close each before descending.
        def walk(fd, depth):
            nonlocal complete
            for child in os.listdir(fd):
                if visited >= max_nodes:
                    complete = False
                    return
                try:
                    child_info = os.stat(child, dir_fd=fd, follow_symlinks=False)
                except OSError:
                    complete = False
                    continue
                descend = add(child_info, depth + 1)
                if stat.S_ISDIR(child_info.st_mode) and descend:
                    if depth + 1 >= max_depth:
                        complete = False
                        continue
                    try:
                        nested_fd = os.open(child, _DIRECTORY_FLAGS, dir_fd=fd)
                    except OSError:
                        complete = False
                        continue
                    try:
                        opened = os.fstat(nested_fd)
                        if (opened.st_dev, opened.st_ino) != (child_info.st_dev, child_info.st_ino):
                            complete = False
                        elif opened.st_uid != owner or opened.st_dev != device:
                            complete = False
                        else:
                            walk(nested_fd, depth + 1)
                    finally:
                        os.close(nested_fd)
        walk(directory_fd, 0)
    except OSError:
        complete = False
    finally:
        os.close(directory_fd)
    return {"allocated_bytes": total_bytes,
            "inode_count": total_inodes,
            "complete": complete}


def _entry_stats(directory_fd, identity):
    entries = {}
    for name in os.listdir(directory_fd):
        entries[name] = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    directory_info = os.fstat(directory_fd)
    return entries, directory_info


def _validate_root_metadata(root_fd, name, identity, device):
    """Accept only preparation's stable lock and owner sidecar formats."""
    match = _ROOT_METADATA.fullmatch(name)
    if not match:
        return False, "unknown_root_entry"
    key, suffix = match.groups()
    try:
        info = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or info.st_uid != os.getuid() or info.st_dev != device):
            return False, "unsafe_root_metadata"
        if suffix in {'.lock','.pin.lock'} and info.st_size != 0:
            return False, "unexpected_lock_payload"
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=root_fd)
        try:
            opened = os.fstat(fd)
            if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
                return False, "changed_root_metadata"
            if suffix == ".owner.json":
                import json
                payload = json.loads(os.read(fd, min(info.st_size + 1, 1024 * 1024)))
                if (key != identity or not isinstance(payload, dict)
                        or payload.get("cache_key") != identity
                        or not isinstance(payload.get("owner"), dict)
                        or not isinstance(payload.get("stage"), str)):
                    return False, "invalid_owner_metadata"
        finally:
            os.close(fd)
    except (OSError, ValueError, TypeError):
        return False, "invalid_root_metadata"
    return True, None


def _is_live_pin(root_fd, identity, device):
    """Observe the preparation lock without creating or changing a lock file."""
    name = identity + ".lock"
    try:
        info = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False, None
    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
            or info.st_uid != os.getuid() or info.st_dev != device):
        return False, "unsafe_lock_entry"
    try:
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=root_fd)
    except OSError:
        return False, "unsafe_lock_entry"
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
            return False, "changed_lock_entry"
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True, None
        else:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            return False, None
    finally:
        os.close(descriptor)


def _inspect_artifact(root, root_fd, identity, root_identity):
    root_info = os.fstat(root_fd)
    try:
        # The shared helper returns both descriptors; retain both until the
        # final descriptor-bound qualification has completed.
        bound_root_fd, artifact_fd, observed_root, artifact_identity = (
            preparation._open_cache_directory(root, identity))
    except (OSError, ValueError) as exc:
        return None, {"reason": "linked_or_unopenable_artifact", "detail": str(exc)}
    try:
        if observed_root != root_identity or os.fstat(bound_root_fd).st_ino != root_info.st_ino:
            return None, {"reason": "cache_root_changed"}
        before_entries, before_directory = _entry_stats(artifact_fd, identity)
        qualified = preparation._qualify_cache_directory(
            root, bound_root_fd, artifact_fd, identity, observed_root, artifact_identity)
        if qualified is None:
            return None, {"reason": "unknown_or_linked_contents"}

        allocated = getattr(before_directory, "st_blocks", 0) * 512
        inodes = 1
        mtimes = [before_directory.st_mtime_ns]
        atimes = [before_directory.st_atime_ns]
        signature = {}
        for name, inode in qualified.items():
            info = before_entries.get(name)
            if info is None or (info.st_dev, info.st_ino) != inode:
                return None, {"reason": "artifact_changed_during_inventory"}
            allocated += getattr(info, "st_blocks", 0) * 512
            inodes += 1
            mtimes.append(info.st_mtime_ns)
            atimes.append(info.st_atime_ns)
            signature[name] = _signature(info)

        # Qualification reads only the manifest. Recheck names, inode identity,
        # size, allocation, mtime, and ctime without rereading it (which could
        # itself update atime on filesystems without O_NOATIME support).
        after_entries, after_directory = _entry_stats(artifact_fd, identity)
        if (_signature(after_directory) != _signature(before_directory)
                or set(after_entries) != set(before_entries)
                or any(_signature(after_entries[name]) != signature[name]
                       for name in signature)):
            return None, {"reason": "artifact_changed_during_inventory"}

        pinned, pin_error = _is_live_pin(bound_root_fd, identity, root_info.st_dev)
        if pin_error:
            return None, {"reason": pin_error}
        return {
            "allocated_bytes": allocated,
            "inode_count": inodes,
            "max_mtime_ns": max(mtimes),
            "max_atime_ns": max(atimes),
            "pinned": pinned,
        }, None
    finally:
        os.close(artifact_fd)
        os.close(bound_root_fd)


def import_cache_inventory(authority, qualified_roots, *, protected_paths=()):
    """Import every qualified native artifact from explicit roots under one lock.

    Inventory completeness is separate from quota capacity. A complete inventory
    may exceed the configured quota; the authority's normal growth admission then
    blocks additional cache growth. Any held, protected, or incomplete observation
    prevents all authority-state writes. Callers must quiesce legacy preparation
    writers that do not use the storage authority lock; that lock serializes this
    scan with authority eviction, but cannot prove arbitrary-writer quiescence.
    """
    roots = _canonical_roots(authority, qualified_roots)
    if isinstance(protected_paths, (str, bytes, os.PathLike)):
        raise TypeError("protected_paths must be an explicit sequence of paths")
    protected = tuple(str(Path(value).absolute()) for value in protected_paths)
    if len(set(protected)) != len(protected):
        raise ValueError("Duplicate protected paths")

    discovered = []
    held = []
    held_bytes = 0
    held_inodes = 0
    held_complete = True
    pinned = []
    protected_native = set(protected)
    with authority._locked() as state:
        caches = state.get("caches")
        if not isinstance(caches, dict):
            raise ValueError("Storage authority cache inventory is malformed")

        def hold(path, reason, measurement=None):
            nonlocal held_bytes, held_inodes, held_complete
            measurement = measurement or {
                "allocated_bytes": None, "inode_count": None, "complete": False}
            held_bytes += measurement["allocated_bytes"] or 0
            held_inodes += measurement["inode_count"] or 0
            held_complete = held_complete and measurement["complete"]
            held.append({"path": str(path), "reason": reason, **measurement})

        for root, expected_identity in roots:
            try:
                root_fd = os.open(root, _DIRECTORY_FLAGS)
            except OSError as exc:
                hold(root, "root_unopenable", {"allocated_bytes": None,
                     "inode_count": None, "complete": False, "detail": str(exc)})
                continue
            try:
                root_info = os.fstat(root_fd)
                path_info = os.stat(root, follow_symlinks=False)
                if ((root_info.st_dev, root_info.st_ino) != expected_identity
                        or (path_info.st_dev, path_info.st_ino) != expected_identity):
                    hold(root, "cache_root_changed")
                    continue
                names_before = sorted(os.listdir(root_fd))
                entries_before = {
                    name: _signature(os.stat(name, dir_fd=root_fd, follow_symlinks=False))
                    for name in names_before
                }
                directory_before = _signature(root_info)
                for name in names_before:
                    if _ARTIFACT_ID.fullmatch(name):
                        item, error = _inspect_artifact(root, root_fd, name, expected_identity)
                        if error is None:
                            item.update({"root": str(root), "artifact_id": name,
                                         "path": str(root / name)})
                            discovered.append(item)
                            if item["pinned"]:
                                pinned.append(str(root / name))
                            continue
                        measurement = _measure_unknown(
                            root_fd, name, root_info.st_dev, os.getuid())
                        hold(root / name, error["reason"], measurement)
                        continue

                    metadata = _ROOT_METADATA.fullmatch(name)
                    if metadata:
                        identity = metadata.group(1)
                        valid, reason = _validate_root_metadata(
                            root_fd, name, identity, root_info.st_dev)
                        if valid:
                            continue
                        measurement = _measure_unknown(
                            root_fd, name, root_info.st_dev, os.getuid())
                        hold(root / name, reason, measurement)
                        continue

                    measurement = _measure_unknown(
                        root_fd, name, root_info.st_dev, os.getuid())
                    hold(root / name, "unknown_root_entry", measurement)

                try:
                    names_after = sorted(os.listdir(root_fd))
                    entries_after = {
                        name: _signature(os.stat(name, dir_fd=root_fd, follow_symlinks=False))
                        for name in names_after
                    }
                    directory_after = _signature(os.fstat(root_fd))
                    path_after = os.stat(root, follow_symlinks=False)
                except OSError:
                    names_after, entries_after, directory_after, path_after = [], {}, None, None
                if (names_after != names_before or entries_after != entries_before
                        or directory_after != directory_before or path_after is None
                        or (path_after.st_dev, path_after.st_ino) != expected_identity):
                    hold(root, "cache_root_changed_during_inventory")
            except OSError as exc:
                hold(root, "cache_root_incomplete", {"allocated_bytes": None,
                     "inode_count": None, "complete": False, "detail": str(exc)})
            finally:
                os.close(root_fd)

        protected_reports = [
            {"path": path, "reason": "explicit_protected_path", "allocated_bytes": None,
             "inode_count": None, "complete": False}
            for path in protected
        ]
        protected_native_matches = [item["path"] for item in discovered
                                    if item["path"] in protected_native]
        for path in protected_native_matches:
            hold(path, "explicit_protected_native_artifact")

        qualified_count = len(discovered)
        ready = not held and held_complete and not protected
        if not ready:
            cache_total = sum(record.get("allocated_bytes", 0)
                              for record in caches.values()
                              if isinstance(record, dict)
                              and type(record.get("allocated_bytes", 0)) is int
                              and record.get("allocated_bytes", 0) >= 0)
            return {
                "roots": [str(root) for root, _ in roots],
                "qualified_artifact_count": qualified_count,
                "qualified_artifacts": [item["path"] for item in discovered],
                "registered_artifact_count": 0,
                "observed_allocated_bytes": sum(item["allocated_bytes"] for item in discovered),
                "registered_inventory_bytes": 0,
                "inode_count": sum(item["inode_count"] for item in discovered),
                "max_mtime_ns": max((item["max_mtime_ns"] for item in discovered), default=None),
                "max_atime_ns": max((item["max_atime_ns"] for item in discovered), default=None),
                "pinned_artifacts": pinned,
                "held_artifacts": held,
                "held_paths": held + protected_reports,
                "held_allocated_bytes": held_bytes,
                "held_inode_count": held_inodes,
                "held_accounting_complete": held_complete and not protected,
                "protected_paths": list(protected),
                "cache_total_bytes": cache_total,
                "cache_limit_bytes": authority.cache_limit_bytes,
                "quota_over_limit": cache_total > authority.cache_limit_bytes,
                "admission_ready": False,
            }

        for item in discovered:
            path = item["path"]
            prior = caches.get(path, {})
            if prior and (prior.get("root") != item["root"]
                          or prior.get("artifact_id") != item["artifact_id"]):
                raise ValueError(f"Storage authority identity mismatch for {path}")
            measured = item["allocated_bytes"]
            prior_bytes = prior.get("allocated_bytes", 0)
            if type(prior_bytes) is not int or prior_bytes < 0:
                raise ValueError(f"Storage authority byte measurement is malformed for {path}")
            last_access = prior.get("last_access", item["max_atime_ns"] / 1e9)
            if (isinstance(last_access, bool) or not isinstance(last_access, (int, float))
                    or not math.isfinite(last_access) or last_access < 0):
                raise ValueError(f"Storage authority access time is malformed for {path}")
            record = {
                "root": item["root"],
                "artifact_id": item["artifact_id"],
                # A scan is not a verified refresh. Keep the previous high-water
                # measurement until a separate verified refresh protocol exists.
                "allocated_bytes": max(prior_bytes, measured),
                "last_access": last_access,
                "inventory_inode_count": item["inode_count"],
                "inventory_max_mtime_ns": item["max_mtime_ns"],
                "inventory_max_atime_ns": item["max_atime_ns"],
                "inventory_pinned": item["pinned"],
            }
            record["inventory_observed_allocated_bytes"] = measured
            caches[path] = record
        storage._atomic_json(authority.root / "state.json", state)
        cache_total = sum(record["allocated_bytes"] for record in caches.values())

    return {
        "roots": [str(root) for root, _ in roots],
        "qualified_artifact_count": len(discovered),
        "qualified_artifacts": [item["path"] for item in discovered],
        "registered_artifact_count": len(discovered),
        "observed_allocated_bytes": sum(item["allocated_bytes"] for item in discovered),
        "registered_inventory_bytes": sum(
            max(0, caches[item["path"]]["allocated_bytes"])
            for item in discovered),
        "inode_count": sum(item["inode_count"] for item in discovered),
        "max_mtime_ns": max((item["max_mtime_ns"] for item in discovered), default=None),
        "max_atime_ns": max((item["max_atime_ns"] for item in discovered), default=None),
        "pinned_artifacts": pinned,
        "held_artifacts": held,
        "held_paths": held + protected_reports,
        "held_allocated_bytes": held_bytes,
        "held_inode_count": held_inodes,
        "held_accounting_complete": held_complete and not protected,
        "protected_paths": list(protected),
        "cache_total_bytes": cache_total,
        "cache_limit_bytes": authority.cache_limit_bytes,
        "quota_over_limit": cache_total > authority.cache_limit_bytes,
        "admission_ready": True,
    }
