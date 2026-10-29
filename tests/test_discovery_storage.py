import errno
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import ima.research_storage as storage
import ima.feature_program as feature_program
from ima.feature_discovery_specs import DiscoverySpecV2
from ima.feature_program import materialize, synthesize


def discovery_fixture():
    rows = []
    for day in range(1, 4):
        for horse in range(2):
            rows.append(dict(
                race_id=f"r{day}", horse_no=horse + 1, horse_id=f"h{horse}",
                jockey_id=f"j{horse}", trainer_id="t1", date=f"2020-01-0{day}",
                finish_seconds=60 + horse + day, distance=1200, lengths_raw=horse,
                actual_weight=120 + horse,
            ))
    return pd.DataFrame(rows)


def enabled_authority(tmp_path, monkeypatch, available=64 * 1024**2):
    authority = storage.StorageAuthority(
        tmp_path / "control", headroom_bytes=0, headroom_inodes=0,
        statvfs=lambda path: SimpleNamespace(
            f_frsize=1, f_blocks=available, f_bavail=available,
            f_bfree=available, f_favail=100_000,
        ),
    )
    monkeypatch.setenv("IMA_STORAGE_ROOT", str(authority.root))
    monkeypatch.setattr(storage, "StorageAuthority", lambda root: authority)
    return authority


@pytest.mark.parametrize("shared", [False, True])
def test_discovery_control_admission_precedes_cache_creation_and_synthesis(
    tmp_path, monkeypatch, shared
):
    authority = enabled_authority(tmp_path, monkeypatch)
    with patch.object(authority, "reserve", side_effect=storage.StorageBlocked("blocked")):
        with patch("ima.feature_program.synthesize") as build:
            with pytest.raises(storage.StorageBlocked):
                materialize(
                    discovery_fixture(), DiscoverySpecV2(windows_days=(90,)),
                    "source", tmp_path / "discovery", shared=shared,
                )
        build.assert_not_called()
    assert not (tmp_path / "discovery").exists()


@pytest.mark.parametrize("shared", [False, True])
def test_matrix_admission_precedes_synthesis_and_native_export(tmp_path, monkeypatch, shared):
    authority = enabled_authority(tmp_path, monkeypatch)
    original_reserve = authority.reserve
    stages = []

    def deny_matrix_stage(**kwargs):
        stages.append(kwargs["stage"])
        if kwargs["stage"] in {"discovery_matrix", "discovery_shared_matrix"}:
            raise storage.StorageBlocked("producer budget denied")
        return original_reserve(**kwargs)

    monkeypatch.setattr(authority, "reserve", deny_matrix_stage)
    with patch("ima.feature_program.synthesize") as build:
        with pytest.raises(storage.StorageBlocked, match="producer budget denied"):
            materialize(
                discovery_fixture(), DiscoverySpecV2(windows_days=(90,)),
                "source", tmp_path / "discovery", shared=shared,
            )
    build.assert_not_called()
    assert stages[0] == "discovery_control"
    assert stages[-1] == ("discovery_shared_matrix" if shared else "discovery_matrix")
    assert not list((tmp_path / "discovery").rglob("matrix.*"))
    assert not list((tmp_path / "discovery").rglob("manifest.json"))


@pytest.mark.parametrize("shared", [False, True])
def test_cache_hit_reads_original_format_under_pressure_without_matrix_reservation(
    tmp_path, monkeypatch, shared
):
    authority = enabled_authority(tmp_path, monkeypatch)
    frame = discovery_fixture()
    spec = DiscoverySpecV2(windows_days=(90,))
    expected, _ = synthesize(frame, spec)
    first, manifest = materialize(frame, spec, "source", tmp_path / "discovery", shared=shared)
    if shared:
        assert not first.load().flags.writeable
        np.testing.assert_allclose(first.load(), expected.to_numpy(), equal_nan=True)
    else:
        pd.testing.assert_frame_equal(first[expected.columns], expected)

    original_statvfs = authority._statvfs

    def tight_capacity(path):
        values = original_statvfs(path)
        values.f_bavail = 4 * 1024**2
        return values

    monkeypatch.setattr(authority, "_statvfs", tight_capacity)
    reserved_stages = []
    original_reserve = authority.reserve

    def reject_producer_budget(**kwargs):
        reserved_stages.append(kwargs["stage"])
        if kwargs["stage"] in {"discovery_matrix", "discovery_shared_matrix"}:
            raise AssertionError("cache hit requested a producer reservation")
        return original_reserve(**kwargs)

    monkeypatch.setattr(authority, "reserve", reject_producer_budget)
    with patch("ima.feature_program.synthesize", side_effect=AssertionError("cache hit rebuilt")), \
            patch("ima.research_cache_capacity.trim_for_capacity", side_effect=AssertionError("cache hit trimmed")):
        again, other = materialize(frame, spec, "source", tmp_path / "discovery", shared=shared)
    assert manifest["matrix_id"] == other["matrix_id"]
    assert "discovery_control" in reserved_stages
    assert not ({"discovery_matrix", "discovery_shared_matrix"} & set(reserved_stages))
    if shared:
        assert again.cache_status == "hit"
        np.testing.assert_allclose(again.load(), first.load(), equal_nan=True)
        assert again.load().flags.writeable is False
    else:
        pd.testing.assert_frame_equal(again, first)
    assert authority.status()["reserved_bytes"] == 0


@pytest.mark.parametrize("shared", [False, True])
def test_discovery_miss_reclaims_inactive_fold_capacity_below_cache_quota(
    tmp_path, monkeypatch, shared
):
    from tests.test_cache_physical_capacity import fixture

    authority, _, paths = fixture(tmp_path, available=17 * 1024**2 + 1000)
    monkeypatch.setenv("IMA_STORAGE_ROOT", str(authority.root))
    monkeypatch.setattr(storage, "StorageAuthority", lambda root: authority)
    expected, _ = synthesize(discovery_fixture(), DiscoverySpecV2(windows_days=(90,)))
    result, _ = materialize(discovery_fixture(), DiscoverySpecV2(windows_days=(90,)),
                            "capacity-source", tmp_path / "discovery", shared=shared)
    assert not paths[0].exists() and paths[1].exists()
    assert authority.status()["reserved_bytes"] == 0
    if shared:
        np.testing.assert_allclose(result.load(), expected.to_numpy(), equal_nan=True)
    else:
        pd.testing.assert_frame_equal(result[expected.columns], expected)


@pytest.mark.parametrize("shared", [False, True])
def test_discovery_miss_keeps_pinned_fold_capacity_and_denies_before_synthesis(
    tmp_path, monkeypatch, shared
):
    import fcntl
    from contextlib import ExitStack
    from tests.test_cache_physical_capacity import fixture

    authority, root, paths = fixture(tmp_path, available=17 * 1024**2 + 1000)
    monkeypatch.setenv("IMA_STORAGE_ROOT", str(authority.root))
    monkeypatch.setattr(storage, "StorageAuthority", lambda root: authority)
    with ExitStack() as pins:
        for path in paths:
            stream = pins.enter_context((root / (path.name + ".lock")).open("rb"))
            fcntl.flock(stream, fcntl.LOCK_SH)
        with patch("ima.feature_program.synthesize") as build:
            with pytest.raises(storage.StorageBlocked):
                materialize(discovery_fixture(), DiscoverySpecV2(windows_days=(90,)),
                            "pinned-source", tmp_path / "discovery", shared=shared)
            build.assert_not_called()
    assert all(path.exists() for path in paths)
    assert authority.status()["reserved_bytes"] == 0


@pytest.mark.parametrize("shared", [False, True])
def test_native_partial_write_failure_cleans_payload_and_preserves_primary_error(
    tmp_path, monkeypatch, shared
):
    authority = enabled_authority(tmp_path, monkeypatch)
    root = tmp_path / "discovery"
    frame = discovery_fixture()
    spec = DiscoverySpecV2(windows_days=(90,))
    message = "native serializer primary failure"
    growth_payload = b"x" * (16 * 1024**2 + 1)
    reservations = []
    original_reserve = authority.reserve

    def record_reservation(**kwargs):
        reservations.append(kwargs)
        return original_reserve(**kwargs)

    monkeypatch.setattr(authority, "reserve", record_reservation)
    with patch.object(authority, "extend", wraps=authority.extend) as extend:
        if shared:
            import joblib

            def partial_joblib_dump(value, stream, compress=0):
                stream.write(growth_payload)
                raise RuntimeError(message)

            serializer = patch.object(joblib, "dump", side_effect=partial_joblib_dump)
        else:
            def partial_parquet(self, stream, *args, **kwargs):
                stream.write(growth_payload)
                raise RuntimeError(message)

            serializer = patch.object(pd.DataFrame, "to_parquet", partial_parquet)

        with serializer:
            with pytest.raises(RuntimeError, match=message):
                materialize(frame, spec, "source", root, shared=shared)
        producer_stage = "discovery_shared_matrix" if shared else "discovery_matrix"
        producer = next(item for item in reservations if item["stage"] == producer_stage)
        assert producer["bytes_required"] >= 16 * 1024**2
        assert producer["inodes_required"] == 8
        assert extend.call_count >= 1

    assert not list(root.rglob("manifest.json"))
    assert not list(root.rglob("matrix.*"))
    if shared:
        assert not list((root / "shared-numeric-v1").glob("*-*"))
    else:
        assert not list(root.rglob("matrix-*.parquet"))
    assert authority.status()["reserved_bytes"] == 0


def test_shared_postpublish_validation_failure_removes_manifest_and_payload(tmp_path, monkeypatch):
    authority = enabled_authority(tmp_path, monkeypatch)
    root = tmp_path / "discovery"
    message = "shared validation primary failure"
    with patch("ima.feature_program.DiscoveryMatrixArtifact.load", side_effect=RuntimeError(message)):
        with pytest.raises(RuntimeError, match=message):
            materialize(
                discovery_fixture(), DiscoverySpecV2(windows_days=(90,)),
                "source", root, shared=True,
            )
    assert not list(root.rglob("manifest.json"))
    assert not list(root.rglob("matrix.*"))
    assert authority.status()["reserved_bytes"] == 0


@pytest.mark.parametrize("shared", [False, True])
def test_cleanup_failure_is_not_allowed_to_mask_native_storage_pressure(
    tmp_path, monkeypatch, shared
):
    authority = enabled_authority(tmp_path, monkeypatch)
    root = tmp_path / "discovery"
    cleanup_path = []
    native_error = OSError(errno.ENOSPC, "injected native write exhaustion")

    if shared:
        import joblib

        original_rmdir = feature_program.Path.rmdir

        def fail_owned_temp_cleanup(path, *args, **kwargs):
            path = feature_program.Path(path)
            if path.parent.name == "shared-numeric-v1" and len(path.name) > 25 and path.name[24] == "-":
                cleanup_path.append(path)
                raise OSError(errno.EACCES, "injected temporary directory cleanup failure")
            return original_rmdir(path, *args, **kwargs)

        monkeypatch.setattr(feature_program.Path, "rmdir", fail_owned_temp_cleanup)

        def partial_joblib_dump(value, stream, compress=0):
            stream.write(b"partial-joblib")
            raise native_error

        serializer = patch.object(joblib, "dump", side_effect=partial_joblib_dump)
    else:
        original_unlink = feature_program.Path.unlink

        def fail_owned_temp_unlink(path, *args, **kwargs):
            if path.parent.parent == root and path.name.startswith("matrix-"):
                cleanup_path.append(path)
                raise OSError(errno.EACCES, "injected temporary file cleanup failure")
            return original_unlink(path, *args, **kwargs)

        monkeypatch.setattr(feature_program.Path, "unlink", fail_owned_temp_unlink)

        def partial_parquet(self, stream, *args, **kwargs):
            stream.write(b"partial-parquet")
            raise native_error

        serializer = patch.object(pd.DataFrame, "to_parquet", partial_parquet)

    with serializer:
        with pytest.raises(storage.StorageBlocked) as caught:
            materialize(
                discovery_fixture(), DiscoverySpecV2(windows_days=(90,)),
                "source", root, shared=shared,
            )

    error = caught.value
    assert isinstance(error.__cause__, OSError)
    assert error.__cause__.errno == errno.ENOSPC
    notes = getattr(error.__cause__, "__notes__", [])
    assert cleanup_path
    assert any(str(cleanup_path[0]) in note and "may remain" in note for note in notes)
    assert cleanup_path[0].exists()
    assert not list(root.rglob("manifest.json"))
    assert authority.status()["reserved_bytes"] == 0


def test_corrupt_shared_cache_with_unknown_payload_is_held_intact(tmp_path,monkeypatch):
    authority=enabled_authority(tmp_path,monkeypatch)
    frame=discovery_fixture()
    spec=DiscoverySpecV2(windows_days=(90,))
    artifact,_=materialize(frame,spec,'source',tmp_path/'discovery',shared=True)
    manifest_path=artifact.path/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    manifest['matrix_sha256']='0'*64
    manifest_path.write_text(json.dumps(manifest))
    (artifact.path/'model.bin').write_bytes(b'unknown payload must survive')
    before={path.name:path.read_bytes() for path in artifact.path.iterdir()}
    with pytest.raises(storage.StorageBlocked,match='unqualified discovery payload'):
        materialize(frame,spec,'source',tmp_path/'discovery',shared=True)
    assert {path.name:path.read_bytes() for path in artifact.path.iterdir()}==before
    assert authority.status()['reserved_bytes']==0
