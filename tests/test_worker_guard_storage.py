import os
from pathlib import Path
import tempfile

from ima.research_storage import StorageAuthority
from ima.research_worker_runtime import BoundedFitExecutor, _task_guard_path, read_runtime


def test_enabled_actual_worker_keeps_control_records_with_storage_authority(tmp_path,monkeypatch):
    authority=StorageAuthority(tmp_path/'authority',headroom_bytes=0,headroom_inodes=0)
    monkeypatch.setenv('IMA_STORAGE_ROOT',str(authority.root))
    # Linux commonly uses a shared, root-owned temporary directory. Its ownership
    # must not control whether the admitted worker can publish its own records.
    shared_tmp=Path('/tmp').resolve()
    monkeypatch.setenv('TMPDIR',str(shared_tmp))
    monkeypatch.setattr(tempfile,'tempdir',str(shared_tmp))
    assert Path(tempfile.gettempdir())==shared_tmp
    runtime_path=tmp_path/'attempt'/'runtime.json'
    with BoundedFitExecutor(1) as executor:
        result=executor.submit(pow,2,3,runtime_path=runtime_path,timeout=30).result(40)
    assert result==8
    state=read_runtime(runtime_path)
    guard=_task_guard_path(state['process_identity'],state['boot_id'])
    assert guard.parent==authority.root/'worker-guards'
    assert guard.stat().st_uid==os.getuid()
    assert guard.stat().st_dev==authority.root.stat().st_dev
    assert read_runtime(guard)['status']=='idle'
    status=authority.status()
    assert status['reserved_bytes']==0
    assert status['reserved_inodes']==0
    assert status['reservations']=={}


def test_disabled_guard_path_preserves_legacy_namespace(monkeypatch):
    monkeypatch.delenv('IMA_STORAGE_ROOT',raising=False)
    guard=_task_guard_path({'pid':123,'uid':os.getuid()},'fixture-boot')
    assert guard.parent==Path(tempfile.gettempdir())
    assert guard.name.startswith(f'ima-worker-{os.getuid()}-')
