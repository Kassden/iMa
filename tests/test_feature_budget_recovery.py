import importlib.util
from pathlib import Path
from unittest.mock import patch


def load_budget():
    path=Path(__file__).resolve().parents[1]/'deploy/systemd/ima-feature-exploration-budget.py'
    spec=importlib.util.spec_from_file_location('budget_recovery_fixture',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_stopped_main_reserves_future_partition_without_fake_cgroup(tmp_path,monkeypatch):
    budget=load_budget(); budget.STATE=tmp_path/'state.json'; budget.DROPIN=tmp_path/'dropin.conf'
    # Exercise the real admission path in an owned fixture rather than spoofing
    # process UID and accidentally targeting production control metadata.
    control=tmp_path/'storage-control';control.mkdir(mode=0o700)
    monkeypatch.setattr(budget,'OWN_HOME',tmp_path)
    monkeypatch.setattr(budget,'STORAGE_ROOT',control)
    monkeypatch.setattr(budget,'require_owner',lambda:budget.require_owner_identity(1001,Path('/home/imaopt')))
    monkeypatch.setenv('IMA_STORAGE_ROOT',str(control))
    original_authority=budget.research_storage.StorageAuthority
    monkeypatch.setattr(budget.research_storage,'StorageAuthority',
        lambda root:original_authority(root,headroom_bytes=0,headroom_inodes=0))
    limits={'MemoryMax':str(budget.TOTAL_BYTES),'ActiveState':'failed','MainPID':'0',
            'ControlGroup':'','CPUQuotaPerSecUSec':'26s'}
    calls=[]
    def manager(*args): return limits[args[args.index('-p')+1]]
    def run(args,**kwargs):
        calls.append(args)
        for argument in args:
            if argument.startswith('MemoryMax='): limits['MemoryMax']=argument.split('=')[1]
    with patch.object(budget,'manager',manager),patch.object(budget.subprocess,'run',run):
        budget.main('reserve')
    assert limits['MemoryMax']==str(budget.MODEL_BYTES)
    assert budget.MODEL_BYTES+budget.FEATURE_BYTES==100_000_000_000
    assert not any('restart' in c or 'freeze' in c or 'thaw' in c for c in calls)
    assert budget.DROPIN.read_text()==budget.BODY


def test_stopped_state_with_live_pid_is_not_treated_as_empty():
    budget=load_budget()
    def manager(*args): return 'failed' if 'ActiveState' in args else '99'
    import pytest
    with patch.object(budget,'manager',manager):
        with pytest.raises(RuntimeError,match='live PID'):
            with budget.frozen_model(): pass
