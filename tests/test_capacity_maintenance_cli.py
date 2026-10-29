"""Private-authority CLI tests; no native authority or production mutation."""
import json
from types import SimpleNamespace

import pytest

from scripts import maintain_research_capacity as cli
from ima import research_storage as storage
from tests.test_cache_physical_capacity import fixture
from tests.test_cache_hold_policy import policy


def setup(tmp_path, monkeypatch, *, available=1000):
    monkeypatch.setenv('IMA_STORAGE_ROOT', '')
    authority, root, paths = fixture(tmp_path, available=available)
    canonical = root.with_name('fold-preparation-cache')
    root.rename(canonical)
    state_path = authority.root / 'state.json'
    state = json.loads(state_path.read_text())
    state['caches'] = {str(canonical / row['artifact_id']): dict(row, root=str(canonical))
                       for row in state['caches'].values()}
    state_path.write_text(json.dumps(state))
    paths = [canonical / path.name for path in paths]
    allocated = {path: sum(p.stat().st_blocks * 512 for p in path.iterdir())
                 for path in paths}
    authority._statvfs = lambda _: SimpleNamespace(
        f_bavail=available + sum(size for path, size in allocated.items() if not path.exists()),
        f_frsize=1, f_favail=1000)
    # Both CLI construction and the existing output helper use this private factory.
    monkeypatch.setattr(cli, 'StorageAuthority', lambda path: authority)
    monkeypatch.setattr(storage, 'StorageAuthority', lambda path: authority)
    policy(monkeypatch, tmp_path, [paths[1]])
    receipts = tmp_path / 'receipts'
    argv = ['--authority', str(authority.root), '--required-bytes', '5000',
            '--receipts', str(receipts)]
    return authority, paths, receipts, argv


def output(capsys):
    return [json.loads(line) for line in capsys.readouterr().out.splitlines()]


@pytest.mark.parametrize('fault', ['missing', 'changed'])
def test_invalid_policy_denies_before_trim(tmp_path, monkeypatch, capsys, fault):
    authority, paths, receipts, argv = setup(tmp_path, monkeypatch)
    if fault == 'missing':
        monkeypatch.delenv('IMA_CACHE_HOLD_POLICY_SHA256')
    else:
        source = tmp_path / 'cache-holds.json'
        source.chmod(0o600)
        source.write_text('{}')
    def forbidden(*args, **kwargs):
        pytest.fail('trim called before policy validation')
    monkeypatch.setattr(cli, 'trim_for_capacity', forbidden)
    assert cli.main(argv) == 2
    assert output(capsys)[-1]['status'] == 'blocked'
    assert all(path.exists() for path in paths)
    assert not receipts.exists()


def test_inactive_cache_below_quota_retired_and_held_peer_intact(tmp_path, monkeypatch, capsys):
    authority, paths, receipts, argv = setup(tmp_path, monkeypatch)
    allocated = authority.status()['cache_bytes']
    assert allocated < authority.cache_soft_bytes
    held = {p.name: p.read_bytes() for p in paths[1].iterdir()}
    assert cli.main(argv) == 0
    intent, completion = output(capsys)
    assert intent['event'] == 'intent' and intent['cache_growth_bytes'] == 0
    assert completion['evicted'] == [str(paths[0])]
    assert not paths[0].exists()
    assert {p.name: p.read_bytes() for p in paths[1].iterdir()} == held
    receipt = next(receipts.glob('*.json'))
    record = json.loads(receipt.read_text())
    assert record['retired'] == completion['retired']
    assert record['retired'][0]['allocated_bytes'] > 0
    assert receipt.stat().st_mode & 0o777 == 0o400
    assert list(receipts.iterdir()) == [receipt]
    assert authority.status()['reserved_bytes'] == 0


def test_healthy_repeats_have_no_receipt_growth(tmp_path, monkeypatch, capsys):
    authority, paths, receipts, argv = setup(tmp_path, monkeypatch, available=10**7)
    for _ in range(2):
        assert cli.main(argv) == 0
        assert output(capsys)[-1]['evicted'] == []
    assert not receipts.exists()
    assert all(path.exists() for path in paths)


def test_negative_budget_denied_before_action(tmp_path, monkeypatch, capsys):
    authority, paths, receipts, argv = setup(tmp_path, monkeypatch)
    argv[3] = '-1'
    monkeypatch.setattr(cli, 'trim_for_capacity', lambda *a, **k: pytest.fail('trim called'))
    with pytest.raises(SystemExit) as caught:
        cli.main(argv)
    assert caught.value.code == 2
    assert all(path.exists() for path in paths)
    assert not receipts.exists()


def test_blocked_no_actions_and_partial_retirements(tmp_path, monkeypatch, capsys):
    authority, paths, receipts, argv = setup(tmp_path, monkeypatch)
    (tmp_path / 'cache-holds.json').chmod(0o600)
    policy(monkeypatch, tmp_path, paths)
    assert cli.main(argv) == 2
    assert output(capsys)[-1]['evicted'] == []
    assert not receipts.exists() and all(p.exists() for p in paths)
    (tmp_path / 'cache-holds.json').chmod(0o600)
    policy(monkeypatch, tmp_path, [paths[1]])
    argv[3] = '1000000'
    assert cli.main(argv) == 2
    completion = output(capsys)[-1]
    assert completion['evicted'] == [str(paths[0])]
    assert json.loads(next(receipts.glob('*.json')).read_text())['evicted'] == completion['evicted']


def test_failed_receipt_admission_reports_retirement(tmp_path, monkeypatch, capsys):
    authority, paths, receipts, argv = setup(tmp_path, monkeypatch)
    def denied(*args, **kwargs):
        raise storage.StorageBlocked('test admission denied')
    monkeypatch.setattr(cli, 'admitted_output', denied)
    assert cli.main(argv) == 2
    completion = output(capsys)[-1]
    assert completion['status'] == 'audit_incomplete'
    assert completion['evicted'] == [str(paths[0])]
    assert 'receipt' not in completion and not receipts.exists()


def test_missing_authority_is_not_bootstrapped(tmp_path, monkeypatch, capsys):
    authority, paths, receipts, argv = setup(tmp_path, monkeypatch)
    missing = tmp_path / 'unrelated'
    argv[1] = str(missing)
    assert cli.main(argv) == 2
    assert output(capsys)[-1]['status'] == 'blocked'
    assert not missing.exists() and all(p.exists() for p in paths)


def test_source_api_failure_is_visible_and_closed(tmp_path, monkeypatch, capsys):
    authority, paths, receipts, argv = setup(tmp_path, monkeypatch)
    def failed(*args, **kwargs):
        raise RuntimeError('secret-must-not-be-printed')
    monkeypatch.setattr(cli, 'trim_for_capacity', failed)
    assert cli.main(argv) == 2
    messages = output(capsys)
    assert messages[-1]['error_type'] == 'RuntimeError'
    assert messages[-1]['status'] == 'audit_incomplete'
    assert messages[-1]['retirement_outcome'] == 'unknown'
    assert 'secret-must-not-be-printed' not in json.dumps(messages)
    assert all(p.exists() for p in paths) and not receipts.exists()
