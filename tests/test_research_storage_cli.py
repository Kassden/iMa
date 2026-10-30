from dataclasses import asdict
import hashlib
import json
from unittest.mock import patch

import pytest

from ima.research_control import ControlIntentStore
from scripts import research_storage as cli
from tests.test_research_control import authority,policy


def test_explicit_control_bootstrap_and_actual_pending_readback(tmp_path,monkeypatch,capsys):
    a=authority(tmp_path/'authority')
    policy_file=tmp_path/'policy.json'
    policy_file.write_text(json.dumps(asdict(policy())))
    digest=hashlib.sha256(policy_file.read_bytes()).hexdigest()
    monkeypatch.setattr(cli,'StorageAuthority',lambda root:a)
    assert cli.main(['control-init','--authority',str(a.root),'--policy',str(policy_file),
                     '--policy-sha256',digest])==0
    created=json.loads(capsys.readouterr().out)
    assert created['mode']=='explicit_healthy_control_bootstrap'
    assert created['normal_restart_authorized'] is False and created['rollout_qualified'] is False
    assert created['pending_count']==0
    campaign=tmp_path/'campaign';campaign.mkdir()
    store=ControlIntentStore(a)
    with store.claim(campaign,'tracking'):
        store.stage('execution',7,{'source_input':'complete-but-not-printed'})
    assert cli.main(['control-status','--authority',str(a.root)])==0
    readback=json.loads(capsys.readouterr().out)
    assert readback['pending_count']==1
    assert readback['pending_intents'][0]['sequence']==7
    assert readback['pending_intents'][0]['state']=='complete'
    assert 'payload' not in readback['pending_intents'][0]
    assert readback['policy_identity_sha256']==created['policy_identity_sha256']
    assert readback['normal_headroom_bytes']==a.headroom_bytes


def test_wrong_bootstrap_policy_digest_cannot_create_authority_or_touch_cache(tmp_path):
    config=tmp_path/'policy.json';config.write_text(json.dumps(asdict(policy())))
    protected=tmp_path/'cache.bin';protected.write_bytes(b'protected')
    with patch.object(cli,'StorageAuthority') as construct:
        with pytest.raises(ValueError,match='digest'):
            cli.main(['control-init','--authority',str(tmp_path/'missing'),
                      '--policy',str(config),'--policy-sha256','0'*64])
        construct.assert_not_called()
    assert protected.read_bytes()==b'protected'
    assert not (tmp_path/'missing').exists()


def test_control_readback_never_initializes_a_missing_store(tmp_path):
    with pytest.raises(ValueError,match='Existing authority'):
        cli.main(['control-status','--authority',str(tmp_path/'missing')])
    assert not (tmp_path/'missing').exists()
