"""Static deployment contracts; live membership/enforcement needs separate proof."""
import configparser
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]/'deploy'/'systemd'


def unit(name):
    parsed=configparser.ConfigParser(interpolation=None,strict=True)
    assert parsed.read(ROOT/name)
    return parsed


def test_owned_units_share_one_aggregate_decimal_memory_budget():
    parent=unit('ima-research.slice')['Slice']
    assert int(parent['MemoryMax'])==100_000_000_000
    assert 0<int(parent['MemoryHigh'])<int(parent['MemoryMax'])
    model=unit('ima-research-expansion-supervisor.service')['Service']
    feature=unit('ima-feature-exploration.service')['Service']
    selected=unit('ima-research-shared-budget.conf')['Service']
    assert model['Slice']==feature['Slice']==selected['Slice']=='ima-research.slice'
    assert int(model['MemoryMax'])<=int(parent['MemoryMax'])
    assert int(feature['MemoryMax'])==8_000_000_000
    assert not any(ROOT.glob('user*.slice'))


def test_feature_failures_reach_a_bounded_restart_limit():
    parsed=unit('ima-feature-exploration.service')
    policy,service=parsed['Unit'],parsed['Service']
    interval=int(policy['StartLimitIntervalSec'])
    burst=int(policy['StartLimitBurst'])
    retry=int(service['RestartSec'])
    assert service['Restart']=='on-failure'
    assert 1<=burst<=3 and retry>=60
    assert retry*burst<interval
    assert int(service['TimeoutStartSec'])<=interval
