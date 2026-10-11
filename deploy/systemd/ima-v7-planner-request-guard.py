"""Pinned additive planner guard over the verified own-user discipline policy."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path('/home/imaopt/research-v2')
CAMPAIGN = ROOT/'campaigns/agentic_v7_openrouter_recovery_20261007'
OLD = CAMPAIGN/'ops/uncapped-budget-20261008/ima_uncapped_campaign_launch.py'
OLD_SHA = 'c663043d71021870a42e4f077c5ce2a318f4c118ae31fe153462f3afe05e49c9'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    if os.getuid() != 1001 or Path.home() != Path('/home/imaopt'):
        raise RuntimeError('Requires own imaopt user')
    if hashlib.sha256(OLD.read_bytes()).hexdigest() != OLD_SHA:
        raise RuntimeError('Predecessor policy checksum mismatch')
    stage = Path(__file__).resolve().parents[2]
    manifest = json.loads((stage/'guard-manifest.json').read_text())
    for name, sha in manifest['files'].items():
        path = stage/name
        if path.is_symlink() or not path.resolve().is_relative_to(stage):
            raise RuntimeError('Guard payload path mismatch')
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise RuntimeError('Guard checksum mismatch: '+name)
    guard = load('ima.planner_request_guard', stage/'ima/planner_request_guard.py')
    predecessor = load('ima_guard_predecessor', OLD)
    install = predecessor.install_policy

    def install_combined():
        receipt = install()
        receipt.update(guard.install_guard(CAMPAIGN), guard_manifest=manifest,
                       predecessor_policy_sha256=OLD_SHA)
        print('Planner request guard installed: '+json.dumps(receipt, sort_keys=True), flush=True)
        return receipt

    predecessor.install_policy = install_combined
    if '--guard-self-test' in sys.argv:
        receipt = install()
        receipt.update(predecessor.canary())
        receipt.update(guard.install_guard(CAMPAIGN), guard_manifest=manifest)
        from ima import research_planner_context as context
        evidence = json.loads((CAMPAIGN/'evidence/D000203.json').read_text())
        response = json.loads((CAMPAIGN/'planner-responses/D000203.json').read_text())['response']
        replay = guard.bind_response(response, evidence)
        view = context.decision_planner_evidence({'evidence': evidence})
        receipt.update(replay_binding=replay['proposal_identity_binding'],
                       replay_context_bytes=len(guard.encoded(view)))
        for key in ('completed_trial_index',):
            assert view['evidence'][key] == evidence[key]
        print(json.dumps(receipt, sort_keys=True))
        return 0
    return predecessor.main()


if __name__ == '__main__':
    raise SystemExit(main())
