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

    def dataset_equivalence():
        from ima.dataset_registry import DatasetRegistry, file_sha256
        from ima.research_controller import protocol_spec_parameters
        config = json.loads((CAMPAIGN/'config.json').read_text())
        path = Path(config['dataset_path'])
        identity = json.loads((path.parent/'manifest.json').read_text())['dataset_id']
        manifest = DatasetRegistry(Path(config['dataset_registry_path'])).verify(identity)
        assert file_sha256(path) == file_sha256(Path(manifest['features_path']))
        configured = protocol_spec_parameters(json.loads(Path(config['protocol_path']).read_text()))
        registered = protocol_spec_parameters(json.loads(Path(manifest['protocol_path']).read_text()))
        assert configured == registered
        return identity

    def install_combined():
        receipt = install()
        receipt.update(guard.install_guard(CAMPAIGN, default_dataset_id=dataset_equivalence()), guard_manifest=manifest,
                       predecessor_policy_sha256=OLD_SHA)
        print('Planner request guard installed: '+json.dumps(receipt, sort_keys=True), flush=True)
        return receipt

    predecessor.install_policy = install_combined
    if '--guard-self-test' in sys.argv:
        receipt = install()
        receipt.update(predecessor.canary())
        receipt.update(guard.install_guard(CAMPAIGN, default_dataset_id=dataset_equivalence()), guard_manifest=manifest)
        from ima import research_planner_context as context
        evidence = json.loads((CAMPAIGN/'evidence/D000203.json').read_text())
        response = json.loads((CAMPAIGN/'planner-responses/D000203.json').read_text())['response']
        replay = guard.bind_response(response, evidence)
        view = context.decision_planner_evidence({'evidence': evidence})
        receipt.update(replay_binding=replay['proposal_identity_binding'],
                       replay_context_bytes=len(guard.encoded(view)))
        for key in ('completed_trial_index',):
            assert view['evidence'][key] == evidence[key]
        from ima.research_expansion import PlannerDecision
        from ima import experiment_discipline as discipline
        d205 = json.loads((CAMPAIGN/'planner-responses/D000205.json').read_text())['response']['decision']
        e205 = json.loads((CAMPAIGN/'evidence/D000205.json').read_text())
        receipt['effective_dataset_replay'] = discipline.validate_decision(
            PlannerDecision.model_validate(d205), e205, {'programs': []})
        d207 = json.loads((CAMPAIGN/'planner-responses/D000207.json').read_text())['response']['decision']
        e207 = json.loads((CAMPAIGN/'evidence/D000207.json').read_text())
        try:
            guard.semantic_preflight(PlannerDecision.model_validate(d207), e207)
        except ValueError as error:
            assert 'transforms' in str(error)
            receipt['undeclared_transform_preflight'] = 'rejected inside bounded repair boundary'
        else:
            raise AssertionError('Undeclared transform change escaped preflight')
        print(json.dumps(receipt, sort_keys=True))
        return 0
    return predecessor.main()


if __name__ == '__main__':
    raise SystemExit(main())
