import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from ima.research_external_planner import read_external_decision
from ima.optimizer import CampaignConfig


class ExternalPlannerTests(unittest.TestCase):
    def test_external_mode_requires_explicit_v6_identity(self):
        config = CampaignConfig(campaign_dir=Path('/tmp/external-test'),
                                policy='agentic', research_policy='expansion_v6',
                                planner_mode='external', model='gpt-6-luna')
        config.validate()
        with self.assertRaises(ValueError):
            replace(config, model='openrouter/local-policy').validate()
        with self.assertRaises(ValueError):
            replace(config, research_policy='legacy').validate()

    def test_identity_evidence_and_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'planner-inbox').mkdir()
            path = root / 'planner-inbox/D000001.json'
            config = SimpleNamespace(campaign_dir=root, model='gpt-6-luna',
                                     planner_reasoning_effort='high', planner_timeout_seconds=.01)
            evidence = {'decision_id': 'D000001', 'evidence_id': 'fresh'}
            valid = {'model': 'gpt-6-luna', 'reasoning_effort': 'high',
                     'decision': {'decision_id': 'D000001', 'evidence_id': 'fresh',
                                  'trial_budget': 0, 'review_reason': 'Review current training.'}}
            path.write_text(json.dumps(valid))
            result = read_external_decision(evidence, config)
            self.assertIsNone(result['usage']['total_cost_usd'])
            self.assertEqual(result['orchestrator']['model'], 'gpt-6-luna')
            for change in ({'model': 'fixture'}, {'reasoning_effort': 'low'},
                           {'decision': dict(valid['decision'], evidence_id='stale')},
                           {'decision': dict(valid['decision'], trial_budget=5)}):
                with self.subTest(change=change):
                    path.write_text(json.dumps(valid | change))
                    with self.assertRaises(ValueError):
                        read_external_decision(evidence, config)
            path.unlink()
            with self.assertRaises(TimeoutError):
                read_external_decision(evidence, config)
            (root / 'STOP').touch()
            with self.assertRaises(RuntimeError):
                read_external_decision(evidence, config)

    def test_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'planner-inbox').mkdir()
            target = root / 'outside.json'
            target.write_text('{}')
            (root / 'planner-inbox/D000001.json').symlink_to(target)
            config = SimpleNamespace(campaign_dir=root, model='gpt-6-luna',
                                     planner_reasoning_effort='high', planner_timeout_seconds=.01)
            with self.assertRaises(ValueError):
                read_external_decision({'decision_id': 'D000001', 'evidence_id': 'fresh'}, config)
