"""Paper action contract and real offline PL/EV/portfolio integration tests."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from pydantic import ValidationError

from ima.research_betting import PaperResearchRequest, evaluate_paper_research, validate_paper_research


class PaperResearchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.campaign = Path(temporary.name).resolve()
        self.frame = pd.DataFrame({
            'fold_id': ['f1'] * 4, 'race_id': ['r1'] * 4,
            'horse_no': ['1', '2', '3', '4'], 'date': ['2024-01-01'] * 4,
            'target_win': [1, 0, 0, 0], 'model_probability': [.4, .3, .2, .1],
            'selected_probability': [.7, .1, .1, .1],
        })
        self.rows = [self.attempt('a1'), self.attempt('a2', [.1, .2, .3, .4])]
        self.request = {'request_id': 'paper-1', 'evidence_id': 'e1', 'attempt_ids': ['a1'],
                        'pools': ['win', 'place', 'quinella', 'trio']}

    def attempt(self, name, probabilities=None):
        directory = self.campaign / 'trials' / name
        directory.mkdir(parents=True, exist_ok=True)
        frame = self.frame.copy()
        if probabilities is not None:
            frame['model_probability'] = probabilities
        frame.to_csv(directory / 'predictions.csv', index=False)
        protocol = directory / 'protocol.json'
        protocol.write_text('{"protocol_id":"p1"}')
        population = frame[['fold_id', 'race_id', 'horse_no', 'date', 'target_win']].sort_values(
            ['fold_id', 'race_id', 'horse_no', 'date'], kind='stable')
        digest = hashlib.sha256(population.to_json(orient='records', date_format='iso').encode()).hexdigest()
        result = {'attempt_id': name, 'status': 'completed', 'target_kind': 'win_probability',
                  'recipe_hash': 'recipe-' + name,
                  'artifacts': {'predictions': str(directory / 'predictions.csv'), 'protocol': str(protocol)},
                  'lineage': {'dataset_hash': 'd1', 'dataset_id': 'dataset1', 'protocol_id': 'p1',
                              'protocol_hash': hashlib.sha256(protocol.read_bytes()).hexdigest(),
                              'prediction_sha256': hashlib.sha256((directory / 'predictions.csv').read_bytes()).hexdigest(),
                              'evaluation_population_hash': digest, 'evaluation_population_id': 'evaluated-' + digest,
                              'availability_policy': 'strict', 'code_revision': 'test', 'environment_hash': 'env'}}
        return {'attempt_id': name, 'status': 'completed', 'result': result}

    def run_request(self, **changes):
        return evaluate_paper_research(self.request | changes, campaign_dir=self.campaign, terminal_results=self.rows)

    def test_preflight_validates_frozen_inputs_without_simulation(self):
        with patch('ima.research_betting._race_report', side_effect=AssertionError('simulation in preflight')):
            preflight = validate_paper_research(self.request, campaign_dir=self.campaign,
                                               terminal_results=self.rows)
        report = self.run_request()
        for name in ('comparison', 'evaluation_key', 'model_ids'):
            self.assertEqual(preflight[name], report[name])
        self.assertEqual(preflight['rows'], 4)
        self.assertEqual(preflight['available_races'], 1)
        self.assertGreater(preflight['input_memory_bytes'], 0)
        self.assertGreater(preflight['cold_private_memory_bytes'], 512 * 1024**2)

    def test_contract_strict_bounds_and_unknown_fields(self):
        cases = [{'arbitrary_path': '/tmp/x'}, {'attempt_ids': ['../outside']},
                 {'attempt_ids': ['a1', 'a1']}, {'model_weights': []}, {'model_weights': [0]},
                 {'model_weights': [float('nan')]}, {'simulations': 999}, {'simulations': 100001},
                 {'max_races': 1001}, {'quote_mode': 'ex_post'}, {'currency': 'USD'},
                 {'max_races': 1000, 'simulations': 100000},
                 {'stake': 11}, {'pools': ['WIN', 'win']}, {'pools': ['imaginary']},
                 {'quote_mode': 'scenario'}, {'kelly_policy': {}, 'bankroll': None},
                 {'scenario_decimal_returns': {'WIN': 5}}]
        for change in cases:
            with self.subTest(change=change), self.assertRaises((ValidationError, ValueError)):
                PaperResearchRequest.model_validate(self.request | change)

    def test_weight_normalization_and_exact_model_mix(self):
        report = self.run_request(attempt_ids=['a1', 'a2'], model_weights=[3, 1], pools=['WIN'])
        self.assertEqual(report['request']['model_weights'], [.75, .25])
        ticket = report['races'][0]['reports'][0]
        self.assertEqual(ticket['ticket']['runners'], ['1'])
        self.assertAlmostEqual(ticket['probability'], .325)
        self.assertEqual(len(report['model_ids']), 2)
        with self.assertRaisesRegex(ValidationError, 'dynamic range'):
            self.run_request(attempt_ids=['a1', 'a2'], model_weights=[1e308, 1e-308])

    def test_fair_prices_never_invent_payout_ev_or_sizing(self):
        report = self.run_request(bankroll=1000)
        self.assertTrue(report['paper_only'])
        self.assertFalse(report['executable_evidence'])
        race = report['races'][0]
        self.assertEqual(report['coverage']['evaluated_races'], 1)
        for ticket in race['reports']:
            self.assertIsNone(ticket['expected_profit'])
            self.assertIsNone(ticket['decimal_return'])
            self.assertFalse(ticket['executable_evidence'])
            self.assertGreater(ticket['fair_decimal_return'], 0)
        self.assertIsNone(race['sizing'])
        self.assertIsNone(race['joint_outcome_metrics'])
        self.assertFalse(race['rules']['verified'])

    def test_exact_joint_probability_not_independent_products(self):
        race = self.run_request()['races'][0]
        quinella = next(r for r in race['reports'] if r['ticket']['pool'] == 'QIN')
        self.assertAlmostEqual(quinella['probability'], .4 * .3 / .6 + .3 * .4 / .7)
        self.assertNotAlmostEqual(quinella['probability'], .4 * .3)

    def test_scenario_ev_and_real_joint_sizing_remain_hypothetical(self):
        report = self.run_request(pools=['WIN', 'QIN'], quote_mode='scenario',
                                  scenario_decimal_returns={'WIN': 4, 'QIN': 5}, bankroll=1000)
        race = report['races'][0]
        for ticket in race['reports']:
            self.assertEqual(ticket['mode'], 'scenario')
            self.assertFalse(ticket['executable_evidence'])
            self.assertAlmostEqual(ticket['expected_profit'], 10 * (ticket['probability'] * ticket['decimal_return'] - 1))
        self.assertTrue(race['sizing']['success'])
        self.assertFalse(race['sizing']['executable_evidence'])
        self.assertEqual(race['sizing']['mode'], 'hypothetical_scenario')
        self.assertLessEqual(sum(map(float, race['sizing']['stakes'])), 100)
        self.assertIn('Hypothetical', race['sizing_note'])

    def test_blended_explicitly_tainted(self):
        report = self.run_request(probability_basis='blended', pools=['WIN'])
        self.assertTrue(report['ex_post_market_tainted'])
        self.assertAlmostEqual(report['races'][0]['reports'][0]['probability'], .7)

    def test_current_completed_only(self):
        with self.assertRaisesRegex(ValueError, 'current completed'):
            self.run_request(attempt_ids=['old-other-campaign'])
        self.rows[0]['status'] = 'running'
        with self.assertRaisesRegex(ValueError, 'current completed'):
            self.run_request()

    def test_mismatched_comparison_identity_rejected(self):
        self.rows[1]['result']['lineage']['protocol_id'] = 'other-protocol'
        with self.assertRaisesRegex(ValueError, 'different dataset/protocol'):
            self.run_request(attempt_ids=['a1', 'a2'])

    def test_protocol_corruption_rejected(self):
        Path(self.rows[0]['result']['artifacts']['protocol']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'protocol artifact hash'):
            self.run_request()

    def test_population_or_duplicate_runner_rejected(self):
        path = Path(self.rows[0]['result']['artifacts']['predictions'])
        frame = self.frame.copy()
        frame.loc[0, 'horse_no'] = 'other'
        frame.to_csv(path, index=False)
        self.rows[0]['result']['lineage']['prediction_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, 'population hash'):
            self.run_request()
        pd.concat([self.frame, self.frame.iloc[:1]]).to_csv(path, index=False)
        self.rows[0]['result']['lineage']['prediction_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, 'Duplicate scored'):
            self.run_request()

    def test_malformed_win_marginals_rejected(self):
        path = Path(self.rows[0]['result']['artifacts']['predictions'])
        frame = self.frame.copy()
        frame['model_probability'] *= .5
        frame.to_csv(path, index=False)
        self.rows[0]['result']['lineage']['prediction_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, 'sum to one'):
            self.run_request()

    def test_attempt_path_escape_rejected(self):
        self.rows[0]['result']['artifacts']['predictions'] = '/tmp/predictions.csv'
        with self.assertRaisesRegex(ValueError, 'current attempt output'):
            self.run_request()

    def test_deterministic_and_no_state_writes(self):
        before = {str(p): p.read_bytes() for p in self.campaign.rglob('*') if p.is_file()}
        first, second = self.run_request(), self.run_request()
        self.assertEqual(first, second)
        after = {str(p): p.read_bytes() for p in self.campaign.rglob('*') if p.is_file()}
        self.assertEqual(before, after)

    def test_small_field_counts_fail_honestly(self):
        self.frame = self.frame.iloc[:2].copy()
        self.frame['model_probability'] = [.7, .3]
        self.frame['selected_probability'] = [.7, .3]
        self.rows = [self.attempt('a1')]
        report = self.run_request(pools=['PLACE'], paid_places=3)
        self.assertEqual(report['coverage']['evaluated_races'], 0)
        self.assertEqual(report['status'], 'unsupported')
        self.assertIn('depth exceeds field', report['coverage']['skipped_races'][0]['reason'])
        report = self.run_request(pools=['PLACE'], paid_places=2)
        self.assertEqual(report['coverage']['evaluated_races'], 1)
        self.assertIn('NOT verified', report['races'][0]['rules']['source'])

    def test_sampled_joint_orders_have_no_false_tail_or_kelly_claim(self):
        self.frame = pd.DataFrame({
            'fold_id': ['f1'] * 12, 'race_id': ['r1'] * 12,
            'horse_no': [str(i) for i in range(1, 13)], 'date': ['2024-01-01'] * 12,
            'target_win': [1] + [0]*11, 'model_probability': [1/12]*12,
            'selected_probability': [1/12]*12,
        })
        self.rows = [self.attempt('a1')]
        report = self.run_request(pools=['TRI'], simulations=1000, quote_mode='scenario',
                                  scenario_decimal_returns={'TRI': 20}, bankroll=1000)
        race = report['races'][0]
        self.assertEqual(race['method'], 'fixed_budget_sampled_PL')
        self.assertFalse(race['distribution_metadata']['converged'])
        self.assertIsNone(race['sizing'])
        self.assertIn('unseen outcomes', race['probability_bounds'])
        self.assertEqual(report, self.run_request(pools=['TRI'], simulations=1000, quote_mode='scenario',
                                                 scenario_decimal_returns={'TRI': 20}, bankroll=1000))

    def test_all_supported_pools_share_coherent_orders(self):
        report = self.run_request(pools=['win', 'place', 'quinella', 'qpl', 'trio',
                                         'forecast', 'tierce', 'first4', 'quartet'])
        evs = report['races'][0]['reports']
        self.assertEqual(len(evs), 9)
        self.assertTrue(all(0 <= ev['probability'] <= 1 for ev in evs))
        self.assertAlmostEqual(next(ev['probability'] for ev in evs if ev['ticket']['pool'] == 'FIRST4'), 1)

    def test_symlinked_attempt_cannot_borrow_another_attempt_outputs(self):
        alias = self.campaign / 'trials' / 'alias'
        alias.symlink_to(self.campaign / 'trials/a1', target_is_directory=True)
        row = json.loads(json.dumps(self.rows[0]))
        row['attempt_id'] = row['result']['attempt_id'] = 'alias'
        row['result']['artifacts'] = {'predictions': str(alias / 'predictions.csv'),
                                      'protocol': str(alias / 'protocol.json')}
        self.rows.append(row)
        with self.assertRaisesRegex(ValueError, 'canonical actual attempt'):
            self.run_request(attempt_ids=['alias'])


if __name__ == '__main__':
    unittest.main()
