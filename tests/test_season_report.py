import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.report_season_2026 import (
    MODEL_LABELS, combined_market_section, hindsight_win_ev_section, ranked_predictions, stable_runner_order,
)


def write_combined_proof(evaluation):
    sources = {}
    for name in ('provenance.json', 'calibration.json', 'summary.json'):
        path = evaluation / name
        path.write_text('{}')
        sources[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    path = evaluation / 'tomorrow-market-blends.csv'
    proof = {'ready_for_report': True, 'source_sha256': sources,
             'output_sha256': {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()}}
    (evaluation / 'tomorrow-market-blend-provenance.json').write_text(json.dumps(proof))


class SeasonReportTests(unittest.TestCase):
    def test_ranked_all_runners_and_safe_quote_units(self):
        cases = [(True, False, -5.0), (False, False, None), (True, True, None)]
        for verified, ceiling, expected in cases:
            with self.subTest(verified=verified, ceiling=ceiling), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for name in ('evaluation', 'models', 'official'):
                    (root / name).mkdir()
                (root / 'models/frozen-candidates.json').write_text(json.dumps({'candidates': {'pool': {}}}))
                (root / 'evaluation/summary.json').write_text(json.dumps({
                    'pool': {'order_exponents': {'second': 1.0, 'third': 1.0}}}))
                pd.DataFrame([{'race_id': 'r1', 'race_no': 1, 'date': '2026-10-07', 'venue': 'HV',
                               'horse_no': n, 'horse_name': f'Horse {n}', 'model_probability': 1 / 8}
                              for n in range(1, 9)]).to_csv(root / 'evaluation/pool-tomorrow-runners.csv', index=False)
                (root / 'official/odds-quotes.json').write_text(json.dumps([{
                    'meeting_date': '2026-10-07', 'venue': 'HV', 'race_no': 1, 'pool': 'WIN',
                    'combination': '1', 'odds_value_raw': '4', 'odds_value_numeric': 4,
                    'retrieved_at_utc': '2026-10-06T12:00:00Z', 'source_url': 'official',
                    'possible_display_ceiling': ceiling}]))
                (root / 'official/odds-unit-semantics.json').write_text(json.dumps({
                    'pool_unit_conversion': {'WIN': {'displayed_odds_to_D10_factor': 10,
                        'conversion_status': 'verified_test' if verified else 'unknown'}}}))
                result = ranked_predictions(root)
                self.assertEqual(len(result), 8)
                self.assertEqual(result.horse_no.tolist(), list(range(1, 9)))
                self.assertEqual(result['rank'].tolist(), list(range(1, 9)))
                np.testing.assert_allclose(result.pplace.to_numpy(), [3 / 8] * 8)
                self.assertAlmostEqual(result.pwin.sum(), 1)
                if expected is None:
                    self.assertTrue(pd.isna(result.iloc[0].win_ev_hkd_per_10))
                else:
                    self.assertAlmostEqual(result.iloc[0].win_ev_hkd_per_10, expected)
                self.assertTrue(result.iloc[1:].win_ev_hkd_per_10.isna().all())

    def test_optional_combined_market_section(self):
        for probability_column in ('combinedp', 'pcombined'):
            with self.subTest(probability_column=probability_column), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.assertEqual(combined_market_section(root), [])
                pd.DataFrame([{'model': 'pool', 'race_no': 1, 'horse_no': n,
                               probability_column: p, 'pplace': 1.0, 'quoted_odds_raw': '1.5',
                               'quoted_ev_hkd': p * 15 - 10, 'quote_status': 'indicative_snapshot_not_final'}
                              for n, p in [(1, .2), (2, .3), (3, .5)]]).to_csv(
                                  root / 'tomorrow-market-blends.csv', index=False)
                write_combined_proof(root)
                proof_path = root / 'tomorrow-market-blend-provenance.json'
                proof = json.loads(proof_path.read_text())
                base = root.parent.resolve()
                proof['source_sha256'] = {os.path.relpath(name, base): digest
                                          for name, digest in proof['source_sha256'].items()}
                proof['output_sha256'] = {os.path.relpath(name, base): digest
                                          for name, digest in proof['output_sha256'].items()}
                proof_path.write_text(json.dumps(proof))
                with patch('os.getcwd', return_value=str(base)):
                    rendered = '\n'.join(combined_market_section(root))
                self.assertIn('not a validated pre-off betting strategy', rendered)
                self.assertLess(rendered.index('#3'), rendered.index('#2'))
                self.assertLess(rendered.index('#2'), rendered.index('#1'))
                self.assertIn('Top pick snapshot WIN EV HKD/10', rendered)
                self.assertIn('| 1.5 | -2.50 |', rendered)
                self.assertIn('3 are negative, with maximum HKD-2.5000', rendered)
                self.assertIn('not treated as a confidence-qualified betting edge', rendered)

    def test_combined_market_rejects_unready_or_stale_proof(self):
        changes = ('not_ready', 'csv', 'calibration.json', 'summary.json',
                   'provenance.json', 'missing_proof', 'missing_binding')
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path = root / 'tomorrow-market-blends.csv'
                pd.DataFrame([{'model': 'pool', 'race_no': 1, 'horse_no': 1,
                               'combinedp': 1.0, 'pplace': 1.0}]).to_csv(path, index=False)
                write_combined_proof(root)
                proof_path = root / 'tomorrow-market-blend-provenance.json'
                if change == 'missing_proof':
                    proof_path.unlink()
                elif change in ('not_ready', 'missing_binding'):
                    proof = json.loads(proof_path.read_text())
                    if change == 'not_ready':
                        proof['ready_for_report'] = False
                    else:
                        proof['source_sha256'].pop(str((root / 'calibration.json').resolve()))
                    proof_path.write_text(json.dumps(proof))
                else:
                    changed = path if change == 'csv' else root / change
                    changed.write_text(changed.read_text() + '\n')
                with self.assertRaisesRegex(ValueError, 'Combined-market'):
                    combined_market_section(root)

    def test_hindsight_win_ev_is_distinct_from_realized_profit(self):
        scores = {'pool': {'hindsight_approximate_win_ev_hkd': -12.5,
                           'by_pool': {'WIN': {'nsettled': 78, 'stake_hkd': 780, 'profit_hkd': 45}}},
                  'market_final_odds_hindsight': {}}
        rendered = '\n'.join(hindsight_win_ev_section(scores, {'pool': 'Pool'}))
        self.assertIn('| Pool | 78 | 780.00 | -12.50 | 45.00 |', rendered)
        self.assertIn('not actionable pre-off EV', rendered)
        self.assertNotIn('market_final_odds_hindsight', rendered)
        self.assertEqual(hindsight_win_ev_section({}, {}), [])

    def test_top_picks_ignore_float_noise_and_input_order(self):
        for column in ('model_probability', 'combinedp'):
            with self.subTest(column=column):
                frame = pd.DataFrame({'horse_no': ['10', '2', '1'],
                                      column: [.25 + 1e-16, .25, .25 - 1e-16]})
                for race in (frame, frame.iloc[::-1]):
                    self.assertEqual(stable_runner_order(race, column).horse_no.tolist(), ['1', '2', '10'])

    def test_human_labels_cover_all_ten_variants_without_changing_ids(self):
        families = {'benter_conditional_logit', 'boosted', 'pool', 'gaussian_probit'}
        self.assertEqual(set(MODEL_LABELS), families | {name + '_market_blend_hindsight' for name in families} | {
            'market_final_odds_hindsight', 'market_calibrated_hindsight'})
        self.assertEqual(MODEL_LABELS['benter_conditional_logit_market_blend_hindsight'], 'Benter + market (hindsight)')
        self.assertEqual(MODEL_LABELS['market_final_odds_hindsight'], 'Market (final odds, hindsight)')
        self.assertTrue(all('_' not in label for label in MODEL_LABELS.values()))


if __name__ == '__main__':
    unittest.main()
