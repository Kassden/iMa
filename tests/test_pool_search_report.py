import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from scripts.report_pool_search import INPUTS, LABELS, render, render_inputs, write_report


def seal(root, ready=True):
    mapping = {str((root / name).resolve()): hashlib.sha256((root / name).read_bytes()).hexdigest()
               for name in INPUTS}
    source = root / 'source.txt'
    source.write_text('frozen source')
    receipt = {'ready_for_report': ready,
               'source_sha256': {str(source.resolve()): hashlib.sha256(source.read_bytes()).hexdigest()},
               'output_sha256': mapping}
    (root / 'result-provenance.json').write_text(json.dumps(receipt))


def fixture(root):
    (root / 'evaluation').mkdir()
    recipe = {'pair': 'standalone_pair', 'method': 'geometric', 'benter_weight': .15, 'temperature': .85}
    search = {'grid_count': 588, 'selected_recipe': recipe,
              'baseline_recipe': dict(recipe, pair='original_pair', method='arithmetic', benter_weight=.6, temperature=1),
              'search_best_loss': 1.1, 'confirmation_candidate_loss': 1.2, 'confirmation_baseline_loss': 1.3,
              'confirmation_delta': -.1, 'adopted': True, 'adopted_model': 'candidate_pool',
              'policy': 'Already inspected exploratory evaluation', 'downstream_calibration_policy': 'Final window only',
              'splits': {'search': {'races': 89, 'meetings': 30, 'date_min': '2026-01-14', 'date_max': '2026-04-29'},
                         'confirmation': {'races': 34, 'meetings': 10, 'date_min': '2026-05-03', 'date_max': '2026-06-07'},
                         'calibration': {'races': 97, 'meetings': 10, 'date_min': '2026-06-10', 'date_max': '2026-07-15'}}}
    (root / 'search.json').write_text(json.dumps(search))
    pools = {pool: {'nsettled': 78, 'stake_hkd': 780, 'gross_hkd': 390, 'profit_hkd': -390, 'roi': -.5}
             for pool in ('WIN', 'PLACE', 'QIN', 'QPL', 'TRI', 'TIERCE', 'FIRST4', 'QUARTET')}
    summary = {name: {'race_log_loss': 2.2 if name == 'candidate_pool' else 2.1,
                      'stake_hkd': 6240, 'gross_hkd': 3120, 'profit_hkd': -3120, 'roi': -.5,
                      'by_pool': pools} for name in LABELS}
    (root / 'evaluation/summary.json').write_text(json.dumps(summary))
    for name in ('provenance.json', 'calibration.json'):
        (root / 'evaluation' / name).write_text('{}')
    rows = []
    for family in ('baseline_pool', 'candidate_pool'):
        raw = []
        for race in range(1, 10):
            for horse in range(1, 13):
                identity = {'race_no': race, 'horse_no': horse, 'horse_name': f'Horse {horse}'}
                raw.append(identity | {'model_probability': 1 / 12})
                rows.append(identity | {'family': family, 'model': family, 'rank': horse,
                                        'pcombined': 1 / 12, 'pplace': .25, 'quoted_ev_hkd': -1.25})
        pd.DataFrame(raw).to_csv(root / f'evaluation/{family}-tomorrow-runners.csv', index=False)
    pd.DataFrame(rows).to_csv(root / 'evaluation/tomorrow-market-blends.csv', index=False)
    seal(root)


class PoolSearchReportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        fixture(self.root)

    def test_valid_report_and_write(self):
        report = render(self.root)
        self.assertIn('588 pool combinations, not 588 model fits', report)
        self.assertIn('worse than baseline', report)
        self.assertIn('not a recommendation', report)
        self.assertIn('not a new untouched holdout', report)
        for label in LABELS.values():
            self.assertIn(label, report)
        self.assertIn('108 known snapshot WIN EVs', report)
        self.assertIn('108 negative', report)
        self.assertEqual(report.count('| WIN |'), 2)
        output = self.root / 'report.md'
        write_report(self.root, output)
        self.assertEqual(output.read_text(), report)

    def test_changed_inputs_refuse_write(self):
        for name in ('search.json', 'evaluation/summary.json', 'evaluation/calibration.json',
                     'evaluation/provenance.json', 'evaluation/tomorrow-market-blends.csv', 'source.txt'):
            with self.subTest(name=name):
                path = self.root / name
                before = path.read_bytes()
                path.write_bytes(before + b'\n')
                output = self.root / 'report.md'
                with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                    write_report(self.root, output)
                self.assertFalse(output.exists())
                path.write_bytes(before)

    def test_bad_hashes_and_missing_binding(self):
        proof_path = self.root / 'result-provenance.json'
        original = proof_path.read_bytes()
        for change in ('short', 'wrong', 'missing', 'conflict'):
            with self.subTest(change=change):
                proof = json.loads(original)
                key = str((self.root / 'search.json').resolve())
                if change == 'short':
                    proof['output_sha256'][key] = 'abc'
                elif change == 'wrong':
                    proof['output_sha256'][key] = '0' * 64
                elif change == 'missing':
                    del proof['output_sha256'][key]
                else:
                    proof['source_sha256'][key] = '0' * 64
                proof_path.write_text(json.dumps(proof))
                with self.assertRaises(ValueError):
                    render(self.root)
        proof_path.write_bytes(original)

    def test_unready_receipt(self):
        seal(self.root, ready=False)
        with self.assertRaisesRegex(ValueError, 'not ready'):
            render(self.root)

    def test_relative_receipt_paths(self):
        path = self.root / 'result-provenance.json'
        proof = json.loads(path.read_text())
        base = self.root.parent.resolve()
        for key in ('source_sha256', 'output_sha256'):
            proof[key] = {os.path.relpath(name, base): digest for name, digest in proof[key].items()}
        path.write_text(json.dumps(proof))
        with patch('os.getcwd', return_value=str(base)):
            self.assertIn('# Preseason Pool Search Results', render(self.root))

    def test_nonfinite_json_metrics(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=value):
                path = self.root / 'evaluation/summary.json'
                summary = json.loads(path.read_text())
                summary['candidate_pool']['race_log_loss'] = value
                path.write_text(json.dumps(summary))
                seal(self.root)
                with self.assertRaisesRegex(ValueError, 'Non-finite'):
                    render(self.root)

    def test_nonfinite_forecast_probabilities(self):
        path = self.root / 'evaluation/tomorrow-market-blends.csv'
        original = path.read_bytes()
        for value in (float('nan'), float('inf'), -.1, 1.1):
            with self.subTest(value=value):
                frame = pd.read_csv(path)
                frame.loc[0, 'pcombined'] = value
                frame.to_csv(path, index=False)
                seal(self.root)
                with self.assertRaisesRegex(ValueError, 'probabilities'):
                    render(self.root)
                path.write_bytes(original)

    def test_unknown_and_positive_quote_ev(self):
        path = self.root / 'evaluation/tomorrow-market-blends.csv'
        frame = pd.read_csv(path)
        frame.loc[(frame.family == 'candidate_pool') & (frame.horse_no == 1), 'quoted_ev_hkd'] = float('nan')
        frame.loc[(frame.family == 'candidate_pool') & (frame.horse_no == 2), 'quoted_ev_hkd'] = .5
        frame.to_csv(path, index=False)
        seal(self.root)
        report = render(self.root)
        self.assertIn('| unknown |', report)
        self.assertIn('9 positive', report)
        self.assertIn('9 unknown', report)
        self.assertIn('does not guarantee an edge', report)

    def test_overlapping_windows_rejected(self):
        path = self.root / 'search.json'
        search = json.loads(path.read_text())
        search['splits']['confirmation']['date_min'] = search['splits']['search']['date_max']
        path.write_text(json.dumps(search))
        seal(self.root)
        with self.assertRaisesRegex(ValueError, 'ordered and disjoint'):
            render(self.root)

    def test_adoption_inconsistency_rejected(self):
        path = self.root / 'search.json'
        search = json.loads(path.read_text())
        search['adopted_model'] = 'baseline_pool'
        path.write_text(json.dumps(search))
        seal(self.root)
        with self.assertRaisesRegex(ValueError, 'Adoption decision'):
            render(self.root)

    def test_change_during_render_preserves_existing_output(self):
        output = self.root / 'report.md'
        output.write_text('previous report')

        def changed(inputs):
            report = render_inputs(inputs)
            path = self.root / 'evaluation/summary.json'
            path.write_bytes(path.read_bytes() + b'\n')
            return report

        with patch('scripts.report_pool_search.render_inputs', side_effect=changed):
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                write_report(self.root, output)
        self.assertEqual(output.read_text(), 'previous report')

    def test_report_cannot_overwrite_input(self):
        with self.assertRaisesRegex(ValueError, 'must not overwrite'):
            write_report(self.root, self.root / 'search.json')


if __name__ == '__main__':
    unittest.main()
