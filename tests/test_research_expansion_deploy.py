"""Read-only launcher preflight tests with explicitly mocked official evidence."""
import contextlib
import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from ima.dataset_registry import _digest, _keys, file_sha256


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / 'deploy/systemd/ima-research-expansion-supervisor'


class ResearchExpansionDeployTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.release = self.base / 'release'
        self.campaign = self.base / 'campaigns/agentic_v6_mock'
        self.dependencies = self.base / 'dependencies'
        self.snapshot = self.campaign / 'inputs/snapshot'
        self.registry = self.campaign / 'registry'
        for path in (self.release, self.dependencies, self.snapshot, self.registry):
            path.mkdir(parents=True)
        self.revision = '5b63862360f5380b6f247bf1c4da0a0854528aa6'
        (self.release / 'REVISION').write_text(self.revision)
        self.python = self.dependencies / 'python'
        self.python.touch()
        self.frame = pd.DataFrame({
            'race_id': ['mock-race', 'mock-race'],
            'horse_id': ['HK_2020_A001', 'HK_2020_A002'],
            'source': ['official:hkjc-results'] * 2,
            'source_url': ['https://racing.hkjc.com/en-us/local/information/localresults'] * 2,
            'source_body_hash': ['a' * 64] * 2,
        })
        self.frame.to_parquet(self.snapshot / 'runners.parquet', index=False)
        (self.snapshot / 'events.jsonl').write_text('')
        self.source = {
            'source_policy': 'HKJC-only; raw replay; no third-party values',
            'rows': 2, 'races': 1,
            'files': {name: file_sha256(self.snapshot / name)
                      for name in ('runners.parquet', 'events.jsonl')},
        }
        self.raw = {'manifest_id': 'mock-official-corpus'}
        self.values = json.loads((ROOT / 'config/agentic_research_expansion.json').read_text())
        self.values.update(dataset_registry_path=str(self.registry),
                           official_snapshot_path=str(self.snapshot))
        self.values.pop('reference_campaign_dir', None)
        self.config = self.campaign / 'config.json'
        self.env = {
            'IMA_V6_RELEASE': str(self.release), 'IMA_V6_REVISION': self.revision,
            'IMA_V6_CAMPAIGN': str(self.campaign), 'IMA_V6_DEPENDENCIES': str(self.dependencies),
            'IMA_V6_CONFIG': str(self.config), 'IMA_V6_PYTHON': str(self.python),
            'IMA_V6_RESEARCH_POLICY': 'expansion_v6',
        }
        self.seal()

    def seal(self):
        source_bytes = json.dumps(self.source).encode()
        (self.snapshot / 'manifest.json').write_bytes(source_bytes)
        raw_bytes = json.dumps(self.raw).encode()
        (self.snapshot / 'raw_manifest.json').write_bytes(raw_bytes)
        identity = {
            'source_manifest': hashlib.sha256(source_bytes).hexdigest(),
            'source_files': self.source['files'],
            'raw_manifest': hashlib.sha256(raw_bytes).hexdigest(),
            'predictor_catalog_sha256': _digest({}),
            'eligible_categorical_predictors_sha256': _digest([]),
        }
        dataset_id = 'dataset-' + _digest(identity)
        self.artifact = self.registry / 'datasets' / dataset_id
        self.artifact.mkdir(parents=True, exist_ok=True)
        self.frame.to_parquet(self.artifact / 'features.parquet', index=False)
        (self.artifact / 'protocol.json').write_text('{}')
        (self.artifact / 'source_manifest.json').write_bytes(source_bytes)
        (self.artifact / 'raw_manifest.json').write_bytes(raw_bytes)
        self.manifest = {
            'dataset_id': dataset_id, 'status': 'verified', 'identity': identity,
            'predictor_catalog': {}, 'eligible_categorical_predictors': [],
            'source_snapshot': str(self.snapshot), 'rows': 2, 'races': 1,
            'features_path': str(self.artifact / 'features.parquet'),
            'protocol_path': str(self.artifact / 'protocol.json'),
            'ordered_row_keys_sha256': _digest(_keys(self.frame)),
            'ordered_row_hashes_sha256': hashlib.sha256(
                pd.util.hash_pandas_object(self.frame, index=False).to_numpy().tobytes()).hexdigest(),
            'files': {path.name: file_sha256(path) for path in self.artifact.iterdir()
                      if path.name not in ('manifest.json', 'manifest.sha256')},
        }
        self.reseal_manifest()
        self.values.update(dataset_path=self.manifest['features_path'],
                           protocol_path=self.manifest['protocol_path'])

    def reseal_manifest(self):
        (self.artifact / 'manifest.json').write_text(json.dumps(self.manifest))
        (self.artifact / 'manifest.sha256').write_text(file_sha256(self.artifact / 'manifest.json'))

    def check(self):
        self.config.write_text(json.dumps(self.values))
        # Exercise the actual embedded preflight, changing only its fixed own-user root.
        code = LAUNCHER.read_text().split('"$IMA_V6_PYTHON" - <<\'PY\'\n', 1)[1].split('\nPY\n', 1)[0]
        code = code.replace("base = Path('/home/imaopt/research-v2')", f'base = Path({str(self.base)!r})')
        before = sorted(str(path.relative_to(self.base)) for path in self.base.rglob('*'))
        output = io.StringIO()
        with patch.dict(os.environ, self.env), contextlib.redirect_stdout(output):
            exec(compile(code, str(LAUNCHER), 'exec'), {})
        self.assertEqual(before, sorted(str(path.relative_to(self.base)) for path in self.base.rglob('*')))
        return output.getvalue()

    def test_mocked_production_accepted_without_state_creation(self):
        output = self.check()
        self.assertIn('preflight passed', output)
        self.assertIn('NOT a guaranteed hard billing bound', output)
        self.assertFalse((self.registry / 'locks').exists())

    def test_production_policy_and_numeric_caps(self):
        cases = {'planner_mode': ['fixture', 'local'], 'research_policy': ['discovery_v5'],
                 'max_total_cost_usd': [0, -1, 5.01, None, float('nan'), float('inf'), True],
                 'max_concurrent_trials': [27, 'auto', 26.5, True],
                 'cpu_thread_budget': [25, 24.5, True],
                 'memory_budget_gb_decimal': [100.01, float('nan')],
                 'ram_budget_gib': [100, float('inf')], 'max_active_preparations': [3],
                 'worker_max_tasks': [2], 'host_reserve_cpu_threads': [0, 24],
                 'host_reserve_ram_gib': [0, 100, float('nan')]}
        for field, failures in cases.items():
            original = self.values[field]
            for value in failures:
                with self.subTest(field=field, value=value):
                    self.values[field] = value
                    with self.assertRaises(SystemExit):
                        self.check()
            self.values[field] = original

    def test_raw_fixture_marker_rejected_despite_official_names_and_hashes(self):
        self.raw['fixture_only'] = True
        self.seal()
        with self.assertRaisesRegex(SystemExit, 'fixture provenance'):
            self.check()

    def test_synthetic_snapshot_rejected_despite_verified_registry(self):
        self.source['source_policy'] = 'HKJC-only synthetic test evidence'
        self.seal()
        with self.assertRaisesRegex(SystemExit, 'fixture provenance'):
            self.check()

    def test_corrupt_dataset_or_protocol_rejected(self):
        for name in ('features.parquet', 'protocol.json'):
            path = self.artifact / name
            original = path.read_bytes()
            path.write_bytes(original + b'corruption')
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                self.check()
            path.write_bytes(original)

    def test_resealed_inventory_lie_rejected(self):
        self.manifest['rows'] = 3
        self.reseal_manifest()
        with self.assertRaisesRegex(SystemExit, 'payload inventory mismatch'):
            self.check()

    def test_snapshot_corruption_rejected(self):
        (self.snapshot / 'events.jsonl').write_text('corrupted')
        with self.assertRaisesRegex(SystemExit, 'snapshot checksum mismatch'):
            self.check()

    def test_frozen_raw_manifest_corruption_rejected(self):
        (self.snapshot / 'raw_manifest.json').write_text('{}')
        with self.assertRaisesRegex(SystemExit, 'raw corpus manifest identity mismatch'):
            self.check()

    def test_manifest_checksum_corruption_rejected(self):
        (self.artifact / 'manifest.sha256').write_text('0' * 64)
        with self.assertRaisesRegex(ValueError, 'manifest checksum mismatch'):
            self.check()

    def test_renamed_fixture_snapshot_still_rejected(self):
        self.source['fixture_only'] = True
        self.seal()
        with self.assertRaisesRegex(SystemExit, 'fixture provenance'):
            self.check()

    def test_unbound_input_copy_rejected(self):
        copy = self.artifact / 'different.parquet'
        copy.write_bytes(b'unregistered data')
        self.values['dataset_path'] = str(copy)
        with self.assertRaisesRegex(SystemExit, 'differs from verified payload'):
            self.check()

    def test_renamed_bound_input_accepted(self):
        copy = self.artifact / 'renamed-input.parquet'
        copy.write_bytes((self.artifact / 'features.parquet').read_bytes())
        self.values['dataset_path'] = str(copy)
        self.assertIn('preflight passed', self.check())

    def test_frozen_snapshot_identity_mismatch_rejected(self):
        self.source['created_at'] = 'changed-after-registration'
        (self.snapshot / 'manifest.json').write_text(json.dumps(self.source))
        with self.assertRaisesRegex(SystemExit, 'snapshot identity mismatch'):
            self.check()

    def test_relocated_frozen_snapshot_matches_original_identity(self):
        self.manifest['source_snapshot'] = '/home/imaopt/acquisition/hkjc-20261002/snapshots/original'
        self.reseal_manifest()
        self.assertIn('preflight passed', self.check())
        self.assertEqual(self.manifest['source_snapshot'],
                         json.loads((self.artifact / 'manifest.json').read_text())['source_snapshot'])

    def test_relocated_snapshot_hash_mismatch_rejected(self):
        self.manifest['source_snapshot'] = '/home/imaopt/acquisition/hkjc-20261002/snapshots/original'
        self.reseal_manifest()
        (self.snapshot / 'events.jsonl').write_text('not the original payload')
        with self.assertRaisesRegex(SystemExit, 'snapshot checksum mismatch'):
            self.check()

    def test_foreign_upstream_provenance_rejected(self):
        self.manifest['source_snapshot'] = '/home/other-user/acquisition/snapshot'
        self.reseal_manifest()
        with self.assertRaisesRegex(SystemExit, 'own-user source root'):
            self.check()
