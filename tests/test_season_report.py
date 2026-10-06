import hashlib
import json
import os

import pandas as pd
import pytest

from scripts.report_season_2026 import (
    MODEL_LABELS, combined_market_section, hindsight_win_ev_section, ranked_predictions, stable_runner_order,
)


@pytest.mark.parametrize('verified,ceiling,expected', [(True, False, -5.0),
                                                     (False, False, None),
                                                     (True, True, None)])
def test_ranked_all_runners_and_safe_quote_units(tmp_path, verified, ceiling, expected):
    for directory in ('evaluation', 'models', 'official'):
        (tmp_path / directory).mkdir()
    (tmp_path / 'models/frozen-candidates.json').write_text(json.dumps({'candidates': {'pool': {}}}))
    (tmp_path / 'evaluation/summary.json').write_text(json.dumps({
        'pool': {'order_exponents': {'second': 1.0, 'third': 1.0}}}))
    pd.DataFrame([{'race_id': 'r1', 'race_no': 1, 'date': '2026-10-07', 'venue': 'HV',
                   'horse_no': n, 'horse_name': f'Horse {n}', 'model_probability': 1 / 8}
                  for n in range(1, 9)]).to_csv(tmp_path / 'evaluation/pool-tomorrow-runners.csv', index=False)
    (tmp_path / 'official/odds-quotes.json').write_text(json.dumps([{
        'meeting_date': '2026-10-07', 'venue': 'HV', 'race_no': 1, 'pool': 'WIN',
        'combination': '1', 'odds_value_raw': '4', 'odds_value_numeric': 4,
        'retrieved_at_utc': '2026-10-06T12:00:00Z', 'source_url': 'official',
        'possible_display_ceiling': ceiling}]))
    (tmp_path / 'official/odds-unit-semantics.json').write_text(json.dumps({
        'pool_unit_conversion': {'WIN': {'displayed_odds_to_D10_factor': 10,
            'conversion_status': 'verified_test' if verified else 'unknown'}}}))
    result = ranked_predictions(tmp_path)
    assert len(result) == 8
    assert result.horse_no.tolist() == list(range(1, 9))
    assert result['rank'].tolist() == list(range(1, 9))
    assert result.pplace.tolist() == pytest.approx([3 / 8] * 8)
    assert result.pwin.sum() == pytest.approx(1)
    if expected is None:
        assert pd.isna(result.iloc[0].win_ev_hkd_per_10)
    else:
        assert result.iloc[0].win_ev_hkd_per_10 == pytest.approx(expected)
    assert result.iloc[1:].win_ev_hkd_per_10.isna().all()


@pytest.mark.parametrize('probability_column', ['combinedp', 'pcombined'])
def test_optional_combined_market_section(tmp_path, probability_column, monkeypatch):
    assert combined_market_section(tmp_path) == []
    pd.DataFrame([{'model': 'pool', 'race_no': 1, 'horse_no': n,
                   probability_column: p, 'pplace': 1.0, 'quoted_odds_raw': '1.5',
                   'quoted_ev_hkd': p * 15 - 10, 'quote_status': 'indicative_snapshot_not_final'}
                  for n, p in [(1, .2), (2, .3), (3, .5)]]).to_csv(
                      tmp_path / 'tomorrow-market-blends.csv', index=False)
    write_combined_proof(tmp_path)
    monkeypatch.chdir(tmp_path.parent)
    proof_path = tmp_path / 'tomorrow-market-blend-provenance.json'
    proof = json.loads(proof_path.read_text())
    proof['source_sha256'] = {os.path.relpath(name): digest for name, digest in proof['source_sha256'].items()}
    proof['output_sha256'] = {os.path.relpath(name): digest for name, digest in proof['output_sha256'].items()}
    proof_path.write_text(json.dumps(proof))
    rendered = '\n'.join(combined_market_section(tmp_path))
    assert 'not a validated pre-off betting strategy' in rendered
    assert rendered.index('#3') < rendered.index('#2') < rendered.index('#1')
    assert 'Top pick snapshot WIN EV HKD/10' in rendered
    assert '| 1.5 | -2.50 |' in rendered
    assert '3 are negative, with maximum HKD-2.5000' in rendered
    assert 'not treated as a confidence-qualified betting edge' in rendered


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


@pytest.mark.parametrize('change', ['not_ready', 'csv', 'calibration.json', 'summary.json',
                                  'provenance.json', 'missing_proof', 'missing_binding'])
def test_combined_market_rejects_unready_or_stale_proof(tmp_path, change):
    path = tmp_path / 'tomorrow-market-blends.csv'
    pd.DataFrame([{'model': 'pool', 'race_no': 1, 'horse_no': 1,
                   'combinedp': 1.0, 'pplace': 1.0}]).to_csv(path, index=False)
    write_combined_proof(tmp_path)
    proof_path = tmp_path / 'tomorrow-market-blend-provenance.json'
    if change == 'missing_proof':
        proof_path.unlink()
    elif change in ('not_ready', 'missing_binding'):
        proof = json.loads(proof_path.read_text())
        if change == 'not_ready':
            proof['ready_for_report'] = False
        else:
            proof['source_sha256'].pop(str((tmp_path / 'calibration.json').resolve()))
        proof_path.write_text(json.dumps(proof))
    else:
        changed = path if change == 'csv' else tmp_path / change
        changed.write_text(changed.read_text() + '\n')
    with pytest.raises(ValueError, match='Combined-market'):
        combined_market_section(tmp_path)


def test_hindsight_win_ev_is_distinct_from_realized_profit():
    scores = {'pool': {'hindsight_approximate_win_ev_hkd': -12.5,
                       'by_pool': {'WIN': {'nsettled': 78, 'stake_hkd': 780, 'profit_hkd': 45}}},
              'market_final_odds_hindsight': {}}
    rendered = '\n'.join(hindsight_win_ev_section(scores, {'pool': 'Pool'}))
    assert '| Pool | 78 | 780.00 | -12.50 | 45.00 |' in rendered
    assert 'not actionable pre-off EV' in rendered
    assert 'market_final_odds_hindsight' not in rendered
    assert hindsight_win_ev_section({}, {}) == []


@pytest.mark.parametrize('column', ['model_probability', 'combinedp'])
def test_top_picks_ignore_float_noise_and_input_order(column):
    frame = pd.DataFrame({'horse_no': ['10', '2', '1'],
                          column: [.25 + 1e-16, .25, .25 - 1e-16]})
    for race in (frame, frame.iloc[::-1]):
        assert stable_runner_order(race, column).horse_no.tolist() == ['1', '2', '10']


def test_human_labels_cover_all_ten_variants_without_changing_ids():
    families = {'benter_conditional_logit', 'boosted', 'pool', 'gaussian_probit'}
    assert set(MODEL_LABELS) == families | {name + '_market_blend_hindsight' for name in families} | {
        'market_final_odds_hindsight', 'market_calibrated_hindsight'}
    assert MODEL_LABELS['benter_conditional_logit_market_blend_hindsight'] == 'Benter + market (hindsight)'
    assert MODEL_LABELS['market_final_odds_hindsight'] == 'Market (final odds, hindsight)'
    assert all('_' not in label for label in MODEL_LABELS.values())
