"""Render receipt-bound pool search results without fitting or forecasting."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd


LABELS = {
    'baseline_pool': 'Original baseline pool',
    'candidate_pool': 'Search-selected candidate pool',
    'baseline_pool_market_blend_hindsight': 'Baseline + market (hindsight)',
    'candidate_pool_market_blend_hindsight': 'Candidate + market (hindsight)',
    'market_final_odds_hindsight': 'Market (final odds, hindsight)',
    'market_calibrated_hindsight': 'Market (temperature-calibrated, hindsight)',
}
INPUTS = ('search.json', 'evaluation/summary.json', 'evaluation/provenance.json',
          'evaluation/calibration.json', 'evaluation/tomorrow-market-blends.csv',
          'evaluation/baseline_pool-tomorrow-runners.csv',
          'evaluation/candidate_pool-tomorrow-runners.csv')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def finite_json(payload):
    if isinstance(payload, float) and not math.isfinite(payload):
        raise ValueError('Non-finite JSON value')
    if isinstance(payload, dict):
        for value in payload.values():
            finite_json(value)
    elif isinstance(payload, list):
        for value in payload:
            finite_json(value)
    return payload


def read_json(data):
    return finite_json(json.loads(data))


class VerifiedInputs:
    def __init__(self, root):
        self.root = Path(root)
        self.receipt_path = self.root / 'result-provenance.json'
        self.receipt_bytes = self.receipt_path.read_bytes()
        receipt = read_json(self.receipt_bytes)
        if receipt.get('ready_for_report') is not True:
            raise ValueError('Result provenance is not ready for report')
        self.hashes = {}
        for key in ('source_sha256', 'output_sha256'):
            mapping = receipt.get(key)
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError(f'Missing receipt hash map: {key}')
            for name, digest in mapping.items():
                path = Path(name).resolve()
                if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
                    raise ValueError(f'Invalid full SHA256: {name}')
                if path in self.hashes and self.hashes[path] != digest:
                    raise ValueError(f'Conflicting receipt hashes: {path}')
                self.hashes[path] = digest
        for name in INPUTS:
            if (self.root / name).resolve() not in self.hashes:
                raise ValueError(f'Receipt does not bind report input: {name}')
        self.verify()
        self.data = {}
        for name in INPUTS:
            data = (self.root / name).read_bytes()
            if hashlib.sha256(data).hexdigest() != self.hashes[(self.root / name).resolve()]:
                raise ValueError(f'Input changed during read: {name}')
            self.data[name] = data

    def verify(self):
        if self.receipt_path.read_bytes() != self.receipt_bytes:
            raise ValueError('Result provenance changed during report')
        for path, digest in self.hashes.items():
            if not path.is_file() or sha(path) != digest:
                raise ValueError(f'Receipt hash mismatch: {path}')


def table(headers, rows):
    def cell(value):
        return str(value).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |',
                      *['| ' + ' | '.join(cell(value) for value in row) + ' |' for row in rows]])


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Expected finite numeric metric')
    return value


def percent(value):
    return f'{100 * number(value):.2f}%'


def money(value):
    return f'{number(value):,.2f}'


def recipe(value):
    weight, temperature = number(value['benter_weight']), number(value['temperature'])
    if not isinstance(value['pair'], str) or value['method'] not in ('arithmetic', 'geometric'):
        raise ValueError('Invalid frozen pool recipe')
    if not 0 <= weight <= 1 or temperature <= 0:
        raise ValueError('Invalid recipe weight or temperature')
    return f"{value['pair']}; {value['method']}; Benter weight {weight:g}; temperature {temperature:g}"


def validate_forecasts(inputs):
    frame = pd.read_csv(io.BytesIO(inputs.data['evaluation/tomorrow-market-blends.csv']))
    required = {'family', 'model', 'race_no', 'horse_no', 'horse_name', 'rank', 'pcombined', 'pplace', 'quoted_ev_hkd'}
    if not required.issubset(frame):
        raise ValueError('Forecast schema is incomplete')
    if set(frame.family) != {'baseline_pool', 'candidate_pool'} or not frame.family.eq(frame.model).all():
        raise ValueError('Forecast families do not match the two frozen pools')
    raw_frames = {}
    for family in ('baseline_pool', 'candidate_pool'):
        live = frame[frame.family.eq(family)].copy()
        raw = pd.read_csv(io.BytesIO(inputs.data[f'evaluation/{family}-tomorrow-runners.csv']))
        for current, probabilities in ((live, ['pcombined', 'pplace']), (raw, ['model_probability'])):
            columns = ['race_no', 'horse_no', *probabilities]
            if not set(columns + ['horse_name']).issubset(current):
                raise ValueError('Runner forecast schema is incomplete')
            values = current[columns].to_numpy(dtype=float)
            if not np.isfinite(values).all() or not current[probabilities].ge(0).all().all() or not current[probabilities].le(1).all().all():
                raise ValueError('Forecast has non-finite or out-of-range probabilities')
            if len(current) != 108 or set(current.race_no) != set(range(1, 10)) or current.duplicated(['race_no', 'horse_no']).any():
                raise ValueError('Forecast requires 108 unique starters across nine races per family')
            if not np.equal(current[['race_no', 'horse_no']], np.floor(current[['race_no', 'horse_no']])).all().all():
                raise ValueError('Forecast runner identities must be integers')
            win_column = probabilities[0]
            if not np.allclose(current.groupby('race_no')[win_column].sum(), 1, rtol=0, atol=1e-6):
                raise ValueError('WIN probabilities do not sum to one')
        keys = ['race_no', 'horse_no', 'horse_name']
        if set(map(tuple, raw[keys].to_numpy())) != set(map(tuple, live[keys].to_numpy())):
            raise ValueError('Fundamental and market forecast identities differ')
        quoted = pd.to_numeric(live.quoted_ev_hkd, errors='raise')
        if np.isinf(quoted.to_numpy(dtype=float)).any():
            raise ValueError('Non-finite quoted EV')
        raw_frames[family] = raw
    return frame, raw_frames


def render_inputs(inputs):
    search = read_json(inputs.data['search.json'])
    summary = read_json(inputs.data['evaluation/summary.json'])
    calibration = read_json(inputs.data['evaluation/calibration.json'])
    if set(summary) != set(LABELS):
        raise ValueError('Summary must contain exactly six variants')
    if search['grid_count'] != 588:
        raise ValueError('Expected frozen grid of 588 combinations')
    adopted = search['adopted']
    if not isinstance(adopted, bool) or search['adopted_model'] != ('candidate_pool' if adopted else 'baseline_pool'):
        raise ValueError('Adoption decision and selected model disagree')
    candidate = number(search['confirmation_candidate_loss'])
    baseline = number(search['confirmation_baseline_loss'])
    if not math.isclose(number(search['confirmation_delta']), candidate - baseline, abs_tol=1e-10):
        raise ValueError('Confirmation delta does not match losses')
    rows, previous_end = [], None
    for name in ('search', 'confirmation', 'calibration'):
        split = search['splits'][name]
        first, last = pd.Timestamp(split['date_min']), pd.Timestamp(split['date_max'])
        if pd.isna(first) or pd.isna(last) or first > last or (previous_end is not None and previous_end >= first):
            raise ValueError('The three preseason windows must be ordered and disjoint')
        previous_end = last
        rows.append([name.title(), split['races'], split['meetings'], str(first.date()), str(last.date())])
    selected = search['adopted_model']
    lines = ['# Preseason Pool Search Results', '',
             '## Scope And Decision', '',
             '**588 pool combinations, not 588 model fits.** This report performs no retraining, no LLM calls and no wagering. '
             'The frozen component predictions are reused; only pool rules were searched.', '',
             f"Selected recipe: {recipe(search['selected_recipe'])}. Baseline: {recipe(search['baseline_recipe'])}.", '',
             f"Best search loss: {number(search['search_best_loss']):.9f}. Confirmation candidate loss: {candidate:.9f}; "
             f"baseline: {baseline:.9f}; candidate minus baseline: {candidate - baseline:+.9f}. "
             f"Protocol adoption decision: {'candidate selected' if adopted else 'baseline retained'}. "
             f"Forecast family: **{LABELS[selected]}**. This is a **provisional experiment only**; "
             'the original deployment was not changed.', '',
             '## Three Disjoint Preseason Windows', '',
             table(['Window', 'Races', 'Meetings', 'First date', 'Last date'], rows), '',
             'Search chooses the pool recipe; confirmation compares it with the fixed baseline; '
             'the final calibration window fits the downstream market blend and finish-order exponents. '
             'These date windows do not overlap. The September/October season was already viewed in the earlier report, '
             'so this is **exploratory, not a new untouched holdout or independent proof of improvement**.', '',
             f"The confirmation cohort has only **{search['splits']['confirmation']['races']} races**; "
             'this small comparison is not strong evidence of generalization or a betting edge.', '',
             f"Search policy: {search['policy']}. Calibration policy: {search['downstream_calibration_policy']}.", '',
             '## Season Comparison', '']
    rows = []
    for name in LABELS:
        value = summary[name]
        rows.append([LABELS[name], f"{number(value['race_log_loss']):.9f}", money(value['stake_hkd']),
                     money(value['gross_hkd']), money(value['profit_hkd']), percent(value['roi'])])
    lines += [table(['Variant', 'Win log loss', 'Stake HKD', 'Gross HKD', 'Net HKD', 'ROI'], rows), '']
    raw_candidate = number(summary['candidate_pool']['race_log_loss'])
    raw_baseline = number(summary['baseline_pool']['race_log_loss'])
    comparison = 'worse than' if raw_candidate > raw_baseline else 'no worse than'
    lines += [f"Candidate fundamental season loss **{raw_candidate:.9f}** is {comparison} baseline **{raw_baseline:.9f}**. "
              'Confirmation-based adoption is not a recommendation that the candidate is a stronger fundamental model. '
              'Lower log loss is better; final-odds market and blend rows are hindsight benchmarks, not validated pre-off strategies.', '',
              '## Candidate Returns By Pool', '']
    for name in ('candidate_pool', 'candidate_pool_market_blend_hindsight'):
        rows = [[pool, value['nsettled'], money(value['stake_hkd']), money(value['gross_hkd']),
                 money(value['profit_hkd']), percent(value['roi'])] for pool, value in summary[name]['by_pool'].items()]
        lines += [f'### {LABELS[name]}', '', table(['Pool', 'Settled', 'Stake HKD', 'Gross HKD', 'Net HKD', 'ROI'], rows), '']
    rows, zero_fundamental = [], []
    for family in ('baseline_pool', 'candidate_pool'):
        blend = calibration[family]['blend']
        a, b = number(blend['fundamental_weight']), number(blend['market_weight'])
        rows.append([LABELS[family], f'{a:.9f}', f'{b:.9f}', calibration[family]['calibration_races']])
        if a == 0:
            zero_fundamental.append(LABELS[family])
    lines += ['## Market Blend Calibration', '',
              'Combined WIN probability is proportional to fundamental probability raised to a, '
              'times market probability raised to b, normalized within each race.', '',
              table(['Pool', 'Fundamental exponent a', 'Market exponent b', 'Calibration races'], rows), '']
    if zero_fundamental:
        lines += [f"Zero fundamental weight for {', '.join(zero_fundamental)} means a **pure calibrated-market forecast, "
                  'not a model edge**: those blends discard the fundamental prediction. '
                  'A zero exponent is a calibration result, not evidence that the selected pool adds signal beyond the market.', '']
    calibration_sizes = ', '.join(str(n) for n in sorted({calibration[family]['calibration_races']
                                                         for family in ('baseline_pool', 'candidate_pool')}))
    lines += [f"**Not a like-for-like coefficient comparison:** this study fits the market blends on {calibration_sizes} races "
              'in the final disjoint calibration window. The earlier season report fitted the original pool blend on 220 races '
              'and reported fundamental weight approximately 0.0721. The fitting populations differ; comparing those weights '
              'does not isolate a recipe improvement or prove that a previously established model edge disappeared.', '']
    forecasts, raw_frames = validate_forecasts(inputs)
    live = forecasts[forecasts.family.eq(selected)]
    raw = raw_frames[selected]
    rows = []
    for race_no, race in live.groupby('race_no', sort=True):
        top = race.assign(_p=race.pcombined.round(12)).sort_values(['_p', 'horse_no'], ascending=[False, True]).iloc[0]
        fundamental = raw[raw.race_no.eq(race_no)].assign(_p=lambda frame: frame.model_probability.round(12)).sort_values(
            ['_p', 'horse_no'], ascending=[False, True]).iloc[0]
        ev = top.quoted_ev_hkd
        rows.append([int(race_no), f"#{int(top.horse_no)} {top.horse_name}", percent(top.pcombined), percent(top.pplace),
                     money(ev) if pd.notna(ev) else 'unknown',
                     f"#{int(fundamental.horse_no)} {fundamental.horse_name}", percent(fundamental.model_probability)])
    known = pd.to_numeric(live.quoted_ev_hkd, errors='raise').dropna()
    lines += ['## Selected October 7 Snapshot Forecast', '']
    if 'quote_retrieved_at_utc' in live:
        timestamps = pd.to_datetime(live.quote_retrieved_at_utc.dropna(), utc=True, errors='raise').dropna()
        if len(timestamps):
            first = timestamps.min().tz_convert('Asia/Hong_Kong')
            last = timestamps.max().tz_convert('Asia/Hong_Kong')
            lines += [f"WIN quote snapshot captured **{first.strftime('%Y-%m-%d %H:%M:%S')} to "
                      f"{last.strftime('%Y-%m-%d %H:%M:%S')} HKT**, from the selected family's quote retrieval rows. "
                      'These are capture times, not a claim that prices remained unchanged until race time.', '']
    lines += [
              table(['Race', 'Combined WIN pick', 'Combined WIN p', 'Paid PLACE p', 'WIN EV HKD/10',
                     'Fundamental WIN pick', 'Fundamental WIN p'], rows), '',
              f"{len(known)} known snapshot WIN EVs for the selected family: {int(known.lt(0).sum())} negative, "
              f"{int(known.gt(0).sum())} positive, {int(known.eq(0).sum())} zero; {int(live.quoted_ev_hkd.isna().sum())} unknown. "
              'EV is expected net profit at the captured quote, p x gross payout per HKD10 minus HKD10. '
              '**Positive snapshot EV does not guarantee an edge; negative EV is not a value recommendation.** '
              'Missing quotes remain unknown, not zero. Prices and fields can change. Fundamental-only optimism is not confidence-qualified betting evidence.', '',
              'Settlement uses exact official dividends and fixed HKD10 straight tickets; published dividends already reflect takeout. '
              'ROI is realized net divided by settled stake. Missing settlements are not losing tickets. '
              'There is no compounding or guaranteed bankroll growth. Fourth-stage FIRST4/QUARTET probabilities reuse the third-position exponent.', '',
              '## Reproduction', '', '```sh',
              '.venv/bin/python -m scripts.report_pool_search --root ' + str(inputs.root), '```', '',
              'All source and output SHA256 entries in `result-provenance.json` were verified, including every report input. '
              'Any stale receipt or changed input prevents writing.', '']
    return '\n'.join(lines)


def render(root):
    inputs = VerifiedInputs(root)
    report = render_inputs(inputs)
    inputs.verify()
    return report


def write_report(root, output):
    inputs = VerifiedInputs(root)
    report = render_inputs(inputs)
    output = Path(output)
    if output.resolve() in inputs.hashes or output.resolve() == inputs.receipt_path.resolve():
        raise ValueError('Report output must not overwrite a receipt-bound input')
    inputs.verify()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('artifacts/pool-search-20261007'))
    parser.add_argument('--output', type=Path, default=Path('docs/PRESEASON_POOL_SEARCH_RESULTS.md'))
    args = parser.parse_args()
    write_report(args.root, args.output)
    print(args.output)


if __name__ == '__main__':
    main()
