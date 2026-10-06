"""Prepare past-only queries and evaluate frozen 2026/27 season predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from ima.modeling import MarketBlend, TemperatureCalibrator, race_log_loss
from ima.pools import SUPPORTED_POOLS, fit_order_exponents, rank_combinations, paid_place_count
from ima.season_evaluation import evaluate_season
from ima.season_features import build_season_features
from ima.rich_features import prepare_rich_runner_dataset
from ima.speed_features import build_speed_features


ROOT = Path('artifacts/race-readiness-20261007')


def quote_lookup(official_dir):
    path = official_dir / 'odds-quotes.json'
    units_path = official_dir / 'odds-unit-semantics.json'
    if not path.exists() or not units_path.exists():
        return {}, {}
    units = json.loads(units_path.read_text())['pool_unit_conversion']
    quotes = {}
    for quote in json.loads(path.read_text()):
        pool = {'TRIO': 'TRI'}.get(quote['pool'], quote['pool'])
        runners = tuple(str(int(value)) for value in quote['combination'].replace(',', '/').split('/'))
        if pool not in ('TIERCE', 'QUARTET'):
            runners = tuple(sorted(runners, key=int))
        key = (quote['meeting_date'], quote['venue'], int(quote['race_no']), pool, runners)
        if key in quotes:
            raise ValueError('Duplicate official quote identity')
        quotes[key] = quote
    return quotes, units


def ticket_quote(ticket, race, pool, quotes, units):
    runners = tuple(ticket.runners)
    if pool not in ('TIERCE', 'QUARTET'):
        runners = tuple(sorted(runners, key=int))
    date = str(pd.Timestamp(race.date.iloc[0]).date())
    key = (date, race.venue.iloc[0], int(race.race_no.iloc[0]), pool, runners)
    quote = quotes.get(key)
    output = {'quoted_ev_hkd': None, 'quote_status': 'not_available', 'quoted_gross_hkd_per_10': None}
    if quote is None:
        return output
    output.update(quoted_odds_raw=quote['odds_value_raw'], quote_retrieved_at_utc=quote['retrieved_at_utc'],
                  quote_source_last_update=quote.get('source_last_update'), quote_source_url=quote['source_url'])
    unit = units.get('TRIO' if pool == 'TRI' else pool, {})
    factor = unit.get('displayed_odds_to_D10_factor')
    value = quote.get('odds_value_numeric')
    if quote.get('possible_display_ceiling') or value == 999:
        output['quote_status'] = 'possible_display_ceiling_no_point_ev'
    elif factor is None or not unit.get('conversion_status', '').startswith('verified_'):
        output['quote_status'] = 'unit_unverified_no_point_ev'
    elif value is None or not np.isfinite(value) or value <= 0:
        output['quote_status'] = 'non_numeric_no_point_ev'
    else:
        gross = float(value) * factor
        output.update(quote_status='indicative_snapshot_not_final', quoted_gross_hkd_per_10=gross,
                      quoted_ev_hkd=float(ticket.probability * gross - 10))
    return output


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False, default=str) + '\n')


def runner_rows(records, *, card=False):
    rows = []
    for race in records:
        for runner in race['runners']:
            row = {k: race.get(k) for k in ('race_no', 'venue', 'distance', 'course', 'going', 'race_class', 'prize')}
            row.update({key: value for key, value in runner.items()
                        if not isinstance(value, (list, dict))})
            row.update(date=race['race_date'],
                       race_id=f"HKJC:{race['race_date']}:{race['venue']}:R{race['race_no']}",
                       horse_id=runner['horse_page_id'], result=runner.get('place'),
                       jockey_name=runner.get('jockey'), trainer_name=runner.get('trainer'),
                       source='official:hkjc-results', field_size=len(race['runners']))
            if card:
                row['actual_weight'] = runner.get('net_carried_weight_lbs')
                if row['actual_weight'] is None:
                    row['actual_weight'] = runner.get('weight')
                row['horse_rating'] = runner.get('rating')
                row['horse_age'] = runner.get('age')
                row['horse_country'] = runner.get('horse_country', runner.get('country'))
            rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError('No official runner records')
    frame['date'] = pd.to_datetime(frame.date)
    return frame


def settlement_rows(records):
    rows = []
    for race in records:
        race_id = f"HKJC:{race['race_date']}:{race['venue']}:R{race['race_no']}"
        for pool, values in race['dividends'].items():
            canonical = 'TRI' if pool == 'TRIO' else pool
            if canonical not in SUPPORTED_POOLS:
                raise ValueError(f'Unsupported settlement pool: {canonical}')
            cutoff = {'WIN': 1, 'QIN': 2, 'TRI': 3, 'TIERCE': 3, 'FIRST4': 4, 'QUARTET': 4}.get(
                canonical, paid_place_count(len(race['runners'])))
            paid_positions = [runner['place'] for runner in race['runners']
                              if runner.get('place') is not None and runner['place'] <= cutoff]
            paid_dead_heat = len(set(paid_positions)) != len(paid_positions)
            if not values.get('records') and values.get('status') == 'unavailable':
                continue
            children = values.get('records', []) or [{'combination': [], 'status': values.get('status')}]
            for record in children:
                status = record.get('status') if values.get('status') in {'payable', 'mixed'} else values.get('status')
                dividend = None
                if status == 'payable':
                    if record.get('currency') != 'HKD':
                        raise ValueError('Settlement requires an explicit HKD currency')
                    normalized = record.get('dividend_per_hkd_10_decimal')
                    if normalized is not None:
                        dividend = float(Decimal(str(normalized)))
                    elif record.get('unit_stake_hkd') is not None:
                        unit = float(record['unit_stake_hkd'])
                        if not np.isfinite(unit) or unit <= 0:
                            raise ValueError('Invalid dividend unit')
                        dividend = float(record['dividend_hkd']) * 10 / unit
                    else:
                        raise ValueError('Unknown dividend unit')
                rows.append({'race_id': race_id, 'pool': canonical,
                    'winning_combination': record['combination'],
                    'dividend_hkd_per_10': dividend,
                    'status': status, 'dead_heat': paid_dead_heat})
    return rows


def target_frame(frame):
    output = frame.copy()
    output['result'] = pd.to_numeric(output.result, errors='coerce')
    output['target_win'] = output.result.eq(1).astype(int)
    counts = output.groupby('race_id').target_win.transform('sum')
    if not counts.eq(1).all():
        raise ValueError('Evaluation requires exactly one winning runner per race')
    output['target_probability'] = output.target_win.astype(float)
    return output


def market_probability(frame):
    odds = pd.to_numeric(frame.win_odds, errors='coerce')
    if odds.isna().any() or not np.isfinite(odds).all() or not odds.gt(0).all():
        raise ValueError('Missing full-field market odds')
    raw = 1.0 / odds
    return (raw / raw.groupby(frame.race_id).transform('sum')).to_numpy()


def enrich_official_history(history, official_dir):
    """Add full race fields; use horse forms only to fill documented input fields."""
    history = history.copy()
    extra_path = official_dir / 'history-races.json'
    if extra_path.exists():
        coverage = json.loads((official_dir / 'history-race-coverage.json').read_text())
        if not coverage.get('ready_for_freeze', coverage.get('complete', False)):
            raise ValueError('Historical backfill is still running or incomplete')
        extra = runner_rows(json.loads(extra_path.read_text()))
        extra = extra[~extra.race_id.isin(history.race_id)]
        history = pd.concat([history, extra], ignore_index=True)
    return history


def enrich_form_inputs(frame, official_dir):
    path = official_dir / 'horse_histories.json'
    if not path.exists():
        return frame
    frame = frame.copy()
    inputs = {}
    for horse in json.loads(path.read_text()):
        for record in horse.get('form_records', []):
            key = (record.get('race_date'), record.get('venue'), record.get('race_no'), horse['horse_page_id'])
            previous = inputs.get(key)
            if previous is not None and any(previous.get(name) != record.get(name) for name in ('rating', 'gear')):
                raise ValueError(f'Conflicting official form input records: {key}')
            inputs[key] = record
    for index, row in frame.iterrows():
        record = inputs.get((str(row.date.date()), row.venue, row.race_no, row.horse_id))
        if record is None:
            continue
        for column, source in (('horse_rating', 'rating'), ('gear', 'gear')):
            if column not in frame or pd.isna(frame.at[index, column]):
                frame.at[index, column] = record.get(source)
                frame.at[index, column + '_source'] = 'official:horse-form:' + record.get('result_source_url', '')
    return frame


def enrich_archived_cards(frame, official_dir):
    flat_paths = [official_dir / (scope + '-runner-enrichment.json')
                  for scope in ('season', 'calibration')
                  if (official_dir / (scope + '-runner-enrichment.json')).exists()]
    if flat_paths:
        output = frame.copy()
        inputs = {}
        for flat_path in flat_paths:
            scope = flat_path.name.split('-')[0]
            coverage = json.loads((official_dir / (scope + '-enrichment-coverage.json')).read_text())
            finished_partial_calibration = scope == 'calibration' and coverage.get('acquisition_finished')
            if not coverage.get('complete') and not finished_partial_calibration:
                raise ValueError('Archived declaration enrichment is incomplete')
            for record in json.loads(flat_path.read_text()):
                key = (record['race_date'], record['venue'], record['race_no'], record['horse_page_id'])
                if key in inputs:
                    raise ValueError('Duplicate archived declaration identity')
                inputs[key] = record
        for index, row in output.iterrows():
            record = inputs.get((str(row.date.date()), row.venue, row.race_no, row.horse_id))
            if record is None:
                continue
            if record['horse_no'] != row.horse_no:
                raise ValueError('Archived declaration horse number mismatch')
            for name, alias in (('horse_rating', 'rating'), ('horse_age', 'age'), ('horse_country', 'country'), ('gear', 'gear')):
                value = record.get(name, record.get(alias))
                if name not in output or pd.isna(output.at[index, name]):
                    output.at[index, name] = value
                    output.at[index, name + '_source'] = record['source']['source_url']
        return output
    path = official_dir / 'season-racecards.json'
    if not path.exists():
        return frame
    coverage = json.loads((official_dir / 'season-racecards-coverage.json').read_text())
    if not coverage.get('complete'):
        raise ValueError('Archived season card acquisition is incomplete')
    cards = runner_rows(json.loads(path.read_text()), card=True)
    keys = ['race_id', 'horse_no', 'horse_id']
    if cards.duplicated(keys).any():
        raise ValueError('Duplicate archived card identity')
    inputs = cards.set_index(keys)
    output = frame.copy()
    for index, row in output.iterrows():
        key = tuple(row[name] for name in keys)
        if key not in inputs.index:
            continue
        card = inputs.loc[key]
        for name in ('horse_rating', 'horse_age', 'horse_country', 'gear'):
            if name not in output or pd.isna(output.at[index, name]):
                output.at[index, name] = card.get(name)
                output.at[index, name + '_source'] = 'official:archived-racecard'
    return output


def enrich_past_country(frame, history):
    """Birth country is immutable, but only use a horse's earlier recorded evidence."""
    known = history[history.horse_country.notna() & history.horse_country.ne('UNKNOWN')].copy()
    if known.groupby('horse_id').horse_country.nunique().gt(1).any():
        raise ValueError('Conflicting immutable horse-country evidence')
    first = known.sort_values('date').drop_duplicates('horse_id').set_index('horse_id')
    output = frame.copy()
    if 'horse_country' not in output:
        output['horse_country'] = None
    for index, row in output.iterrows():
        if (pd.isna(row.horse_country) or row.horse_country == 'UNKNOWN') and row.horse_id in first.index:
            evidence = first.loc[row.horse_id]
            if evidence.date < row.date:
                output.at[index, 'horse_country'] = evidence.horse_country
                output.at[index, 'horse_country_source'] = 'official:strict-prior-immutable-country'
    return output


def prepare(args):
    official = json.loads((args.official_dir / 'coverage_manifest.json').read_text())
    if not official['results_complete'] or not official['racecards_complete']:
        raise ValueError('Official result/racecard census is incomplete; finish acquisition first')
    history = pd.read_parquet(args.history)
    history['date'] = pd.to_datetime(history.date)
    history = enrich_form_inputs(enrich_archived_cards(enrich_official_history(history, args.official_dir), args.official_dir), args.official_dir)
    season = runner_rows(json.loads((args.official_dir / 'races.json').read_text()))
    season = enrich_form_inputs(enrich_archived_cards(season, args.official_dir), args.official_dir)
    tomorrow = runner_rows(json.loads((args.official_dir / 'racecards.json').read_text()), card=True)
    # A published calendar/field denominator must not be replaced by a filtered model population.
    exclusions = []
    for race_id, race in season.groupby('race_id', sort=False):
        positions = pd.to_numeric(race.result, errors='coerce')
        finishers = positions.dropna()
        statuses = race.get('finishing_status', pd.Series('', index=race.index)).astype(str).str.upper()
        known_missing = statuses[positions.isna()].isin({'PU', 'UR', 'FE', 'DNF', 'DISQ', 'TNP'}).all()
        first_four = finishers[finishers.le(4)]
        if (not known_missing or sorted(first_four.tolist()) != [1, 2, 3, 4]):
            exclusions.append({'race_id': race_id, 'reason': 'nonfinisher_or_dead_heat_not_supported_in_fixed_ticket_audit'})
    excluded = {r['race_id'] for r in exclusions}
    evaluable = season[~season.race_id.isin(excluded)].copy()
    before = history[history.date.lt('2026-09-01') & history.date.gt('2024-11-16')].copy()
    eligible = []
    for race_id, race in before.groupby('race_id', sort=False):
        positions = pd.to_numeric(race.result, errors='coerce')
        if sorted(positions.tolist()) == list(range(1, len(race) + 1)) and pd.to_numeric(race.win_odds, errors='coerce').gt(0).all():
            eligible.append(race_id)
    dates = before[before.race_id.isin(eligible)][['race_id', 'date', 'race_no']].drop_duplicates().sort_values(['date', 'race_no', 'race_id'])
    selected_ids = set(dates.tail(args.calibration_races).race_id)
    calibration = before[before.race_id.isin(selected_ids)].copy()
    if len(selected_ids) < args.calibration_races:
        raise ValueError('Insufficient independent pre-season calibration races')
    calibration['cohort'] = 'calibration'
    evaluable['cohort'] = 'season'
    tomorrow['cohort'] = 'tomorrow'
    queries = pd.concat([calibration, evaluable, tomorrow], ignore_index=True)
    if 'horse_country' in history:
        queries = enrich_past_country(queries, history)
    if queries.duplicated(['race_id', 'horse_no']).any():
        raise ValueError('Duplicate query runner keys')
    args.output.mkdir(parents=True, exist_ok=True)
    queries.to_parquet(args.output / 'query-metadata.parquet', index=False)
    features = build_season_features(history, queries, season_results=season)
    features.to_parquet(args.output / 'query-features.parquet', index=False)
    if args.prepare_training:
        from ima.season_features import _normalize, _merge_exports
        prior = _merge_exports(_normalize(history[history.date.lt(calibration.date.min())], query=False))
        training = prepare_rich_runner_dataset(prior, strict_before_meeting=True)
        complete = training.groupby('race_id').result.apply(
            lambda positions: sorted(positions.tolist()) == list(range(1, len(positions) + 1)))
        training = training[training.race_id.isin(complete[complete].index)].reset_index(drop=True)
        speed = build_speed_features(training)
        training = pd.concat([training.drop(columns=list(set(speed) & set(training))), speed], axis=1)
        training.to_parquet(args.output / 'training-features.parquet', index=False)
        save_json(args.output / 'training-provenance.json', {
            'rows': len(training), 'races': training.race_id.nunique(),
            'actual_last_training_date': str(training.date.max()),
            'exclusive_calibration_boundary': str(calibration.date.min()),
            'input_sha256': hashlib.sha256((args.output / 'training-features.parquet').read_bytes()).hexdigest(),
            'policy': 'Fixed recipes, no model selection from season outcomes; original rich feature builder, strict prior-meeting histories',
            'numeric_missing_fraction': {name: float(training[name].isna().mean())
                                         for name in training.select_dtypes(include='number')}})
    save_json(args.output / 'preparation.json', {
        'history_path': str(args.history), 'history_sha256': hashlib.sha256(args.history.read_bytes()).hexdigest(),
        'history_rows': len(history), 'query_rows': len(queries), 'query_races': queries.race_id.nunique(),
        'source_hashes': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in args.official_dir.glob('*.json')},
        'feature_code_sha256': hashlib.sha256(Path('ima/season_features.py').read_bytes()).hexdigest(),
        'historical_form_note': 'Input ratings/gear reconstructed from official horse forms, not timestamped historical captures; outcomes are never used as current predictors.',
        'card_weight_note': 'Use explicitly derived declared net weight when available, preserving handicap weight and allowance provenance; subject to late declared changes.',
        'calibration_races': len(selected_ids), 'calibration_first_date': str(calibration.date.min()),
        'calibration_last_date': str(calibration.date.max()), 'season_races': evaluable.race_id.nunique(),
        'season_exclusions': exclusions, 'tomorrow_races': tomorrow.race_id.nunique(),
        'cohort_policy': 'Frozen base packages; blend/temperature/order exponents fitted before September only; strict prior-day query features',
        'feature_missing_fraction': {name: float(features[name].isna().mean()) for name in features.columns}})
    print(json.dumps({'query_rows': len(queries), 'query_races': queries.race_id.nunique(), 'season_exclusions': exclusions}))


def evaluate(args):
    metadata = pd.read_parquet(args.output / 'query-metadata.parquet')
    metadata['date'] = pd.to_datetime(metadata.date)
    calibration_dates = metadata.loc[metadata.cohort.eq('calibration'), 'date']
    if calibration_dates.empty or not calibration_dates.lt('2026-09-01').all():
        raise ValueError('Calibration must be strictly before the season holdout')
    keys = ['race_id', 'horse_no', 'horse_id']
    records = json.loads((args.official_dir / 'races.json').read_text())
    settlements = settlement_rows(records)
    result_dir = args.output / 'evaluation'
    result_dir.mkdir(parents=True, exist_ok=True)
    summaries, calibration_models, tomorrow_rows = {}, {}, []
    quotes, quote_units = quote_lookup(args.official_dir)
    paths = sorted(args.predictions_dir.glob('*.csv'))
    if not paths:
        raise ValueError('No frozen model predictions')
    receipt_path = args.predictions_dir / 'readback.json'
    if not receipt_path.exists():
        raise ValueError('Missing trusted inference readback')
    receipt = json.loads(receipt_path.read_text())
    feature_hash = hashlib.sha256((args.output / 'query-features.parquet').read_bytes()).hexdigest()
    if receipt.get('input_sha256') != feature_hash:
        raise ValueError('Predictions are stale for the prepared feature dataset')
    if {path.stem for path in paths} != set(receipt.get('models', {})):
        raise ValueError('Prediction candidate population differs from trusted inference receipt')
    candidate_path = args.output / 'models' / 'frozen-candidates.json'
    if candidate_path.exists() and set(receipt['models']) != set(json.loads(candidate_path.read_text())['candidates']):
        raise ValueError('Prediction candidates differ from the predeclared frozen manifest')
    for path in paths:
        if hashlib.sha256(path.read_bytes()).hexdigest() != receipt['models'][path.stem].get('output_sha256'):
            raise ValueError('Prediction file hash does not match trusted inference receipt')
    base_frames = {}
    for path in paths:
        predictions = pd.read_csv(path)
        if predictions.duplicated(keys).any():
            raise ValueError('Duplicate model prediction keys')
        frame = metadata.merge(predictions[[*keys, 'model_probability']], on=keys, validate='one_to_one', how='left')
        if frame.model_probability.isna().any() or len(frame) != len(predictions):
            raise ValueError('Model/official runner populations do not match')
        base_frames[path.stem] = frame
    benchmark = metadata.copy()
    benchmark['model_probability'] = np.nan
    nonlive = benchmark.cohort.ne('tomorrow')
    benchmark.loc[nonlive, 'model_probability'] = market_probability(benchmark[nonlive])
    base_frames['market_final_odds_hindsight'] = benchmark
    for name, source in base_frames.items():
        calibration = target_frame(source[source.cohort.eq('calibration')].copy())
        if 'horse_rating' in calibration:
            known_rating = pd.to_numeric(calibration.horse_rating, errors='coerce')
            rating_complete = np.isfinite(known_rating).groupby(calibration.race_id).transform('all')
            if not rating_complete.any():
                raise ValueError('No complete pre-season calibration fields have current ratings')
            calibration = calibration[rating_complete].copy()
        test = target_frame(source[source.cohort.eq('season')].copy())
        raw_market_cal, raw_market_test = market_probability(calibration), market_probability(test)
        temperature = TemperatureCalibrator.fit(raw_market_cal, calibration)
        fundamental = calibration.model_probability.to_numpy()
        blend = MarketBlend.fit(fundamental, raw_market_cal, calibration)
        calibration_models[name] = {'temperature': temperature.temperature,
            'blend': asdict(blend), 'calibration_races': calibration.race_id.nunique(),
            'calibration_quality_policy': 'Within the predeclared last500 races, use only whole fields with finite current ratings; no outcome or score based selection'}
        variants = {name: test.model_probability.to_numpy()}
        if not name.startswith('market_'):
            variants[name + '_market_blend_hindsight'] = blend.transform(test.model_probability.to_numpy(), raw_market_test, test.race_id)
        else:
            variants['market_calibrated_hindsight'] = temperature.transform(raw_market_test, test.race_id)
        for variant, probabilities in variants.items():
            work = test.assign(model_probability=probabilities)
            calibration_probabilities = blend.transform(fundamental, raw_market_cal, calibration.race_id) if '_market_blend_' in variant else (
                temperature.transform(raw_market_cal, calibration.race_id) if variant == 'market_calibrated_hindsight' else fundamental)
            order = fit_order_exponents(calibration.assign(_p=calibration_probabilities), '_p')
            evaluation = evaluate_season(work, settlements, exponents=order)
            evaluation.ledger.to_csv(result_dir / f'{variant}-tickets.csv', index=False)
            evaluation.bankroll_curve.to_csv(result_dir / f'{variant}-bankroll.csv', index=False)
            settled = evaluation.ledger[evaluation.ledger.status.eq('settled')]
            meetings = settled.groupby('date')[['stake_hkd', 'realized_gross_hkd', 'realized_net_hkd']].sum()
            meetings['roi'] = meetings.realized_net_hkd / meetings.stake_hkd
            meetings.to_csv(result_dir / f'{variant}-meetings.csv')
            roi_interval = None
            if len(meetings):
                sampled = np.random.default_rng(20261007).integers(0, len(meetings), size=(10000, len(meetings)))
                roi_samples = meetings.realized_net_hkd.to_numpy()[sampled].sum(axis=1) / meetings.stake_hkd.to_numpy()[sampled].sum(axis=1)
                roi_interval = np.quantile(roi_samples, [0.025, 0.975]).tolist()
            work.to_csv(result_dir / f'{variant}-runners.csv', index=False)
            top = work.loc[work.groupby('race_id').model_probability.idxmax()]
            summary = evaluation.summary | {'race_log_loss': race_log_loss(probabilities, work),
                'top_pick_win_hit_rate': float(top.result.eq(1).mean()),
                'top_pick_top3_hit_rate': float(top.result.le(3).mean()),
                'order_exponents': asdict(order), 'by_pool': evaluation.by_pool,
                'price_basis': 'Recorded final win odds hindsight benchmark' if 'hindsight' in variant else 'Fundamental pre-race predictions; official realized dividends',
                'ev_available_tickets': int(evaluation.ledger.estimated_ev_hkd.notna().sum())}
            summary['meeting_cluster_bootstrap_roi_interval_95'] = roi_interval
            summary['uncertainty_note'] = 'Exploratory percentile bootstrap over entire meetings, 10000 draws, seed20261007; only eight meetings, not evidence of a stable betting edge.'
            if 'hindsight' not in variant:
                selections = evaluation.ledger[evaluation.ledger.pool.eq('WIN')].copy()
                prices = work.assign(_horse=work.horse_no.astype(int).astype(str))
                selections = selections.merge(prices[['race_id', '_horse', 'win_odds']], left_on=['race_id', 'combination'], right_on=['race_id', '_horse'], validate='one_to_one')
                selections['hindsight_approximate_win_ev_hkd'] = selections.probability * selections.win_odds * 10 - 10
                selections.to_csv(result_dir / f'{variant}-hindsight-win-ev.csv', index=False)
                summary['hindsight_approximate_win_ev_hkd'] = float(selections.hindsight_approximate_win_ev_hkd.sum())
                summary['hindsight_win_price_note'] = 'Rounded final WIN odds times HKD10; settlement uses exact dividend. Not actionable pre-off EV; no odds-based ticket selection.'
            summaries[variant] = summary
        if not name.startswith('market_'):
            live = source[source.cohort.eq('tomorrow')].copy()
            order = fit_order_exponents(calibration.assign(_p=fundamental), '_p')
            for race_id, race in live.groupby('race_id', sort=False):
                for pool in SUPPORTED_POOLS:
                    combinations = rank_combinations(race.horse_no.astype(int).astype(str).tolist(), race.model_probability.to_numpy(), pool, exponents=order)
                    for rank, ticket in enumerate(combinations[:3], 1):
                        tomorrow_rows.append({'model': name, 'race_id': race_id, 'race_no': int(race.race_no.iloc[0]),
                            'pool': pool, 'rank': rank, 'combination': '/'.join(ticket.runners),
                            'probability': ticket.probability, 'stake_hkd_per_combination': 10,
                            'break_even_dividend_hkd_per_10': 10 / ticket.probability,
                            **ticket_quote(ticket, race, pool, quotes, quote_units)})
            live.to_csv(result_dir / f'{name}-tomorrow-runners.csv', index=False)
    pd.DataFrame(tomorrow_rows).to_csv(result_dir / 'tomorrow-combinations.csv', index=False)
    save_json(result_dir / 'calibration.json', calibration_models)
    save_json(result_dir / 'summary.json', summaries)
    save_json(result_dir / 'provenance.json', {
        'prediction_dir': str(args.predictions_dir), 'query_features_sha256': feature_hash,
        'inference_readback_sha256': hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        'evaluation_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'policy': 'Fixed probability-ranked HKD10 tickets; independent pre-September calibration; no paid order odds inferred from winning dividends'})
    print(json.dumps({name: {k: value[k] for k in ('race_log_loss', 'stake_hkd', 'gross_hkd', 'profit_hkd', 'roi', 'top_pick_win_hit_rate')} for name, value in summaries.items()}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'evaluate'), required=True)
    parser.add_argument('--official-dir', type=Path, default=ROOT / 'official')
    parser.add_argument('--output', type=Path, default=ROOT)
    parser.add_argument('--history', type=Path)
    parser.add_argument('--predictions-dir', type=Path)
    parser.add_argument('--calibration-races', type=int, default=500)
    parser.add_argument('--prepare-training', action='store_true', help='Also build strict pre-calibration inputs for fixed-recipe final fits')
    args = parser.parse_args()
    if args.calibration_races < 1:
        parser.error('calibration races must be positive')
    if args.stage == 'prepare' and args.history is None:
        parser.error('--history required for prepare')
    if args.stage == 'evaluate' and args.predictions_dir is None:
        parser.error('--predictions-dir required for evaluate')
    (prepare if args.stage == 'prepare' else evaluate)(args)


if __name__ == '__main__':
    main()
