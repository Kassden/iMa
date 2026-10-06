"""Render the verified paper evaluation and next-meeting probability tables."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shlex
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from ima.pools import OrderExponents, rank_pool_combinations
from scripts.evaluate_season_2026 import quote_lookup, ticket_quote


MODEL_LABELS = {
    'benter_conditional_logit': 'Benter conditional logit',
    'boosted': 'Boosted',
    'pool': '60% Benter + 40% boosted probability pool',
    'gaussian_probit': 'Gaussian probit (common variance)',
    'benter_conditional_logit_market_blend_hindsight': 'Benter + market (hindsight)',
    'boosted_market_blend_hindsight': 'Boosted + market (hindsight)',
    'pool_market_blend_hindsight': 'Benter/boosted pool + market (hindsight)',
    'gaussian_probit_market_blend_hindsight': 'Gaussian probit + market (hindsight)',
    'market_final_odds_hindsight': 'Market (final odds, hindsight)',
    'market_calibrated_hindsight': 'Market (temperature-calibrated, hindsight)',
}


def stable_runner_order(race, probability_column='model_probability'):
    return race.assign(_rank_probability=race[probability_column].round(12),
                       _rank_horse_no=pd.to_numeric(race.horse_no, errors='raise')).sort_values(
                           ['_rank_probability', '_rank_horse_no'], ascending=[False, True])


def ranked_predictions(root):
    """All fundamental runners, with paid-place probabilities and verified quotes."""
    evaluation = root / 'evaluation'
    candidates = json.loads((root / 'models/frozen-candidates.json').read_text())['candidates']
    scores = json.loads((evaluation / 'summary.json').read_text())
    quotes, units = quote_lookup(root / 'official')
    rows = []
    for name in candidates:
        frame = pd.read_csv(evaluation / f'{name}-tomorrow-runners.csv')
        order = OrderExponents(**scores[name]['order_exponents'])
        for race_id, race in frame.groupby('race_id', sort=True):
            ids = race.horse_no.astype(int).astype(str).tolist()
            placing = rank_pool_combinations(ids, race.model_probability.to_numpy(),
                                            pools=['PLACE'], exponents=order)['PLACE']
            place = {ticket.runners[0]: ticket.probability for ticket in placing}
            ranked = stable_runner_order(race)
            for rank, runner in enumerate(ranked.itertuples(), 1):
                horse_no = str(int(runner.horse_no))
                quote = ticket_quote(SimpleNamespace(runners=(horse_no,), probability=runner.model_probability),
                                     race, 'WIN', quotes, units)
                rows.append({'model': name, 'race_id': race_id, 'race_no': int(runner.race_no),
                             'rank': rank, 'horse_no': int(horse_no), 'horse_name': runner.horse_name,
                             'pwin': runner.model_probability, 'pplace': place[horse_no],
                             'win_raw_quote': quote.get('quoted_odds_raw'),
                             'win_gross_quote_hkd_per_10': quote['quoted_gross_hkd_per_10'],
                             'win_ev_hkd_per_10': quote['quoted_ev_hkd'],
                             'quote_status': quote['quote_status'],
                             'quote_retrieved_at_utc': quote.get('quote_retrieved_at_utc'),
                             'quote_source_url': quote.get('quote_source_url')})
    return pd.DataFrame(rows).sort_values(['model', 'race_no', 'rank']).reset_index(drop=True)


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |',
                      *['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]])


def money(value):
    return f'{value:,.2f}'


def percent(value):
    return f'{100 * value:.2f}%'


def hindsight_win_ev_section(scores, names):
    rows = []
    for name, value in scores.items():
        ev = value.get('hindsight_approximate_win_ev_hkd')
        if ev is None:
            continue
        win = value['by_pool']['WIN']
        rows.append([names.get(name, name), win['nsettled'], money(win['stake_hkd']),
                     money(ev), money(win['profit_hkd'])])
    if not rows:
        return []
    return ['## Season Hindsight WIN EV', '',
            'Total approximate expected net profit over each model\'s selected HKD10 WIN tickets: '
            'sum(pwin x rounded final WIN odds x 10 - 10). These are **hindsight prices, not actionable pre-off EV** '
            'or realized returns. Realized net profit uses exact official dividends; no second takeout deduction is applied.', '',
            table(['Model', 'Settled WIN tickets', 'WIN stake HKD', 'Hindsight total WIN EV HKD',
                   'Realized WIN net HKD'], rows), '']


def combined_market_section(evaluation, primary='pool'):
    path = evaluation / 'tomorrow-market-blends.csv'
    if not path.exists():
        return []
    proof_path = evaluation / 'tomorrow-market-blend-provenance.json'
    if not proof_path.is_file():
        raise ValueError('Combined-market forecast requires report-ready provenance')
    proof_bytes = proof_path.read_bytes()
    proof = json.loads(proof_bytes)
    if proof.get('ready_for_report') is not True:
        raise ValueError('Combined-market forecast is not ready for report')

    def expected_hash(mapping_name, target):
        mapping = proof.get(mapping_name)
        if not isinstance(mapping, dict):
            raise ValueError(f'Combined-market provenance missing {mapping_name}')
        matches = [digest for name, digest in mapping.items() if Path(name).resolve() == target.resolve()]
        if len(matches) != 1 or not isinstance(matches[0], str):
            raise ValueError(f'Combined-market provenance requires one hash for {target}')
        return matches[0]

    sources = [evaluation / name for name in ('provenance.json', 'calibration.json', 'summary.json')]

    def verify_sources():
        for source in sources:
            digest = expected_hash('source_sha256', source)
            if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != digest:
                raise ValueError(f'Combined-market forecast has stale source: {source}')

    verify_sources()
    csv_bytes = path.read_bytes()
    if hashlib.sha256(csv_bytes).hexdigest() != expected_hash('output_sha256', path):
        raise ValueError('Combined-market forecast CSV hash mismatch')
    frame = pd.read_csv(io.BytesIO(csv_bytes))
    verify_sources()
    if proof_path.read_bytes() != proof_bytes:
        raise ValueError('Combined-market provenance changed during read')
    aliases = {'pcombined': 'combinedp', 'pfund': 'fundp', 'pmarket': 'marketp',
               'combined_probability': 'combinedp', 'combined_probability_win': 'combinedp',
               'combined_pwin': 'combinedp', 'combined_pplace': 'pplace',
               'fundamental_probability': 'fundp', 'market_probability': 'marketp'}
    frame = frame.rename(columns={k: v for k, v in aliases.items() if v not in frame.columns})
    required = {'model', 'race_no', 'horse_no', 'combinedp', 'pplace'}
    if not required.issubset(frame.columns):
        raise ValueError(f'Combined-market forecast missing columns: {sorted(required - set(frame.columns))}')
    ev_note = ''
    if {'quoted_ev_hkd', 'quote_status'}.issubset(frame.columns):
        available_ev = pd.to_numeric(frame.loc[
            frame.quote_status.eq('indicative_snapshot_not_final'), 'quoted_ev_hkd'], errors='coerce').dropna()
        if len(available_ev):
            ev_note = (f'Across all models, {len(available_ev)} unit-verified combined WIN snapshot EVs are available; '
                       f'{int(available_ev.lt(0).sum())} are negative, with maximum HKD{available_ev.max():.4f} per HKD10. ')
    frame = frame[frame.model.eq(primary)]
    if frame.empty or frame.duplicated(['race_no', 'horse_no']).any():
        raise ValueError('Combined-market forecast has no default pool rows or duplicate identities')
    rows = []
    for race_no, race in frame.groupby('race_no', sort=True):
        if race[['combinedp', 'pplace']].isna().any().any() or abs(race.combinedp.sum() - 1) > 1e-6:
            raise ValueError('Combined-market forecast has incomplete probabilities')
        top = stable_runner_order(race, 'combinedp').head(3)
        pick = top.iloc[0]
        verified = pick.get('quote_status') == 'indicative_snapshot_not_final'
        ev = pick.get('quoted_ev_hkd') if verified else None
        raw = pick.get('quoted_odds_raw')
        rows.append([int(race_no), '; '.join(
            f"#{int(r.horse_no)} {getattr(r, 'horse_name', '')} (WIN {percent(r.combinedp)}, PLACE {percent(r.pplace)})"
            for r in top.itertuples()), str(raw) if pd.notna(raw) else 'unavailable',
            money(ev) if ev is not None and pd.notna(ev) else 'unavailable'])
    return ['### Indicative Combined-Market Snapshot', '',
            'Default pool fundamentals combined forward-only with captured public WIN quotes using the pre-fitted market blend. '
            'This snapshot forecast is **not a validated pre-off betting strategy**; prices and fields can change. '
            'No October 7 outcomes refit the blend. Full probabilities and unit-verified WIN EV are in '
            '`evaluation/tomorrow-market-blends.csv`; tickets are in `evaluation/tomorrow-market-blend-combinations.csv`.', '',
            ev_note + 'Positive fundamental-only snapshot EV can be optimistic and is not treated as a confidence-qualified betting edge. '
            'The combined forecast and its EV are distinct from those fundamental-only estimates.', '',
            table(['Race', 'Combined-market top3', 'Top pick raw WIN quote', 'Top pick snapshot WIN EV HKD/10'], rows), '']


def render(root):
    evaluation = root / 'evaluation'
    scores = json.loads((evaluation / 'summary.json').read_text())
    preparation = json.loads((root / 'preparation.json').read_text())
    candidates = json.loads((root / 'models/frozen-candidates.json').read_text())['candidates']
    calibration = json.loads((evaluation / 'calibration.json').read_text())
    provenance = json.loads((evaluation / 'provenance.json').read_text())
    receipt = json.loads((Path(provenance['prediction_dir']) / 'readback.json').read_text())
    coverage = json.loads((root / 'official/coverage_manifest.json').read_text())
    history = json.loads((root / 'official/history-race-coverage.json').read_text())
    primary = 'pool'
    names = MODEL_LABELS
    lines = ['# 2026/27 Season Evaluation And October 7 Predictions', '',
        '## Interpretation First', '',
        'This is a **paper backtest and probability forecast, not a demonstrated betting edge**. '
        'Models were selected using prior development results before observing this season. '
        'Fresh final fits use only outcomes before the independent January-July calibration cohort. '
        'Older saved research-fold estimators are preserved as a separate benchmark: three actually fit through March 16, 2024, and the pool through November 13, 2024. '
        'Their feature-context cutoff was not their actual final fitting date. '
        'Historical features are reconstructed from official records; we do not have timestamped historical racecard or odds snapshots. '
        'No bets were placed.', '',
        'The default forecast is the freshly fitted Benter/boosted probability pool configuration, chosen by its lowest prior development loss, '
        'not by whichever model happened to profit most this September. '
        'The Gaussian package tested here is **homoscedastic**, not the newer horse-specific-variance configuration.', '',
        '## Coverage And Protocol', '',
        f"- Completed season: **{preparation['season_races']} races across eight meetings**, September 6 through October 4, 2026. September 20 was cancelled.",
        f"- Tomorrow: **{preparation['tomorrow_races']} Happy Valley races, October 7**; 108 captured starters, subject to subsequent scratches and updates.",
        f"- History: {preparation['history_rows']:,} runner records. January-July official census: {history['covered_races']} races; {history['new_races']} newly recovered full race fields.",
        f"- Reserved calibration: {preparation['calibration_races']} races ({preparation['calibration_first_date']} to {preparation['calibration_last_date']}). "
        f"Actual fitting uses {', '.join(str(n) for n in sorted({v['calibration_races'] for v in calibration.values()}))} whole fields with finite ratings "
        f"of the {preparation['calibration_races']} reserved races, selected by input completeness, not outcomes. Only these labels fit market blending, temperature and finish-order exponents.",
        '- Each query uses only strictly earlier calendar-day outcomes. Current and later outcomes are blanked; entire fields and exact horse identities are retained.',
        f"- Season exclusions: {json.dumps(preparation['season_exclusions'])}.",
        '- Pool settlement uses exact published dividends, explicit HKD unit conversion and pool-specific dead-heat/nonfinisher validation. One absent QPL pool is not counted as a losing ticket.', '',
        'Sources: [season opening](https://racingnews.hkjc.com/english/2026/08/31/hong-kong-saddles-up-for-2026-27-season-opening-at-sha-tin-on-sunday/), '
        '[October 7 official racecard](https://racing.hkjc.com/en-us/local/information/racecard?RaceNo=1&Racecourse=HV&racedate=2026%2F10%2F07). '
        'Every captured page has its official URL, retrieval timestamp and SHA256 in the coverage manifests.', '',
        '## HKD10 Costs And EV', '',
        'One straight WIN, PLACE, QIN, QPL, TRIO, TIERCE, FIRST4 or QUARTET combination costs HKD10 in this test. '
        'Exactly one highest-probability ticket per pool per race is selected **without consulting the result or payout**. '
        'A four-horse Trio box has four combinations and costs HKD40; a four-horse Tierce box has 24 ordered combinations and costs HKD240. '
        'Flexi minimums are conditional, so they are not assumed here. Cross-race pools and Forecast are outside this evaluation. '
        '[HKJC Flexi rules](https://special.hkjc.com/e-win/en-US/betting-info/racing/flexi-bet/info/).', '',
        'Official payout shares are 82.5% for win/place/quinella/QPL, 77% for trio, and 75% for tierce/first-four/quartet. '
        '**Published dividends already reflect pool deductions: do not deduct takeout again.** '
        '[HKJC local pools](https://special.hkjc.com/e-win/en-US/betting-info/racing/beginners-guide/local-pools/).', '',
        'For a quoted gross payout D per HKD10, expected net profit is **pD - 10**; break-even D is **10/p**. '
        'Realized profit is actual gross return minus settled stake; ROI is realized profit / settled stake. '
        'Recorded final win odds allow only an approximate hindsight-priced WIN EV (rounded odds, not exact dividends). '
        'Winning-only historical PLACE/exotic dividends do not give quotes for losing selections, so their actionable EV is **unknown**. '
        'Captured official October 7 WIN, FIRST4 and QUARTET quotes support indicative snapshot EV only where their units are verified. '
        'These are not final or guaranteed prices. Unverified units, missing/non-numeric quotes and possible display ceilings have no point EV; no fabricated prices are used.', '',
        '## Model Comparison', '',
        'Portfolio below means all eight tested pools, HKD10 each. Lower log loss is better. '
        'The final-odds market and market-blend rows are explicitly hindsight benchmarks, not deployable pre-off results.', '']
    rows = []
    for name, value in scores.items():
        label = names.get(name, name)
        interval = value['meeting_cluster_bootstrap_roi_interval_95']
        rows.append([label, f"{value['race_log_loss']:.6f}", percent(value['top_pick_win_hit_rate']),
                     percent(value['top_pick_top3_hit_rate']), money(value['stake_hkd']), money(value['gross_hkd']),
                     money(value['profit_hkd']), percent(value['roi']),
                     f'{percent(interval[0])} to {percent(interval[1])}' if interval else 'unavailable'])
    lines += [table(['Model', 'Win log loss', 'Top-pick win', 'Top-pick top3', 'Stake HKD', 'Gross HKD', 'Net HKD', 'ROI', 'Meeting-bootstrap 95% ROI interval'], rows), '',
        '**Tie audit:** an earlier summary used raw floating-point `idxmax`, allowing tiny probability differences and input row order '
        'to disagree with the ticket ledger. Top-pick WIN and top3 metrics now use probabilities rounded to 12 decimals, '
        'then numeric horse number, matching the fixed WIN ticket policy. The correction changes top-pick metrics, not log loss or portfolio ROI. '
        'Final reported metrics must come from the corrected fresh and archived-benchmark reruns, not mixed interim artifacts.', '',
        'Top-pick top3 is not automatically PLACE hit rate in small fields. Paid PLACE and QPL counts are cross-checked against official winning combinations. '
        'The intervals resample whole meetings (10,000 draws, fixed seed), not independent tickets. Eight meetings are too few to establish a reliable edge; '
        'one exotic payout can dominate returns. Comparing these models on this season is exploratory, not fresh validation of a selected winner.', '',
        '## Returns By Pool', '']
    old_path = root / 'frozen-benchmark/evaluation/summary.json'
    if old_path.exists():
        old = json.loads(old_path.read_text())
        lines[-2:] = ['## Older Packages Versus Fresh Fixed-Recipe Fits', '',
            table(['Family', 'Older log loss', 'Fresh log loss', 'Older portfolio ROI', 'Fresh portfolio ROI'],
                  [[names[name], f"{old[name]['race_log_loss']:.6f}", f"{scores[name]['race_log_loss']:.6f}",
                    percent(old[name]['roi']), percent(scores[name]['roi'])] for name in candidates]), '',
            'This compares fixed configurations on identical season fields after updating their estimator weights with strictly pre-calibration outcomes. '
            'It does not choose a winner by September ROI.', '', '## Returns By Pool', '']
    for name in scores:
        rows = []
        for pool, value in scores[name]['by_pool'].items():
            rows.append([pool, value['nsettled'], money(value['stake_hkd']), money(value['gross_hkd']),
                         money(value['profit_hkd']), percent(value['roi']), percent(value['hit_rate'])])
        lines += [f'### {names.get(name, name)}', '', table(['Pool', 'Settled tickets', 'Stake HKD', 'Gross HKD', 'Net HKD', 'ROI', 'Hit rate'], rows), '']
    lines += hindsight_win_ev_section(scores, names)
    value = scores[primary]
    train_path = root / 'training-provenance.json'
    if train_path.exists():
        trained = json.loads(train_path.read_text())
        lines += ['## Final Fit Dates', '',
                  f"Fresh fits use **{trained['races']:,} races / {trained['rows']:,} runners**, ending **{trained['actual_last_training_date']}**. "
                  f"The independent calibration period starts {trained['exclusive_calibration_boundary']}. "
                  'No September or October outcome updates estimator weights. All fixed-recipe fitted packages are saved separately under `final-fits/`.', '']
        lines += ['Historical horse age remains **NaN in the training dataset, not zero-filled**. '
                  'The training imputer found no finite values and dropped the four age columns '
                  '(`horse_age`, `age_rank`, `poly_age_sq`, `poly_age_distance`); no historical age effect was learned.', '']
    lines += ['## Progression And Capital', '',
        f"Default pool-model all-pool paper portfolio: minimum capital needed for this exact fixed-stake schedule was **HKD{money(value['min_bankroll_capital_required_hkd'])}**. "
        f"HKD1,000 could finance it: **{value['can_finance_with_1000']}**. Maximum paper drawdown: HKD{money(value['maximum_drawdown_hkd'])}. "
        'If a schedule exhausts its capital, an algebraic ending balance is not achievable bankroll growth without extra funding. '
        'There is no compounding, Kelly staking or automatic reinvestment in this fixed-HKD10 test.', '']
    meetings = pd.read_csv(evaluation / f'{primary}-meetings.csv')
    meetings['cum_net'] = meetings.realized_net_hkd.cumsum()
    meetings['cum_roi'] = meetings.cum_net / meetings.stake_hkd.cumsum()
    lines += [table(['Meeting', 'Stake HKD', 'Gross HKD', 'Net HKD', 'Cumulative net HKD', 'Cumulative ROI'],
                    [[row.date, money(row.stake_hkd), money(row.realized_gross_hkd), money(row.realized_net_hkd), money(row.cum_net), percent(row.cum_roi)] for row in meetings.itertuples()]), '',
        'Full race-by-race progression is in the `*-bankroll.csv` and `*-tickets.csv` files, including every losing ticket.', '',
        '## October 7 Forecasts', '',
        'Default fundamental probability pool; these are model estimates, not guarantees. Refresh scratches, jockey changes, going and prices before considering a ticket. '
        'A higher win probability does not imply positive EV. Break-even prices below contain no uncertainty cushion.', '']
    live = pd.read_csv(evaluation / f'{primary}-tomorrow-runners.csv')
    ranked = ranked_predictions(root)
    combos = pd.read_csv(evaluation / 'tomorrow-combinations.csv', dtype={'combination': str})
    combos = combos[combos.model.eq(primary)]
    rows = []
    for race_no, race in live.groupby('race_no', sort=True):
        top = stable_runner_order(race).head(3)
        pick = top.iloc[0]
        place = ranked[ranked.model.eq(primary) & ranked.race_no.eq(race_no) & ranked.horse_no.eq(int(pick.horse_no))]
        rows.append([int(race_no), int(race.distance.iloc[0]),
                     '; '.join(f"#{int(r.horse_no)} {r.horse_name} ({percent(r.model_probability)})" for r in top.itertuples()),
                     percent(place.pplace.iloc[0]),
                     money(10 / pick.model_probability)])
    lines += [table(['Race', 'Distance m', 'Ranked top3: win probability', 'Top WIN pick placing probability', 'Top WIN pick break-even gross HKD/10'], rows), '']
    lines += combined_market_section(evaluation, primary)
    lines += [
        '### Top Exotic Combinations', '',
        'One straight ticket in each cell costs HKD10. QIN/QPL/TRIO/FIRST4 are unordered; TIERCE/QUARTET order is exactly as shown. '
        'These are maximum-probability combinations, **not price-qualified value bets**. Probabilities are derived using Benter-corrected finish-order exponents fitted only on the rated calibration fields. '
        '**FIRST4 and QUARTET extrapolate the third-position exponent to the fourth stage; no separate fourth-position exponent is learned.**', '']
    pools = ['QIN', 'QPL', 'TRI', 'TIERCE', 'FIRST4', 'QUARTET']
    rows = []
    for race_no, race in combos[combos['rank'].eq(1)].groupby('race_no', sort=True):
        cells = []
        for pool in pools:
            ticket = race[race.pool.eq(pool)].iloc[0]
            ev = getattr(ticket, 'quoted_ev_hkd', None)
            quote_text = f'; snapshot EV HKD{money(ev)}' if ev is not None and pd.notna(ev) else '; snapshot EV unavailable'
            cells.append(f"{ticket.combination}: {percent(ticket.probability)}; break-even {money(ticket.break_even_dividend_hkd_per_10)}{quote_text}")
        rows.append([int(race_no), *cells])
    lines += [table(['Race', *pools], rows), '', '### Other Models: Top WIN Picks', '']
    rows = []
    for name in candidates:
        frame = pd.read_csv(evaluation / f'{name}-tomorrow-runners.csv')
        picks = []
        for _, race in frame.groupby('race_no', sort=True):
            pick = stable_runner_order(race).iloc[0]
            picks.append(f'#{int(pick.horse_no)} {pick.horse_name} ({percent(pick.model_probability)})')
        rows.append([names[name], *picks])
    lines += [table(['Model', *[f'R{n}' for n in range(1, 10)]], rows), '', '## Calibration And Provenance', '']
    lines += [table(['Model', 'Fundamental exponent', 'Market exponent', 'Prior development log loss', 'Package ID'],
                    [[names[name], f"{calibration[name]['blend']['fundamental_weight']:.8f}", f"{calibration[name]['blend']['market_weight']:.8f}",
                      f"{candidate['prior_loss']:.8f}", receipt['models'][name]['package_id']] for name, candidate in candidates.items()]), '',
        'Zero fundamental weight is allowed: it means that calibration found no additional conditional signal given the market in that calibration cohort. '
        'It is not proof the optimizer is broken or that probabilities should be forced into the blend. '
        'The audit did find a numerical floor/finite-difference flaw in the previous implementation; the repaired path uses log-softmax, analytic gradients and convergence checks. '
        'Original calibration predictions were not preserved for every older run, so the retrospective gradient audit is descriptive, not proof of a historical fitting error.', '',
        f"Input feature SHA256: `{receipt['input_sha256']}`. Exact inference environment: `{json.dumps(receipt['environment'], sort_keys=True)}`. "
        'No dependency guards were bypassed. Trusted packages were executed in their original remote environment without modifying running campaigns.', '',
        'All four models and all ten evaluated variants are shown above, including each variant\'s pool ROI. '
        'The compact runner ranking is `docs/OCT07_2026_RANKED_PREDICTIONS.csv` (four models x 108 starters = 432 rows for the captured fields); '
        'it includes WIN and paid-PLACE probabilities, raw WIN quotes, unit-verified snapshot EV and quote timestamps.', '',
        f'Artifacts: `{root}`. Older estimators and evaluations remain archived under `frozen-benchmark/`. '
        '`official/` contains raw source captures and coverage; `models/frozen-candidates.json` contains selection and package hashes; '
        f"`preparation.json` documents data/input coverage; `{Path(provenance['prediction_dir']) / 'readback.json'}` binds model outputs to the prepared feature hash; "
        '`evaluation/` contains all runners, tickets, pool combinations, daily totals, capital curves and calibration parameters.', '',
        '## Reproduction', '', '```sh',
        '.venv/bin/python -m scripts.collect_season_2026 --help',
        '.venv/bin/python -m scripts.evaluate_season_2026 --stage prepare --history .tmp/season-official-history.parquet',
        '# Fresh fixed-recipe fitting and inference: scripts/refit_season_2026.py; see --help.',
        f".venv/bin/python -m scripts.evaluate_season_2026 --stage evaluate --predictions-dir {shlex.quote(provenance['prediction_dir'])}",
        '.venv/bin/python -m scripts.forecast_season_market',
        '.venv/bin/python -m scripts.report_season_2026', '```', '',
        'Re-evaluation rejects stale prediction-input hashes, changed CSVs, identity mismatches and incomplete acquisition. '
        'After any racecard refresh, rebuild features and regenerate all predictions before evaluating.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('artifacts/race-readiness-20261007'))
    parser.add_argument('--output', type=Path, default=Path('docs/SEASON_2026_EVALUATION_AND_OCT07_PREDICTIONS.md'))
    parser.add_argument('--ranked-output', type=Path, default=Path('docs/OCT07_2026_RANKED_PREDICTIONS.csv'))
    args = parser.parse_args()
    report = render(args.root)
    ranked = ranked_predictions(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.ranked_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    ranked.to_csv(args.ranked_output, index=False)
    print(args.output)
    print(args.ranked_output)


if __name__ == '__main__':
    main()
