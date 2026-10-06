# Preseason Pool Search Results

## Scope And Decision

**588 pool combinations, not 588 model fits.** This report performs no retraining, no LLM calls and no wagering. The frozen component predictions are reused; only pool rules were searched.

Selected recipe: standalone_pair; geometric; Benter weight 0.15; temperature 0.85. Baseline: original_pair; arithmetic; Benter weight 0.6; temperature 1.

Best search loss: 2.070706386. Confirmation candidate loss: 2.037677197; baseline: 2.058602248; candidate minus baseline: -0.020925052. Protocol adoption decision: candidate selected. Forecast family: **Search-selected candidate pool**. This is a **provisional experiment only**; the original deployment was not changed.

## Three Disjoint Preseason Windows

| Window | Races | Meetings | First date | Last date |
| --- | --- | --- | --- | --- |
| Search | 89 | 30 | 2026-01-14 | 2026-04-29 |
| Confirmation | 34 | 10 | 2026-05-03 | 2026-06-07 |
| Calibration | 97 | 10 | 2026-06-10 | 2026-07-15 |

Search chooses the pool recipe; confirmation compares it with the fixed baseline; the final calibration window fits the downstream market blend and finish-order exponents. These date windows do not overlap. The September/October season was already viewed in the earlier report, so this is **exploratory, not a new untouched holdout or independent proof of improvement**.

The confirmation cohort has only **34 races**; this small comparison is not strong evidence of generalization or a betting edge.

Search policy: already-inspected exploratory evaluation, not untouched confirmation. Calibration policy: market/order fitting exclusively in final disjoint window.

## Season Comparison

| Variant | Win log loss | Stake HKD | Gross HKD | Net HKD | ROI |
| --- | --- | --- | --- | --- | --- |
| Original baseline pool | 2.194847973 | 6,230.00 | 3,853.70 | -2,376.30 | -38.14% |
| Search-selected candidate pool | 2.202447937 | 6,230.00 | 3,778.70 | -2,451.30 | -39.35% |
| Baseline + market (hindsight) | 2.066618742 | 6,230.00 | 6,768.20 | 538.20 | 8.64% |
| Candidate + market (hindsight) | 2.066618742 | 6,230.00 | 6,768.20 | 538.20 | 8.64% |
| Market (final odds, hindsight) | 2.067326991 | 6,230.00 | 6,768.20 | 538.20 | 8.64% |
| Market (temperature-calibrated, hindsight) | 2.066618759 | 6,230.00 | 6,768.20 | 538.20 | 8.64% |

Candidate fundamental season loss **2.202447937** is worse than baseline **2.194847973**. Confirmation-based adoption is not a recommendation that the candidate is a stronger fundamental model. Lower log loss is better; final-odds market and blend rows are hindsight benchmarks, not validated pre-off strategies.

## Candidate Returns By Pool

### Search-selected candidate pool

| Pool | Settled | Stake HKD | Gross HKD | Net HKD | ROI |
| --- | --- | --- | --- | --- | --- |
| WIN | 78 | 780.00 | 467.00 | -313.00 | -40.13% |
| PLACE | 78 | 780.00 | 628.70 | -151.30 | -19.40% |
| QIN | 78 | 780.00 | 673.00 | -107.00 | -13.72% |
| QPL | 77 | 770.00 | 619.00 | -151.00 | -19.61% |
| TRI | 78 | 780.00 | 697.00 | -83.00 | -10.64% |
| TIERCE | 78 | 780.00 | 619.00 | -161.00 | -20.64% |
| FIRST4 | 78 | 780.00 | 75.00 | -705.00 | -90.38% |
| QUARTET | 78 | 780.00 | 0.00 | -780.00 | -100.00% |

### Candidate + market (hindsight)

| Pool | Settled | Stake HKD | Gross HKD | Net HKD | ROI |
| --- | --- | --- | --- | --- | --- |
| WIN | 78 | 780.00 | 599.00 | -181.00 | -23.21% |
| PLACE | 78 | 780.00 | 677.70 | -102.30 | -13.12% |
| QIN | 78 | 780.00 | 1,286.00 | 506.00 | 64.87% |
| QPL | 77 | 770.00 | 987.50 | 217.50 | 28.25% |
| TRI | 78 | 780.00 | 529.00 | -251.00 | -32.18% |
| TIERCE | 78 | 780.00 | 2,111.00 | 1,331.00 | 170.64% |
| FIRST4 | 78 | 780.00 | 578.00 | -202.00 | -25.90% |
| QUARTET | 78 | 780.00 | 0.00 | -780.00 | -100.00% |

## Market Blend Calibration

Combined WIN probability is proportional to fundamental probability raised to a, times market probability raised to b, normalized within each race.

| Pool | Fundamental exponent a | Market exponent b | Calibration races |
| --- | --- | --- | --- |
| Original baseline pool | 0.000000000 | 1.031768923 | 97 |
| Search-selected candidate pool | 0.000000000 | 1.031768923 | 97 |

Zero fundamental weight for Original baseline pool, Search-selected candidate pool means a **pure calibrated-market forecast, not a model edge**: those blends discard the fundamental prediction. A zero exponent is a calibration result, not evidence that the selected pool adds signal beyond the market.

**Not a like-for-like coefficient comparison:** this study fits the market blends on 97 races in the final disjoint calibration window. The earlier season report fitted the original pool blend on 220 races and reported fundamental weight approximately 0.0721. The fitting populations differ; comparing those weights does not isolate a recipe improvement or prove that a previously established model edge disappeared.

## Selected October 7 Snapshot Forecast

WIN quote snapshot captured **2026-10-06 21:23:25 to 2026-10-06 21:23:34 HKT**, from the selected family's quote retrieval rows. These are capture times, not a claim that prices remained unchanged until race time.

| Race | Combined WIN pick | Combined WIN p | Paid PLACE p | WIN EV HKD/10 | Fundamental WIN pick | Fundamental WIN p |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | #1 CAN'T GO WONG | 20.63% | 49.44% | -1.75 | #1 CAN'T GO WONG | 31.89% |
| 2 | #10 LOVING VIBES | 32.51% | 65.83% | -1.55 | #10 LOVING VIBES | 35.59% |
| 3 | #10 FORTUNE STAR | 33.65% | 68.73% | -1.59 | #6 DECISION LINK | 24.38% |
| 4 | #2 SKY DEEP | 19.22% | 47.46% | -1.74 | #8 ABSOLUTE HONOUR | 20.15% |
| 5 | #4 SUPERB KING | 27.05% | 60.47% | -1.61 | #4 SUPERB KING | 26.10% |
| 6 | #9 NO OTHER CHOICE | 26.06% | 58.07% | -1.66 | #9 NO OTHER CHOICE | 23.84% |
| 7 | #5 TARGET AUDIENCE | 20.78% | 50.03% | -1.69 | #5 TARGET AUDIENCE | 22.59% |
| 8 | #1 DAZZLING FIT | 17.96% | 45.07% | -1.74 | #1 DAZZLING FIT | 23.51% |
| 9 | #7 THE HEIR | 20.78% | 49.41% | -1.69 | #9 LIVE WIRE | 24.11% |

108 known snapshot WIN EVs for the selected family: 108 negative, 0 positive, 0 zero; 0 unknown. EV is expected net profit at the captured quote, p x gross payout per HKD10 minus HKD10. **Positive snapshot EV does not guarantee an edge; negative EV is not a value recommendation.** Missing quotes remain unknown, not zero. Prices and fields can change. Fundamental-only optimism is not confidence-qualified betting evidence.

Settlement uses exact official dividends and fixed HKD10 straight tickets; published dividends already reflect takeout. ROI is realized net divided by settled stake. Missing settlements are not losing tickets. There is no compounding or guaranteed bankroll growth. Fourth-stage FIRST4/QUARTET probabilities reuse the third-position exponent.

## Reproduction

```sh
.venv/bin/python -m scripts.report_pool_search --root artifacts/pool-search-20261007-final
```

All source and output SHA256 entries in `result-provenance.json` were verified, including every report input. Any stale receipt or changed input prevents writing.
