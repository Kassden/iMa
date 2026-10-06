# 2026/27 Season Evaluation And October 7 Predictions

## Interpretation First

This is a **paper backtest and probability forecast, not a demonstrated betting edge**. Models were selected using prior development results before observing this season. Fresh final fits use only outcomes before the independent January-July calibration cohort. Older saved research-fold estimators are preserved as a separate benchmark: three actually fit through March 16, 2024, and the pool through November 13, 2024. Their feature-context cutoff was not their actual final fitting date. Historical features are reconstructed from official records; we do not have timestamped historical racecard or odds snapshots. No bets were placed.

The default forecast is the freshly fitted Benter/boosted probability pool configuration, chosen by its lowest prior development loss, not by whichever model happened to profit most this September. The Gaussian package tested here is **homoscedastic**, not the newer horse-specific-variance configuration.

## Coverage And Protocol

- Completed season: **78 races across eight meetings**, September 6 through October 4, 2026. September 20 was cancelled.
- Tomorrow: **9 Happy Valley races, October 7**; 108 captured starters, subject to subsequent scratches and updates.
- History: 181,479 runner records. January-July official census: 564 races; 217 newly recovered full race fields.
- Reserved calibration: 500 races (2026-01-14 00:00:00 to 2026-07-15 00:00:00). Actual fitting uses 220 whole fields with finite ratings of the 500 reserved races, selected by input completeness, not outcomes. Only these labels fit market blending, temperature and finish-order exponents.
- Each query uses only strictly earlier calendar-day outcomes. Current and later outcomes are blanked; entire fields and exact horse identities are retained.
- Season exclusions: [].
- Pool settlement uses exact published dividends, explicit HKD unit conversion and pool-specific dead-heat/nonfinisher validation. One absent QPL pool is not counted as a losing ticket.

Sources: [season opening](https://racingnews.hkjc.com/english/2026/08/31/hong-kong-saddles-up-for-2026-27-season-opening-at-sha-tin-on-sunday/), [October 7 official racecard](https://racing.hkjc.com/en-us/local/information/racecard?RaceNo=1&Racecourse=HV&racedate=2026%2F10%2F07). Every captured page has its official URL, retrieval timestamp and SHA256 in the coverage manifests.

## HKD10 Costs And EV

One straight WIN, PLACE, QIN, QPL, TRIO, TIERCE, FIRST4 or QUARTET combination costs HKD10 in this test. Exactly one highest-probability ticket per pool per race is selected **without consulting the result or payout**. A four-horse Trio box has four combinations and costs HKD40; a four-horse Tierce box has 24 ordered combinations and costs HKD240. Flexi minimums are conditional, so they are not assumed here. Cross-race pools and Forecast are outside this evaluation. [HKJC Flexi rules](https://special.hkjc.com/e-win/en-US/betting-info/racing/flexi-bet/info/).

Official payout shares are 82.5% for win/place/quinella/QPL, 77% for trio, and 75% for tierce/first-four/quartet. **Published dividends already reflect pool deductions: do not deduct takeout again.** [HKJC local pools](https://special.hkjc.com/e-win/en-US/betting-info/racing/beginners-guide/local-pools/).

For a quoted gross payout D per HKD10, expected net profit is **pD - 10**; break-even D is **10/p**. Realized profit is actual gross return minus settled stake; ROI is realized profit / settled stake. Recorded final win odds allow only an approximate hindsight-priced WIN EV (rounded odds, not exact dividends). Winning-only historical PLACE/exotic dividends do not give quotes for losing selections, so their actionable EV is **unknown**. Captured official October 7 WIN, FIRST4 and QUARTET quotes support indicative snapshot EV only where their units are verified. These are not final or guaranteed prices. Unverified units, missing/non-numeric quotes and possible display ceilings have no point EV; no fabricated prices are used.

## Model Comparison

Portfolio below means all eight tested pools, HKD10 each. Lower log loss is better. The final-odds market and market-blend rows are explicitly hindsight benchmarks, not deployable pre-off results.

| Model | Win log loss | Top-pick win | Top-pick top3 | Stake HKD | Gross HKD | Net HKD | ROI | Meeting-bootstrap 95% ROI interval |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Benter conditional logit | 2.192417 | 21.79% | 50.00% | 6,230.00 | 3,218.70 | -3,011.30 | -48.34% | -76.15% to -19.13% |
| Benter + market (hindsight) | 2.066442 | 29.49% | 58.97% | 6,230.00 | 6,653.70 | 423.70 | 6.80% | -32.33% to 56.76% |
| Boosted | 2.208155 | 23.08% | 51.28% | 6,230.00 | 4,031.70 | -2,198.30 | -35.29% | -65.03% to -3.89% |
| Boosted + market (hindsight) | 2.066958 | 30.77% | 57.69% | 6,230.00 | 6,648.20 | 418.20 | 6.71% | -31.53% to 56.29% |
| Gaussian probit (common variance) | 2.214886 | 21.79% | 51.28% | 6,230.00 | 4,069.70 | -2,160.30 | -34.68% | -64.75% to -3.42% |
| Gaussian probit + market (hindsight) | 2.067695 | 28.21% | 58.97% | 6,230.00 | 6,768.20 | 538.20 | 8.64% | -30.11% to 58.30% |
| 60% Benter + 40% boosted probability pool | 2.194848 | 20.51% | 48.72% | 6,230.00 | 3,853.70 | -2,376.30 | -38.14% | -72.27% to -3.13% |
| Benter/boosted pool + market (hindsight) | 2.066173 | 30.77% | 58.97% | 6,230.00 | 6,794.70 | 564.70 | 9.06% | -30.98% to 59.21% |
| Market (final odds, hindsight) | 2.067327 | 28.21% | 58.97% | 6,230.00 | 6,768.20 | 538.20 | 8.64% | -30.11% to 58.30% |
| Market (temperature-calibrated, hindsight) | 2.067695 | 28.21% | 58.97% | 6,230.00 | 6,768.20 | 538.20 | 8.64% | -30.11% to 58.30% |

**Tie audit:** an earlier summary used raw floating-point `idxmax`, allowing tiny probability differences and input row order to disagree with the ticket ledger. Top-pick WIN and top3 metrics now use probabilities rounded to 12 decimals, then numeric horse number, matching the fixed WIN ticket policy. The correction changes top-pick metrics, not log loss or portfolio ROI. Final reported metrics must come from the corrected fresh and archived-benchmark reruns, not mixed interim artifacts.

Top-pick top3 is not automatically PLACE hit rate in small fields. Paid PLACE and QPL counts are cross-checked against official winning combinations. The intervals resample whole meetings (10,000 draws, fixed seed), not independent tickets. Eight meetings are too few to establish a reliable edge; one exotic payout can dominate returns. Comparing these models on this season is exploratory, not fresh validation of a selected winner.

## Older Packages Versus Fresh Fixed-Recipe Fits

| Family | Older log loss | Fresh log loss | Older portfolio ROI | Fresh portfolio ROI |
| --- | --- | --- | --- | --- |
| Benter conditional logit | 2.189921 | 2.192417 | -26.67% | -48.34% |
| Boosted | 2.227584 | 2.208155 | -21.91% | -35.29% |
| 60% Benter + 40% boosted probability pool | 2.194265 | 2.194848 | -47.60% | -38.14% |
| Gaussian probit (common variance) | 2.211602 | 2.214886 | -43.32% | -34.68% |

This compares fixed configurations on identical season fields after updating their estimator weights with strictly pre-calibration outcomes. It does not choose a winner by September ROI.

## Returns By Pool

### Benter conditional logit

| Pool | Settled tickets | Stake HKD | Gross HKD | Net HKD | ROI | Hit rate |
| --- | --- | --- | --- | --- | --- | --- |
| WIN | 78 | 780.00 | 493.50 | -286.50 | -36.73% | 21.79% |
| PLACE | 78 | 780.00 | 600.70 | -179.30 | -22.99% | 50.00% |
| QIN | 78 | 780.00 | 537.00 | -243.00 | -31.15% | 8.97% |
| QPL | 77 | 770.00 | 521.50 | -248.50 | -32.27% | 18.18% |
| TRI | 78 | 780.00 | 289.00 | -491.00 | -62.95% | 2.56% |
| TIERCE | 78 | 780.00 | 619.00 | -161.00 | -20.64% | 1.28% |
| FIRST4 | 78 | 780.00 | 158.00 | -622.00 | -79.74% | 2.56% |
| QUARTET | 78 | 780.00 | 0.00 | -780.00 | -100.00% | 0.00% |

### Benter + market (hindsight)

| Pool | Settled tickets | Stake HKD | Gross HKD | Net HKD | ROI | Hit rate |
| --- | --- | --- | --- | --- | --- | --- |
| WIN | 78 | 780.00 | 634.50 | -145.50 | -18.65% | 29.49% |
| PLACE | 78 | 780.00 | 677.70 | -102.30 | -13.12% | 58.97% |
| QIN | 78 | 780.00 | 1,151.00 | 371.00 | 47.56% | 19.23% |
| QPL | 77 | 770.00 | 865.50 | 95.50 | 12.40% | 32.47% |
| TRI | 78 | 780.00 | 636.00 | -144.00 | -18.46% | 6.41% |
| TIERCE | 78 | 780.00 | 2,111.00 | 1,331.00 | 170.64% | 3.85% |
| FIRST4 | 78 | 780.00 | 578.00 | -202.00 | -25.90% | 5.13% |
| QUARTET | 78 | 780.00 | 0.00 | -780.00 | -100.00% | 0.00% |

### Boosted

| Pool | Settled tickets | Stake HKD | Gross HKD | Net HKD | ROI | Hit rate |
| --- | --- | --- | --- | --- | --- | --- |
| WIN | 78 | 780.00 | 674.00 | -106.00 | -13.59% | 23.08% |
| PLACE | 78 | 780.00 | 642.70 | -137.30 | -17.60% | 51.28% |
| QIN | 78 | 780.00 | 673.00 | -107.00 | -13.72% | 8.97% |
| QPL | 77 | 770.00 | 651.00 | -119.00 | -15.45% | 18.18% |
| TRI | 78 | 780.00 | 697.00 | -83.00 | -10.64% | 3.85% |
| TIERCE | 78 | 780.00 | 619.00 | -161.00 | -20.64% | 1.28% |
| FIRST4 | 78 | 780.00 | 75.00 | -705.00 | -90.38% | 1.28% |
| QUARTET | 78 | 780.00 | 0.00 | -780.00 | -100.00% | 0.00% |

### Boosted + market (hindsight)

| Pool | Settled tickets | Stake HKD | Gross HKD | Net HKD | ROI | Hit rate |
| --- | --- | --- | --- | --- | --- | --- |
| WIN | 78 | 780.00 | 679.50 | -100.50 | -12.88% | 30.77% |
| PLACE | 78 | 780.00 | 662.70 | -117.30 | -15.04% | 57.69% |
| QIN | 78 | 780.00 | 1,173.00 | 393.00 | 50.38% | 19.23% |
| QPL | 77 | 770.00 | 888.00 | 118.00 | 15.32% | 32.47% |
| TRI | 78 | 780.00 | 636.00 | -144.00 | -18.46% | 6.41% |
| TIERCE | 78 | 780.00 | 2,111.00 | 1,331.00 | 170.64% | 3.85% |
| FIRST4 | 78 | 780.00 | 498.00 | -282.00 | -36.15% | 3.85% |
| QUARTET | 78 | 780.00 | 0.00 | -780.00 | -100.00% | 0.00% |

### Gaussian probit (common variance)

| Pool | Settled tickets | Stake HKD | Gross HKD | Net HKD | ROI | Hit rate |
| --- | --- | --- | --- | --- | --- | --- |
| WIN | 78 | 780.00 | 534.00 | -246.00 | -31.54% | 21.79% |
| PLACE | 78 | 780.00 | 633.20 | -146.80 | -18.82% | 51.28% |
| QIN | 78 | 780.00 | 649.50 | -130.50 | -16.73% | 10.26% |
| QPL | 77 | 770.00 | 619.00 | -151.00 | -19.61% | 20.78% |
| TRI | 78 | 780.00 | 339.00 | -441.00 | -56.54% | 3.85% |
| TIERCE | 78 | 780.00 | 619.00 | -161.00 | -20.64% | 1.28% |
| FIRST4 | 78 | 780.00 | 676.00 | -104.00 | -13.33% | 2.56% |
| QUARTET | 78 | 780.00 | 0.00 | -780.00 | -100.00% | 0.00% |

### Gaussian probit + market (hindsight)

| Pool | Settled tickets | Stake HKD | Gross HKD | Net HKD | ROI | Hit rate |
| --- | --- | --- | --- | --- | --- | --- |
| WIN | 78 | 780.00 | 599.00 | -181.00 | -23.21% | 28.21% |
| PLACE | 78 | 780.00 | 677.70 | -102.30 | -13.12% | 58.97% |
| QIN | 78 | 780.00 | 1,286.00 | 506.00 | 64.87% | 21.79% |
| QPL | 77 | 770.00 | 987.50 | 217.50 | 28.25% | 35.06% |
| TRI | 78 | 780.00 | 529.00 | -251.00 | -32.18% | 5.13% |
| TIERCE | 78 | 780.00 | 2,111.00 | 1,331.00 | 170.64% | 3.85% |
| FIRST4 | 78 | 780.00 | 578.00 | -202.00 | -25.90% | 5.13% |
| QUARTET | 78 | 780.00 | 0.00 | -780.00 | -100.00% | 0.00% |

### 60% Benter + 40% boosted probability pool

| Pool | Settled tickets | Stake HKD | Gross HKD | Net HKD | ROI | Hit rate |
| --- | --- | --- | --- | --- | --- | --- |
| WIN | 78 | 780.00 | 442.50 | -337.50 | -43.27% | 20.51% |
| PLACE | 78 | 780.00 | 583.20 | -196.80 | -25.23% | 48.72% |
| QIN | 78 | 780.00 | 643.50 | -136.50 | -17.50% | 8.97% |
| QPL | 77 | 770.00 | 600.50 | -169.50 | -22.01% | 18.18% |
| TRI | 78 | 780.00 | 289.00 | -491.00 | -62.95% | 2.56% |
| TIERCE | 78 | 780.00 | 619.00 | -161.00 | -20.64% | 1.28% |
| FIRST4 | 78 | 780.00 | 676.00 | -104.00 | -13.33% | 2.56% |
| QUARTET | 78 | 780.00 | 0.00 | -780.00 | -100.00% | 0.00% |

### Benter/boosted pool + market (hindsight)

| Pool | Settled tickets | Stake HKD | Gross HKD | Net HKD | ROI | Hit rate |
| --- | --- | --- | --- | --- | --- | --- |
| WIN | 78 | 780.00 | 679.50 | -100.50 | -12.88% | 30.77% |
| PLACE | 78 | 780.00 | 676.70 | -103.30 | -13.24% | 58.97% |
| QIN | 78 | 780.00 | 1,279.50 | 499.50 | 64.04% | 20.51% |
| QPL | 77 | 770.00 | 914.00 | 144.00 | 18.70% | 33.77% |
| TRI | 78 | 780.00 | 636.00 | -144.00 | -18.46% | 6.41% |
| TIERCE | 78 | 780.00 | 2,111.00 | 1,331.00 | 170.64% | 3.85% |
| FIRST4 | 78 | 780.00 | 498.00 | -282.00 | -36.15% | 3.85% |
| QUARTET | 78 | 780.00 | 0.00 | -780.00 | -100.00% | 0.00% |

### Market (final odds, hindsight)

| Pool | Settled tickets | Stake HKD | Gross HKD | Net HKD | ROI | Hit rate |
| --- | --- | --- | --- | --- | --- | --- |
| WIN | 78 | 780.00 | 599.00 | -181.00 | -23.21% | 28.21% |
| PLACE | 78 | 780.00 | 677.70 | -102.30 | -13.12% | 58.97% |
| QIN | 78 | 780.00 | 1,286.00 | 506.00 | 64.87% | 21.79% |
| QPL | 77 | 770.00 | 987.50 | 217.50 | 28.25% | 35.06% |
| TRI | 78 | 780.00 | 529.00 | -251.00 | -32.18% | 5.13% |
| TIERCE | 78 | 780.00 | 2,111.00 | 1,331.00 | 170.64% | 3.85% |
| FIRST4 | 78 | 780.00 | 578.00 | -202.00 | -25.90% | 5.13% |
| QUARTET | 78 | 780.00 | 0.00 | -780.00 | -100.00% | 0.00% |

### Market (temperature-calibrated, hindsight)

| Pool | Settled tickets | Stake HKD | Gross HKD | Net HKD | ROI | Hit rate |
| --- | --- | --- | --- | --- | --- | --- |
| WIN | 78 | 780.00 | 599.00 | -181.00 | -23.21% | 28.21% |
| PLACE | 78 | 780.00 | 677.70 | -102.30 | -13.12% | 58.97% |
| QIN | 78 | 780.00 | 1,286.00 | 506.00 | 64.87% | 21.79% |
| QPL | 77 | 770.00 | 987.50 | 217.50 | 28.25% | 35.06% |
| TRI | 78 | 780.00 | 529.00 | -251.00 | -32.18% | 5.13% |
| TIERCE | 78 | 780.00 | 2,111.00 | 1,331.00 | 170.64% | 3.85% |
| FIRST4 | 78 | 780.00 | 578.00 | -202.00 | -25.90% | 5.13% |
| QUARTET | 78 | 780.00 | 0.00 | -780.00 | -100.00% | 0.00% |

## Season Hindsight WIN EV

Total approximate expected net profit over each model's selected HKD10 WIN tickets: sum(pwin x rounded final WIN odds x 10 - 10). These are **hindsight prices, not actionable pre-off EV** or realized returns. Realized net profit uses exact official dividends; no second takeout deduction is applied.

| Model | Settled WIN tickets | WIN stake HKD | Hindsight total WIN EV HKD | Realized WIN net HKD |
| --- | --- | --- | --- | --- |
| Benter conditional logit | 78 | 780.00 | 9.57 | -286.50 |
| Boosted | 78 | 780.00 | -3.16 | -106.00 |
| Gaussian probit (common variance) | 78 | 780.00 | -19.80 | -246.00 |
| 60% Benter + 40% boosted probability pool | 78 | 780.00 | -16.24 | -337.50 |

## Final Fit Dates

Fresh fits use **13,390 races / 165,071 runners**, ending **2026-01-11 00:00:00**. The independent calibration period starts 2026-01-14 00:00:00. No September or October outcome updates estimator weights. All fixed-recipe fitted packages are saved separately under `final-fits/`.

Historical horse age remains **NaN in the training dataset, not zero-filled**. The training imputer found no finite values and dropped the four age columns (`horse_age`, `age_rank`, `poly_age_sq`, `poly_age_distance`); no historical age effect was learned.

## Progression And Capital

Default pool-model all-pool paper portfolio: minimum capital needed for this exact fixed-stake schedule was **HKD3,305.30**. HKD1,000 could finance it: **False**. Maximum paper drawdown: HKD3,225.30. If a schedule exhausts its capital, an algebraic ending balance is not achievable bankroll growth without extra funding. There is no compounding, Kelly staking or automatic reinvestment in this fixed-HKD10 test.

| Meeting | Stake HKD | Gross HKD | Net HKD | Cumulative net HKD | Cumulative ROI |
| --- | --- | --- | --- | --- | --- |
| 2026-09-06 00:00:00+00:00 | 790.00 | 397.10 | -392.90 | -392.90 | -49.73% |
| 2026-09-09 00:00:00+00:00 | 640.00 | 31.00 | -609.00 | -1,001.90 | -70.06% |
| 2026-09-13 00:00:00+00:00 | 800.00 | 216.00 | -584.00 | -1,585.90 | -71.12% |
| 2026-09-16 00:00:00+00:00 | 640.00 | 96.50 | -543.50 | -2,129.40 | -74.20% |
| 2026-09-23 00:00:00+00:00 | 720.00 | 541.00 | -179.00 | -2,308.40 | -64.30% |
| 2026-09-27 00:00:00+00:00 | 880.00 | 217.10 | -662.90 | -2,971.30 | -66.47% |
| 2026-10-01 00:00:00+00:00 | 880.00 | 1,098.50 | 218.50 | -2,752.80 | -51.45% |
| 2026-10-04 00:00:00+00:00 | 880.00 | 1,256.50 | 376.50 | -2,376.30 | -38.14% |

Full race-by-race progression is in the `*-bankroll.csv` and `*-tickets.csv` files, including every losing ticket.

## October 7 Forecasts

Default fundamental probability pool; these are model estimates, not guarantees. Refresh scratches, jockey changes, going and prices before considering a ticket. A higher win probability does not imply positive EV. Break-even prices below contain no uncertainty cushion.

| Race | Distance m | Ranked top3: win probability | Top WIN pick placing probability | Top WIN pick break-even gross HKD/10 |
| --- | --- | --- | --- | --- |
| 1 | 1800 | #1 CAN'T GO WONG (22.58%); #9 GOLDEN FORTUNE (17.43%); #5 OCEAN IMPACT (10.89%) | 57.30% | 44.28 |
| 2 | 1200 | #10 LOVING VIBES (27.29%); #4 WINNING MONEY (13.77%); #2 LUCKY MCQUEEN (12.17%) | 65.32% | 36.65 |
| 3 | 1650 | #6 DECISION LINK (18.90%); #4 SUNDAY'S SERENADE (17.15%); #2 RAGGA BOMB (8.97%) | 50.21% | 52.91 |
| 4 | 1650 | #8 ABSOLUTE HONOUR (18.17%); #2 SKY DEEP (16.92%); #10 PRECISION HOPE (16.72%) | 49.86% | 55.04 |
| 5 | 1000 | #4 SUPERB KING (22.54%); #8 GEORGIAN SIGMA (17.04%); #12 MAPOGO (11.03%) | 57.52% | 44.36 |
| 6 | 1200 | #9 NO OTHER CHOICE (20.27%); #10 KWAI CHUNG TALENTS (15.19%); #5 CALIFORNIA BAY (11.12%) | 52.78% | 49.32 |
| 7 | 1200 | #9 BRIGHT DAY (18.17%); #5 TARGET AUDIENCE (16.09%); #2 FLYING WROTE (12.82%) | 48.90% | 55.03 |
| 8 | 1650 | #1 DAZZLING FIT (19.60%); #2 POPE CODY (16.22%); #9 AUDACIOUS PURSUIT (10.13%) | 51.47% | 51.02 |
| 9 | 1200 | #6 PRESTIGE ALWAYS (20.11%); #9 LIVE WIRE (18.11%); #7 THE HEIR (15.67%) | 53.77% | 49.73 |

### Indicative Combined-Market Snapshot

Default pool fundamentals combined forward-only with captured public WIN quotes using the pre-fitted market blend. This snapshot forecast is **not a validated pre-off betting strategy**; prices and fields can change. No October 7 outcomes refit the blend. Full probabilities and unit-verified WIN EV are in `evaluation/tomorrow-market-blends.csv`; tickets are in `evaluation/tomorrow-market-blend-combinations.csv`.

Across all models, 432 unit-verified combined WIN snapshot EVs are available; 432 are negative, with maximum HKD-0.4036 per HKD10. Positive fundamental-only snapshot EV can be optimistic and is not treated as a confidence-qualified betting edge. The combined forecast and its EV are distinct from those fundamental-only estimates.

| Race | Combined-market top3 | Top pick raw WIN quote | Top pick snapshot WIN EV HKD/10 |
| --- | --- | --- | --- |
| 1 | #1 CAN'T GO WONG (WIN 22.52%, PLACE 52.28%); #4 KINGLY DEMEANOR (WIN 16.73%, PLACE 43.38%); #9 GOLDEN FORTUNE (WIN 10.29%, PLACE 30.95%) | 4.0 | -0.99 |
| 2 | #10 LOVING VIBES (WIN 35.89%, PLACE 69.57%); #6 HAYDAY (WIN 10.86%, PLACE 34.20%); #2 LUCKY MCQUEEN (WIN 10.37%, PLACE 33.09%) | 2.6 | -0.67 |
| 3 | #10 FORTUNE STAR (WIN 34.53%, PLACE 69.64%); #4 SUNDAY'S SERENADE (WIN 14.25%, PLACE 42.39%); #6 DECISION LINK (WIN 12.18%, PLACE 38.13%) | 2.5 | -1.37 |
| 4 | #2 SKY DEEP (WIN 20.61%, PLACE 49.75%); #10 PRECISION HOPE (WIN 16.82%, PLACE 43.72%); #1 VIVA GRACIOUSNESS (WIN 13.75%, PLACE 38.16%) | 4.3 | -1.14 |
| 5 | #4 SUPERB KING (WIN 29.17%, PLACE 63.44%); #8 GEORGIAN SIGMA (WIN 23.01%, PLACE 55.91%); #9 GOLDEN ACE (WIN 11.59%, PLACE 36.05%) | 3.1 | -0.96 |
| 6 | #9 NO OTHER CHOICE (WIN 28.12%, PLACE 60.80%); #10 KWAI CHUNG TALENTS (WIN 16.09%, PLACE 43.62%); #5 CALIFORNIA BAY (WIN 10.87%, PLACE 33.33%) | 3.2 | -1.00 |
| 7 | #5 TARGET AUDIENCE (WIN 22.11%, PLACE 52.11%); #11 DARYL FLASH (WIN 17.92%, PLACE 45.73%); #2 FLYING WROTE (WIN 14.92%, PLACE 40.53%) | 4.0 | -1.16 |
| 8 | #1 DAZZLING FIT (WIN 19.45%, PLACE 47.45%); #4 DO YOU JUST (WIN 14.97%, PLACE 39.97%); #11 URANUS STAR (WIN 11.12%, PLACE 32.47%) | 4.6 | -1.05 |
| 9 | #7 THE HEIR (WIN 22.23%, PLACE 51.67%); #1 AURIO (WIN 16.25%, PLACE 42.35%); #6 PRESTIGE ALWAYS (WIN 11.93%, PLACE 34.23%) | 4.0 | -1.11 |

### Top Exotic Combinations

One straight ticket in each cell costs HKD10. QIN/QPL/TRIO/FIRST4 are unordered; TIERCE/QUARTET order is exactly as shown. These are maximum-probability combinations, **not price-qualified value bets**. Probabilities are derived using Benter-corrected finish-order exponents fitted only on the rated calibration fields. **FIRST4 and QUARTET extrapolate the third-position exponent to the fourth stage; no separate fourth-position exponent is learned.**

| Race | QIN | QPL | TRI | TIERCE | FIRST4 | QUARTET |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1/9: 10.23%; break-even 97.79; snapshot EV unavailable | 1/9: 24.23%; break-even 41.27; snapshot EV unavailable | 1/5/9: 4.53%; break-even 220.78; snapshot EV unavailable | 1/9/5: 0.91%; break-even 1,101.27; snapshot EV unavailable | 1/5/6/9: 2.55%; break-even 392.06; snapshot EV HKD16.27 | 1/9/5/6: 0.15%; break-even 6,456.48; snapshot EV unavailable |
| 2 | 10/4: 9.83%; break-even 101.77; snapshot EV unavailable | 10/4: 23.75%; break-even 42.10; snapshot EV unavailable | 10/2/4: 5.07%; break-even 197.40; snapshot EV unavailable | 10/4/2: 1.04%; break-even 964.85; snapshot EV unavailable | 10/2/4/7: 3.18%; break-even 314.47; snapshot EV HKD-2.69 | 10/4/2/7: 0.20%; break-even 5,071.30; snapshot EV unavailable |
| 3 | 4/6: 8.17%; break-even 122.38; snapshot EV unavailable | 4/6: 20.15%; break-even 49.64; snapshot EV unavailable | 2/4/6: 2.86%; break-even 349.95; snapshot EV unavailable | 6/4/2: 0.56%; break-even 1,773.74; snapshot EV unavailable | 10/2/4/6: 1.58%; break-even 634.28; snapshot EV HKD0.88 | 6/4/2/10: 0.09%; break-even 11,216.10; snapshot EV unavailable |
| 4 | 2/8: 7.65%; break-even 130.79; snapshot EV unavailable | 2/8: 19.91%; break-even 50.22; snapshot EV unavailable | 10/2/8: 5.45%; break-even 183.41; snapshot EV unavailable | 8/2/10: 0.93%; break-even 1,077.80; snapshot EV unavailable | 10/12/2/8: 5.01%; break-even 199.60; snapshot EV HKD-0.48 | 8/2/10/12: 0.24%; break-even 4,225.93; snapshot EV unavailable |
| 5 | 4/8: 9.92%; break-even 100.83; snapshot EV unavailable | 4/8: 23.93%; break-even 41.79; snapshot EV unavailable | 12/4/8: 4.47%; break-even 223.56; snapshot EV unavailable | 4/8/12: 0.89%; break-even 1,120.17; snapshot EV unavailable | 12/4/8/9: 3.11%; break-even 321.56; snapshot EV HKD21.72 | 4/8/12/9: 0.18%; break-even 5,557.33; snapshot EV unavailable |
| 6 | 10/9: 7.73%; break-even 129.36; snapshot EV unavailable | 10/9: 19.28%; break-even 51.87; snapshot EV unavailable | 10/5/9: 3.42%; break-even 292.58; snapshot EV unavailable | 9/10/5: 0.66%; break-even 1,526.45; snapshot EV unavailable | 10/5/7/9: 1.92%; break-even 520.00; snapshot EV HKD-5.19 | 9/10/5/7: 0.11%; break-even 9,328.82; snapshot EV unavailable |
| 7 | 5/9: 7.27%; break-even 137.59; snapshot EV unavailable | 5/9: 18.46%; break-even 54.17; snapshot EV unavailable | 2/5/9: 3.74%; break-even 267.38; snapshot EV unavailable | 9/5/2: 0.68%; break-even 1,472.87; snapshot EV unavailable | 12/2/5/9: 2.51%; break-even 398.29; snapshot EV HKD-2.72 | 9/5/2/12: 0.13%; break-even 7,759.14; snapshot EV unavailable |
| 8 | 1/2: 8.01%; break-even 124.83; snapshot EV unavailable | 1/2: 19.78%; break-even 50.55; snapshot EV unavailable | 1/2/9: 3.19%; break-even 313.39; snapshot EV unavailable | 1/2/9: 0.62%; break-even 1,613.42; snapshot EV unavailable | 1/10/2/9: 1.90%; break-even 526.77; snapshot EV HKD4.81 | 1/2/9/10: 0.11%; break-even 9,503.85; snapshot EV unavailable |
| 9 | 6/9: 9.28%; break-even 107.80; snapshot EV unavailable | 6/9: 23.23%; break-even 43.04; snapshot EV unavailable | 6/7/9: 6.21%; break-even 161.01; snapshot EV unavailable | 6/9/7: 1.11%; break-even 900.51; snapshot EV unavailable | 1/6/7/9: 5.71%; break-even 174.98; snapshot EV HKD114.59 | 6/9/7/1: 0.29%; break-even 3,451.15; snapshot EV HKD126.71 |

### Other Models: Top WIN Picks

| Model | R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 | R9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Benter conditional logit | #9 GOLDEN FORTUNE (23.74%) | #10 LOVING VIBES (23.68%) | #6 DECISION LINK (17.72%) | #8 ABSOLUTE HONOUR (23.31%) | #4 SUPERB KING (21.94%) | #9 NO OTHER CHOICE (16.44%) | #9 BRIGHT DAY (17.45%) | #2 POPE CODY (17.12%) | #6 PRESTIGE ALWAYS (20.02%) |
| Boosted | #1 CAN'T GO WONG (29.60%) | #10 LOVING VIBES (31.66%) | #6 DECISION LINK (22.02%) | #8 ABSOLUTE HONOUR (17.75%) | #4 SUPERB KING (23.10%) | #9 NO OTHER CHOICE (21.92%) | #5 TARGET AUDIENCE (20.96%) | #1 DAZZLING FIT (21.36%) | #9 LIVE WIRE (22.64%) |
| 60% Benter + 40% boosted probability pool | #1 CAN'T GO WONG (22.58%) | #10 LOVING VIBES (27.29%) | #6 DECISION LINK (18.90%) | #8 ABSOLUTE HONOUR (18.17%) | #4 SUPERB KING (22.54%) | #9 NO OTHER CHOICE (20.27%) | #9 BRIGHT DAY (18.17%) | #1 DAZZLING FIT (19.60%) | #6 PRESTIGE ALWAYS (20.11%) |
| Gaussian probit (common variance) | #1 CAN'T GO WONG (20.27%) | #10 LOVING VIBES (21.98%) | #4 SUNDAY'S SERENADE (20.90%) | #8 ABSOLUTE HONOUR (20.23%) | #4 SUPERB KING (21.36%) | #9 NO OTHER CHOICE (16.88%) | #9 BRIGHT DAY (17.85%) | #1 DAZZLING FIT (19.74%) | #6 PRESTIGE ALWAYS (19.89%) |

## Calibration And Provenance

| Model | Fundamental exponent | Market exponent | Prior development log loss | Package ID |
| --- | --- | --- | --- | --- |
| Benter conditional logit | 0.03885547 | 1.08586887 | 2.16165385 | 8ad82c6f6945bb7d |
| Boosted | 0.11743107 | 1.03494535 | 2.16143028 | 35e10a10d40e4f23 |
| 60% Benter + 40% boosted probability pool | 0.07212794 | 1.06466596 | 2.15928116 | 7c787d136fa8696d |
| Gaussian probit (common variance) | 0.00000000 | 1.11184941 | 2.17429126 | bc9bf70b48135d65 |

Zero fundamental weight is allowed: it means that calibration found no additional conditional signal given the market in that calibration cohort. It is not proof the optimizer is broken or that probabilities should be forced into the blend. The audit did find a numerical floor/finite-difference flaw in the previous implementation; the repaired path uses log-softmax, analytic gradients and convergence checks. Original calibration predictions were not preserved for every older run, so the retrospective gradient audit is descriptive, not proof of a historical fitting error.

Input feature SHA256: `08e8c88f03b4d4a570774b1dd165857695c137f7cbbf6e9584b02f54b403277e`. Exact inference environment: `{"numpy": "2.5.3", "pandas": "2.3.3", "scikit-learn": "1.9.1", "scipy": "1.18.1"}`. No dependency guards were bypassed. Trusted packages were executed in their original remote environment without modifying running campaigns.

All four models and all ten evaluated variants are shown above, including each variant's pool ROI. The compact runner ranking is `docs/OCT07_2026_RANKED_PREDICTIONS.csv` (four models x 108 starters = 432 rows for the captured fields); it includes WIN and paid-PLACE probabilities, raw WIN quotes, unit-verified snapshot EV and quote timestamps.

Artifacts: `artifacts/race-readiness-20261007`. Older estimators and evaluations remain archived under `frozen-benchmark/`. `official/` contains raw source captures and coverage; `models/frozen-candidates.json` contains selection and package hashes; `preparation.json` documents data/input coverage; `artifacts/race-readiness-20261007/fresh-predictions/readback.json` binds model outputs to the prepared feature hash; `evaluation/` contains all runners, tickets, pool combinations, daily totals, capital curves and calibration parameters.

## Reproduction

```sh
.venv/bin/python -m scripts.collect_season_2026 --help
.venv/bin/python -m scripts.evaluate_season_2026 --stage prepare --history .tmp/season-official-history.parquet
# Fresh fixed-recipe fitting and inference: scripts/refit_season_2026.py; see --help.
.venv/bin/python -m scripts.evaluate_season_2026 --stage evaluate --predictions-dir artifacts/race-readiness-20261007/fresh-predictions
.venv/bin/python -m scripts.forecast_season_market
.venv/bin/python -m scripts.report_season_2026
```

Re-evaluation rejects stale prediction-input hashes, changed CSVs, identity mismatches and incomplete acquisition. After any racecard refresh, rebuild features and regenerate all predictions before evaluating.
