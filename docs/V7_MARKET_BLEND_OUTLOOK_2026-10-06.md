# V7 Market Blend Outlook: 2026-10-06

Read-only evidence snapshot: **2026-10-06 11:14:47 UTC / 19:14:47 Shanghai**. Campaigns were active during collection; this is a bounded persisted-result snapshot, not final totals. Remote account/root: `imaopt@100.95.24.121:/home/imaopt/research-v2/campaigns`. No code, campaigns, services or credentials were changed; no inference API was called and no model package was unpickled.

## Numeric Outlook

**No demonstrated incremental market edge.** All **149 completed fitted market blends** on the main win contract selected **fundamental coefficient 0 in all three folds**. Another **245 completed win trials** have no market blend, including newer model-only champions. Their possible incremental contribution after blending is **unknown**, not established by their standalone improvement.

The best recorded blend has log loss **2.000518840884**, versus calibrated market **2.000519051132** and raw market **2.001114117835**. Its nominal advantage over calibrated market is only **0.000000210249** in the logged equal-fold metric, with zero fundamental contribution. This is consistent with differences between two numerical optimizations of market temperature. Both race and meeting bootstrap intervals include zero.

The strongest completed model-only result in this snapshot is **2.159281159243**, a fixed 60% Benter / 40% boosted probability pool. It remains worse than calibrated market by **0.158762108110** in logged loss. This does not rule out complementary residual information; that candidate has no fitted market blend and lacks saved leakage-safe calibration predictions.

## Matched Contract And Population

- Target: `win_probability`, unit `1`, metric contract **3**, natural-log categorical winner loss. Exactly one winner per scored race; exclude dead heats and missing/multiple-winner races. Selected champions record zero dataset exclusions.
- Dataset: `dataset-19aaa959e948d6e145ce62c8160413087b46ba3c2578e0de15c452054d9a83db`; byte hash `2321218048ffb1f2b9d9ea9a357ec8a753775420f9808385dc323741ad1795a1`.
- Manifest SHA256: `bbe54689bf9e13d1b1afd06122baf1caa9c67e61ddb007be106ea33eb771ce65`; strict availability; comparison contract `comparison-888e2c897e06477a9b3ad8cdcbd89fce1ef3d7be58c7c6e5e60ed8efc3ff3dba`.
- Protocol: `43d605c2487bd2f1`; file hash `0de67fa0864a07d739f567327001e41d619faa32c23efe620c88b87b4583f2b1`. Latest three expanding chronological folds, minimum training 9,000 races, nominal calibration/score 500 races, whole-meeting boundaries. Dataset has 13,924 eligible races / 171,782 runners, 2001-04-14 through 2025-11-12.
- Actual score population: `evaluated-5b0a1fbf4848b77192c1bf08336f6ce678a1446ffe6d13af432b9c6ff729bd76`, **1,502 races / 18,075 runners / 159 meeting dates**, **2023-09-13 through 2025-05-18**. All 394 main win results share dataset, protocol, metric version and population hashes. The four shortlisted CSVs additionally have identical scored runner keys and labels, verified directly.
- Raw market is the persisted, race-normalized reciprocal recorded final `win_odds`, following `ima/data.py`; this is a benchmark probability construction, not independently verified settlement return semantics or a tradable quote. Calibrated market uses temperature fitted separately on each preceding calibration window. Fundamental calibration, if enabled, also fits there. Training windows and graph refit scopes differ between recipes; matched scores do not isolate those treatments.

| Fold | Training End | Calibration Dates / Races | Scoring Dates / Races |
| --- | --- | --- | --- |
| fold-001 | 2023-01-15 | 2023-01-18 to 2023-09-10 / 505 | 2023-09-13 to 2024-03-16 / 500 |
| fold-002 | 2023-09-10 | 2023-09-13 to 2024-03-16 / 500 | 2024-03-20 to 2024-11-13 / 500 |
| fold-003 | 2024-03-16 | 2024-03-20 to 2024-11-13 / 500 | 2024-11-17 to 2025-05-18 / 502 |

## Completed Coverage

Scanning `agentic_v[67]*/trials/*/result.json` found **411 completed artifacts**: 394 main-contract win results, 10 ranking results on another contract, and 7 small win canaries on another contract. Counts are attempts, not unique recipes or independent discoveries. Smaller/empty technical campaigns contain no completed result artifact in this scan. Other generations are outside this audit.

| Exact Campaign | Completed All Targets | Main Win | Fitted Market Blends |
| --- | ---: | ---: | ---: |
| agentic_v6_luna_session_20261004 | 5 | 5 | 5 |
| agentic_v6_official_controls_c1_parent | 5 | 5 | 5 |
| agentic_v6_openrouter_7da5b0d_r2 | 2 | 2 | 2 |
| agentic_v6_openrouter_ready_8d61102 | 26 | 26 | 25 |
| agentic_v6_openrouter_continuous_4864156 | 87 | 84 | 76 |
| agentic_v6_reliability_0a50bf4 | 167 | 162 | 1 |
| agentic_v7_probit_runtime_efe1f6b | 111 | 109 | 35 |
| agentic_v7_probit_runtime_verification_efe1f6b | 1 | 1 | 0 |
| agentic_v6_reliability_science_0a50bf4 | 3 | 0 | 0 |
| agentic_v6_reliability_science_e45cb21 | 2 | 0 | 0 |
| agentic_v6_reliability_science_f6ec04f | 2 | 0 | 0 |

The repaired successor historically retains the technical `agentic_v6_reliability_0a50bf4` directory although the prior debrief calls it V7. Exact IDs above remove the naming ambiguity. The seven canaries use protocol `490ec493493b6935` and population `evaluated-5f701112d48fc2f31a1e85c32f7dd30e2d220e4549b75a02ff2b63ac9d325972`; they are not included in the main score table or uncertainty estimates. Ranking is likewise excluded.

## Scores And Exact Runs

These are **logged equal-fold means**, faithfully reconstructed from CSVs. `Selected` is a blend only where a blend exists; otherwise it is the model-only endpoint.

| Short Name | Fundamental | Raw Market | Calibrated Market | Selected | Market Blend |
| --- | ---: | ---: | ---: | ---: | --- |
| Best recorded blend | 2.184230845180 | 2.001114117835 | 2.000519051132 | 2.000518840884 | Zero fundamental weight |
| Best fundamental, repaired successor | 2.159281159243 | 2.001114117835 | 2.000519051132 | 2.159281159243 | Absent |
| Earlier V6 fundamental champion | 2.166148682757 | 2.001114117835 | 2.000519051132 | 2.000519164321 | Zero fundamental weight |
| V7 runtime fundamental champion | 2.161430279163 | 2.001114117835 | 2.000519051132 | 2.161430279163 | Absent |

- Best recorded blend: `agentic_v7_probit_runtime_efe1f6b`, `attempt-7f208b95c061bf318c3550478cb6cd8697cd200068058db828ff9a259e8ccf4d`.
- Best fundamental: `agentic_v6_reliability_0a50bf4`, `attempt-a065c94d54bf34e258e17760cc4d6ae5af74cbcc066243771a5849783cc71b1e`.
- Earlier V6 champion: `agentic_v6_openrouter_ready_8d61102`, `attempt-c49b9ecf6a2de6ab473a920d518a5b36deda809368ca7c59088241ee129ee181`.
- V7 runtime champion: `agentic_v7_probit_runtime_efe1f6b`, `attempt-a615ca6df92f6fc5eb6729bbaf2905a364e0f95520ee6aeea732c3aadabf52c5`.

Exact remote evidence for each: `<campaign-root>/<campaign>/trials/<attempt>/result.json`, `predictions.csv`, `protocol.json`, `package/recipe.json`, `package/manifest.json`; the graph champion also has `graph-fits.json`. Safe JSON/CSV copies and recorded prediction hashes are in the audit's own `.tmp` files.

## Winning Recipes And Coefficients

Best blend uses `notebook-rich-v2`, seed 42, `trailing_3_years`, conditional logit `l2=0.04460639973478976`, `max_iter=1655`, temperature calibration and `market_softmax`. Transforms: signed-log prior win/top3 and latest/three-race speed; race ranks of three speed summaries, win/top3 rates and lengths behind; race centering of latest/five-race speed, win rate and lengths behind; latest-speed x win-rate interaction. Full exact recipe is retained as `.tmp/market_blend_outlook_20261006_best_blend/package_recipe.json`.

Formula: `p_selected(i) = p_fundamental(i)^a * p_raw_market(i)^b / sum_j(p_fundamental(j)^a * p_raw_market(j)^b)`. Coefficients are nonnegative exponents, **not convex percentage weights**; fitted bounds are [0,4] each.

| Fold | Fundamental Exponent a | Market Exponent b | Market-Only Temperature T |
| --- | ---: | ---: | ---: |
| fold-001 | 0 | 1.1316018686562939 | 0.8836922290039235 |
| fold-002 | 0 | 1.0314510599260647 | 0.969506487085256 |
| fold-003 | 0 | 1.1336107592971265 | 0.8821395550948053 |

With `a=0`, the blend is simply market raised to `b`, while calibrated market is market raised to `1/T`. Independent optimizers make these almost identical. The saved selected probabilities were reconstructed from the coefficients and matched within 1e-12 absolute tolerance. There is no model information in this winning endpoint.

Best fundamental uses graph `benter-boosted-pool-v6g`: fixed arithmetic pool **0.6 Benter / 0.4 boosted**, notebook-rich-v2 plus six past-race speed mean/recency/std/support/trend predictors, all history, seed 42, no transforms, no calibration and no market blend. Benter: `l2=.06`, `max_iter=2930`. Boosted: learning rate .03, depth 6, iterations 400, leaves 31, minimum leaf 80, L2 5. Graph fitting refits both components on **training + calibration** before scoring (six physical fits; no learned pooling weights or saved OOF predictions). This training scope differs from flat recipes and must be preserved as a treatment, not attributed solely to ensembling.

V7 runtime fundamental champion is flat boosted: learning rate .01432813077434483, depth 7, iterations 1686, leaves 18, minimum leaf 53, L2 16.41647303053197. Full feature/transformation choices for all four shortlisted runs are retained in their JSON recipes. Its blend remains untested.

## Paired Uncertainty

Reuse: `ima.feature_studies.paired_feature_report` / `race_objectives`, NumPy, pandas and `ima.modeling.blend_probabilities`. **2,000 bootstrap resamples, seed 17**, paired losses on identical runners/labels. Race bootstrap samples races; meeting sensitivity in the existing helper samples equal-weight meeting-date mean deltas. An additional date-cluster bootstrap samples entire dates and retains race-weighted loss by resampling meeting loss sums/counts. Dates are the grouping proxy; the saved CSVs omit venue. These intervals are descriptive after adaptive development selection, not an independent confirmation or multiple-testing correction.

The estimates below use **equal race weight**, so they differ slightly from logged equal-fold means (500/500/502 races). Negative delta favors the first endpoint.

| Comparison | Race-Weighted Delta | Race 95% CI | Meeting-Cluster 95% CI |
| --- | ---: | --- | --- |
| Best blend - raw market | -0.0005973951 | [-0.00497764, 0.00376358] | [-0.00466746, 0.00344265] |
| Calibrated market - raw market | -0.0005971852 | [-0.00497758, 0.00376376] | [-0.00466762, 0.00344313] |
| Best blend - calibrated market | -2.09850e-7 | [-5.53441e-7, 1.07290e-7] | [-5.06730e-7, 9.03342e-8] |
| Best blend - its fundamental | -0.183728805 | [-0.21115022, -0.15634379] | [-0.21213521, -0.15723927] |
| Best fundamental - calibrated market | +0.158774741 | [0.13413321, 0.18377041] | [0.13429454, 0.18367084] |

The equal-meeting sensitivity for blend minus calibrated market is also inconclusive: **[-5.31483e-7, 6.66806e-8]**. Market decisively improves on the tested standalone fundamentals; neither market calibration nor blend-versus-market improvement clears either bootstrap interval. Earlier V6 champion's blend is slightly worse than calibrated market: race-weighted delta **+1.13227e-7**, race CI **[-2.00134e-8, 2.48945e-7]**, also inconclusive. No paired result here supports incremental fundamental alpha.

## Missing Blend Evidence And Exact Fit Inputs

**Completed evidence:** 149 fitted blends, all a=0. **Unavailable evidence:** leakage-safe calibration predictions for the 245 unblended candidates. Recipe inspection confirms `blend.kind=none` and no market-blend graph node in these 245. The graph-prediction caches in inspected V6/V7 campaigns contain no files; no named calibration prediction export was found under their trials. The champion graph's `oof_artifacts` and `oof_populations` are explicitly empty. Preparation Joblib caches contain inputs/fitted transforms, not inspected prediction exports; they were not loaded.

Exact source dataset directory:
`/home/imaopt/research-v2/campaigns/agentic_v6_research/datasets/datasets/dataset-19aaa959e948d6e145ce62c8160413087b46ba3c2578e0de15c452054d9a83db`.

Exact required populations are the `train_race_ids`, `calibration_race_ids` and `score_race_ids` arrays in each shortlisted `protocol.json`; dates/counts are tabulated above. For a flat champion, fit its saved recipe and preprocessing on training only, generate **calibration fundamental probabilities** with those frozen fits, and pair them by race/horse with calibration `market_probability` and `target_win`. Fit temperature and `MarketBlend.fit` on calibration only; apply frozen coefficients to scoring predictions. The final saved package is not a substitute for earlier fold models.

For the graph champion, predictions on its calibration rows from the saved final graph would be **in-sample**, because final component refits include calibration. A valid next experiment must generate calibration predictions from training-only component fits or verified forward-OOF fits that exclude each predicted race, then fit calibration/blend using only those rows. Refitting training+calibration afterward changes the prediction distribution and must be an explicitly tested protocol choice. No such refit/export/fitting was performed in this audit.

Fold-001 scored races later become fold-002 calibration, and fold-002 scores become fold-003 calibration. That is normal chronological reuse for later deployment, but **do not use the pooled score CSV as one blend-fitting dataset and then score that same pooled population**. Do not select a blend coefficient on scoring races. This audit performed no fit on any scoring population.

An untouched confirmation is still required after development selection. The manifest reserves a nominal 500-race confirmation concept but its `final_confirmation_race_ids` list is empty. This audit has no evaluated independent confirmation result to report. Unknown blend value of stronger recent fundamentals is the useful remaining question; it is an unexecuted experiment, not evidence of edge.

## Historical EV Boundary

**No EV/ROI was calculated.** The audited prediction exports contain market probabilities but no actual WIN settlement dividends, unnormalized payout returns, or verified timestamped pre-off quotes. Normalized market probability cannot recover the absolute payout multiplier. Repository ingestion copies historical `Win Odds` into `win_odds` and normalizes reciprocal values; that code does not establish an exact per-unit settlement return for these scored races.

HKJC's [Horse Race Betting Rules, Part 3](https://special.hkjc.com/e-win/en-US/betting-info/racing/betting-rules/-/media/Sites/JCBW/Special/betting-rules/Horse_Race_Rule_3_Eng_20260330) specify totalisator dividend allocation from Net Pool, unit-bet scaling and rounding/minimum dividends (Rules 3.6-3.8). Accordingly, any later settlement analysis should join the actual declared WIN dividend and its unit, use **gross return D = dividend / unit stake**, and net unit result **winner * D - 1**. Do not subtract takeout again from a declared payout. A rounded displayed result-page odds value is not automatically the exact declared dividend multiplier.

Actual dividend joins, historical rule/unit consistency, void/refund treatment, and executable quote timing remain unverified for this population. Even a positive final-dividend replay would be observational historical performance, not executable profit. No odds threshold, bet subset or staking policy was tuned here.

## Reproduction And Verification

All helpers, fetched safe artifacts and analysis outputs use the exclusive prefix `.tmp/market_blend_outlook_20261006_`. From the local iMa repository:

```sh
.venv/bin/python .tmp/market_blend_outlook_20261006_collect.py
.venv/bin/python .tmp/market_blend_outlook_20261006_analyze.py
.venv/bin/python .tmp/market_blend_outlook_20261006_probe.py
.venv/bin/python .tmp/market_blend_outlook_20261006_verify.py
```

`collect` records its SSH/read-only Python command and complete result inventory. `analyze` fetches only safe CSV/JSON artifacts and performs paired comparisons. `probe` records manifest, recipes, cache/export availability, champion graph-fit evidence and the deployed graph source at revision `0a50bf4aa61874b199074499f3e2235b893aac2e`. `verify` checks contracts/manifest hashes and adds meeting-cluster intervals. Re-running collection captures a newer live campaign population; retaining the saved inventory reproduces this snapshot's selections.

Machine-readable evidence: `.tmp/market_blend_outlook_20261006_inventory.json`, `_analysis.json`, `_recipes.json`, `_prediction_availability.json`, `_dataset_manifest.json`, four named shortlist directories and `_best_fundamental_graph_fits.json`. The analysis JSON retains all reported coefficients, run lineage, endpoint metrics and bootstrap intervals.

**Verification passed:** four prediction SHA256 checks against result lineage; identical scored key/label populations; no duplicate runners; one winner per race; finite [0,1] race-normalized probabilities; score membership equals protocol arrays; train/calibration/score sets disjoint within each fold; recorded blend reconstruction; logged equal-fold metric reconstruction within 1e-12; common dataset/protocol/population/metric contracts; dataset-manifest byte hash. Remote reads were own-user file reads only. No paid inference, training, deployment, wagering, package deserialization or service operation occurred.
