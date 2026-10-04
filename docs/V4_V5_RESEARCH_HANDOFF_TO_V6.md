# V4/V5 Research Handoff to V6

Read-only server audit, 2026-10-04, legacy observations through 13:06:55 UTC; V6 checkpoint refreshed at 13:11:19 UTC.
Recipient: Luna V6 High, orchestrator `01a106f3-d5e3-71e2-bba4-37773eb3149f`, via parent relay.
This is research evidence and proposed matched tests, not permission to promote models, resume V5, or place wagers.

## Executive Findings

- V4 produced genuine matched-development improvements across five target/model contracts. Its persisted boosted win best is **2.159721150911**, superseding the earlier reported 2.163605 checkpoint.
- V5's observed boosted best is **2.156091109576**. V4/V5 champion comparisons use the same verified dataset bytes, scored runner keys, target labels, and per-target protocol artifacts. The improvement is not explained by a different scored population, but recipes, parameters, runtime revision, and discovery all changed: it is not an isolated causal discovery gain.
- The most reusable signal is **past speed/form expressed relative to the current race**, plus historical support and variability. Horse-only discovery worked for boosted win/odds; jockey histories appeared prominently in linear win, ranking, and placing selections.
- None of the five V5 champion paired reports establishes an independent confirmation win. E1/E2/E3/E4 are `inconclusive`; B is `reject` against the stronger boosted win comparator, not against its own earlier linear baseline.
- Fundamental probability improved, but **market superiority is not established**. Best V5 boosted fundamental log loss is 2.156091 versus calibrated recorded-market 2.006462. Its selected blend is 2.006900, slightly worse than calibrated market, with zero fundamental blend weight in two of three folds. Do not sell this as betting alpha.
- Current V6 Luna's best 2.176621 is **worse than matched C1 control 2.176278 by 0.000343**. No overall matched gain and no additional Luna budget are established by this handoff. Transfer the lessons before considering further research choices.
- V5 is operator-cancelled, not cleanly completed. Preserve the 469 completed records and the eight interrupted records separately. STOP remains in place; no automatic resume.

## Evidence and Comparability

Server campaign root prefix: `/home/imaopt/research-v2/campaigns/`.
Main sources: `agentic_v4_features/` and `agentic_v5_discovery/`, each with `ledger.sqlite`, `trials.jsonl`, `campaign.json`, `campaign-identity.json`, `decisions/`, `evidence/`, `search/programs.jsonl`, and per-attempt artifacts.
SQLite reads used `mode=ro`; MLflow queries were retrieval only.

| Item | V4 Main | V5 Main |
| --- | --- | --- |
| Campaign | `agentic_v4_features` | `agentic_v5_discovery` |
| Recorded completed | 1005 | 469 |
| Recorded failures | 0 in trial results | 2 |
| Model contracts sampled | B 804; E1 51; E2/E3/E4 50 each | B 375; E1/E2 24 each; E3/E4 23 each |
| MLflow experiment | 5 | 6 |
| Runtime revision | `ima-v4-feature-20260930-d` | `0915b619f6a40dcc390c3044107831293f826660` |
| Dataset SHA256 | `8f33d989cb258ecb45f16b49363e5a1b791b95a8ca9a40b8e22db7a8a64feb6c` | Same, independently recomputed |
| Configured fold contract | 17000 train / 1000 calibration / 1000 score, maximum 3 | Same |
| Champion effective train races | 17000 / 18000 / 19000 | Same |

Dataset paths are `agentic_v4_features/inputs/rich-history-v4.csv.gz` and `agentic_v5_discovery/inputs/rich-history-v5.csv.gz`. Identical hashes do not prove every legacy source's historical publication time; do not transfer legacy availability assumptions into V6.

The audit recomputed sorted `(fold_id, race_id, horse_no)` keys and keys plus labels from persisted predictions for each main campaign's first completed and champion result in every contract. All first/champion and V4/V5 champion comparisons matched **within target contract**. This is stronger than comparing only a configured protocol ID.

| Population | Rows / Scored Races | Keys SHA256 | Keys + Target SHA256 |
| --- | --- | --- | --- |
| B/E1 win | 35989 / 3000 | `6024e2f6b0f82cd4a6b62c54bbd7194c5d2b9eb033397363d1db3348d4a8f3f9` | `0349d275391face312cc7e462be2491da6f2291da4cd8ccb5052efcfb6055825` |
| E2 ranking | 36041 / 3000 | `ea70ef9c69cfe7ef177efb61d017e3a5d8a48ae8412a0f8e0d955fc4d00496fa` | `445a6928f4d52159f47257074047e570d4f403faabab9ce4d30ff6cc132b7519` |
| E3 placing top 3 | 36041 / 3000 | Same E2 keys | `07812e4cb577f95bb05e2b38a8ee2d6747ce9701f4274740280cc7c0dedac84e` |
| E4 recorded final odds | 36041 / 3000 | Same E2 keys | `ab8c513393c4e3dc7c3eadfd627be918e5629ade0b3b5994b19442072da3417f` |

Hash method: pandas `to_json(orient="records")` after stable sort, string identity columns; target column `target_win` for win, `label` otherwise. These are audit hashes, not invented legacy native lineage fields.
Win excludes non-single-winner races; auxiliary targets have different eligibility, so do not compare their absolute objective numbers or populations as one global champion.

Persisted trial protocol SHA256s, identical between V4/V5 within contract:

- B/E1: `304efaefc5326166a97e28aa377752309ec1e0a84d5d73ba26c45e8c2a7b366a`, protocol `2218c98ed69d1e11`.
- E2: `cb2ea34ed04d1703ad2e4d17603913003f0d3dda8f6d997185b3c72beeca6c60`, protocol `98b0f0a03509a96e`.
- E3: `ad79f170d8671b772d17e521743ee92b86d7892667a4eb43246decb400038d5f`, protocol `7ecb03e74787953d`.
- E4: `d0958277164805eb141a79cd6517e5045f74f596569ac1a0a13943841295db2f`, protocol `9ddafcdb2869754d`.

Important confound: first V5 E1 used only 2404/2440/2476 effective training races; first V5 E4 used 2421/2443/2476. They score the same races, but are not full-history training controls. V5's first full-history E1 was 2.295168, not a useful strong reference; first full-history E4 was 0.435275. Use V4 champions and explicit fresh ablations instead of crediting the entire first-to-best delta to generated features.
Small canaries also differ: `agentic_v5_canary_final` has dataset SHA `96a2341da17e7c2c2b7889a59e9aca59a56f71cef1323ecf7f233db1601edcd0`, protocol identity hash `0cad1defc390a676464d2da1dca948d47ffbc2190c2b95a99098af80367c7fb2`. Its B best 2.222758 is not a regression relative to main-data B 2.171223.
The old-data V4 control has dataset SHA `ab437757562c5bf914bae1741a8c80d55b3c647a51f99bc08d40495e49221160`; its 2.188790 cannot isolate a transform gain against rich-history results.

## Progress by Contract

V4 baseline means the first completed main-campaign trial in that contract, not an untouched universal default. NDCG is shown positively, reversing the stored minimized negative objective. Every other metric is minimized.

| Contract / Model | V4 First | V4 Best | V5 Observed Best | V5 Improvement over V4 Best | Best Trial Seconds V4 / V5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| B / conditional logit; fundamental log loss | 2.187273 | 2.181688 | 2.171223 | 0.010465 lower | 259 / 7155 |
| E1 / boosted; fundamental log loss | 2.192099 | 2.159721 | 2.156091 | 0.003630 lower | 1279 / 9114 |
| E2 / LambdaRank; race NDCG@3 | 0.752124 | 0.756255 | 0.757127 | 0.000872 higher | 579 / 3794 |
| E3 / CatBoost classifier; top-3 race Brier | 0.169245 | 0.167934 | 0.166593 | 0.001341 lower | 2282 / 6809 |
| E4 / CatBoost regressor; log-final-odds MAE | 0.523041 | 0.422614 | 0.416198 | 0.006416 lower | 2681 / 3664 |

V5 B first full-history result was 2.175235; best reduces that by 0.004012. V5 E2 first was NDCG 0.752706; E3 first was Brier 0.166949. These within-V5 deltas also combine hyperparameters and feature settings, not pure feature ablations.
All ten champions below were independently retrieved from MLflow: actual `FINISHED` runs and matching `READY` model versions, not just local upload receipts. Registered versions are candidates, not production promotions.

| Champion | Attempt ID | MLflow Run ID | Registered Version |
| --- | --- | --- | --- |
| V4 B | `attempt-a9b83a1370b4ce7b38318a5ce434e3868fc23487007656283c1c4b20c71660be` | `57424a31d2874874bc04b7172c697ba3` | win-probability / 245 |
| V4 E1 | `attempt-705531752709411d2774cdce73c111f826c213d04f3d47e1ce7822f582050e0d` | `56e3796120184358b7420ba28d59d984` | win-probability / 367 |
| V4 E2 | `attempt-0578a22a3f719cab04104728595859d34b9d1e63bbe4b5c70e1a944daed138b9` | `1f11d74f106745f99853d7af291ccca2` | ranking-strength / 23 |
| V4 E3 | `attempt-3c71eaeea20356b8a3f31df73e95ba13ac016eeca71c1a6a989e4775f47e3ec9` | `d9c1a20cb36f4ae5912c2afbbbad10e5` | placing-top-k / 39 |
| V4 E4 | `attempt-04bf348ade4608657905260f21e46d8539e3a8554677ae7187b8b9f4262b44a4` | `7523c18b88e844d09e8e5d99d1938d52` | recorded-final-win-odds / 47 |
| V5 B | `attempt-7848da807ceaeb4b4c0e7b4d57e888e7d254b7ebb143d2776d8c2dd6d80dafda` | `eef73fbb11f84b0ca96edaafd0af690c` | win-probability / 414 |
| V5 E1 | `attempt-e36cb7c0fb4eab8c98860d45a51b695050aa6bbf11a502fbcc1ec10aee0e64dc` | `4849319248004aa1b994ca7f58774cfb` | win-probability / 286 |
| V5 E2 | `attempt-ddead4e8047182e1880d18d5ad83c9c7dda85998399413eb2babb18bb83a1298` | `2f405abacc25407d915fbf22754dbce4` | ranking-strength / 21 |
| V5 E3 | `attempt-b62e1d55544dc239cd2d403e1a7238d4591f8e692f843aa503597a48f21bcf68` | `5aa9e1fd99a54878a6d3ac896bc36dea` | placing-top-k / 16 |
| V5 E4 | `attempt-3da8dedf5990bf00655090ea94c2691fb3f0ea0150d7638c09b4f57d59cbcb2b` | `d9f945844e404fa0aa6a1c3c681d3e3e` | recorded-final-win-odds / 18 |

Registered name prefixes: V4 `ima-agentic-v4-feature-candidates-`; V5 `ima-agentic-v5-discovery-candidates-`, followed by the target suffix above. Backend: `http://100.95.24.121:5000`, experiments 5/6.
Exact recipes reside in the attempt's `ledger.sqlite.payload_json` and `trials/<attempt>/package/recipe.json`; predictions, protocol, feature/discovery reports are in `trials/<attempt>/`.

## What the Successful Recipes Actually Changed

All champions use seed 42 and `all_history`. Win recipes retain temperature calibration and `market_softmax` blend; their minimized objective is fundamental model probability, not selected blend. E2/E3/E4 use no blend/calibration in their declared recipes.
Abbreviations below: S=`last_speed_ratio`, W=`prior_win_rate`, HJ=`horse_jockey_win_rate`; rank/center means within-race transformation of past-only inputs. Original inputs are retained according to the persisted feature programs.

### B: Conditional Logit

V4 first: `notebook-rich-v2`, center(S,W), l2=0.02399969. V4 best: signed_log1p(S,W,HJ), rank(S,W), center(S,W), interaction(S,W); l2=0.15607650, max_iter=3824. Joint gain 0.005585 log loss; no one-axis causal estimate.
V5 best: same rich schema, rank(S,W,HJ), center(S,W), interaction(S,W), signed_log1p(`avg_speed_ratio_3`,`prior_top3_rate`), plus rank and center of `avg_speed_ratio_3`, `avg_speed_ratio_5`, `avg_lengths_behind_3`; l2=0.19940633, max_iter=3345.
Discovery: horse+jockey; speed_mps/beaten_lengths/carried_weight; count/mean/std/min/max; depth 2; maximum 500 definitions, 32 selected. Actual catalog 321, fold-001 selection 32. Strong selected examples: jockey last-three speed support, race-centered jockey last-three mean speed, centered jockey last-five speed variability.
Lesson: magnitude and ordinal context both look useful. The last V4 center extension improved an already tuned 2.181700760 recipe by only about 0.000013; do not confuse tiny adaptive refinements with robust new signal.

### E1: Boosted Win Probability

V4 first: `benter-rich-v1`, no transforms, max_iter=160, l2_regularization=2.07378749. V4 best switches to `notebook-rich-v2`, rank(S,W), interaction(S,W), with learning_rate=0.02220407, max_iter=496, max_depth=7, max_leaf_nodes=45, min_samples_leaf=117, l2_regularization=2.25561970.
V5 best retains that schema/transform set, uses learning_rate=0.00996064, max_iter=1314, max_depth=11, max_leaf_nodes=58, min_samples_leaf=224, l2_regularization=39.82874730.
Discovery: horse only; speed_mps/beaten_lengths; count/mean/std; depth 2; 500-definition ceiling, 24 selected. Actual catalog 99. Fold-001 selected: centered fifth-lag speed (`dfs_38a56201a7547f88cc5fc755`), last-five usable-speed count (`dfs_2d3bcfd2165ca69cafc6c685`), centered 90-day mean speed (`dfs_1459d950ace5b08b74f87b75`), raw 90-day mean (`dfs_3e8e84bb3a491f79b63d9168`), last-three speed std (`dfs_dc4577d8313ea35aaf45e284`). Training coverage respectively about 94.8%, 94.8%, 87.9%, 87.9%, 85.5%.
The best's top-pick win rate is 25.53%, winner-top3 rate 54.4%; calibrated market is 30.8% / 61.3%. This is useful fundamental research, not demonstrated market replacement.

### E2: LambdaRank

V4 best: `benter-rich-v1`, rank(S,W,HJ), center(S,W), interaction(S,W); learning_rate=0.07342532, n_estimators=329, num_leaves=8, min_child_samples=30.
V5 best adds signed_log1p(`avg_speed_ratio_3`,`avg_lengths_behind_3`); learning_rate=0.01626989, n_estimators=1146, num_leaves=10, min_child_samples=106. Horse+jockey discovery; count/mean/std; speed/beaten lengths; depth 2, 24 selected from 177 candidates.
Leading selections include relative jockey last-five beaten-length support and relative jockey last-three/last-five speed support. Counts may reflect opportunity or missingness as well as skill: ablate support indicators separately from speed values.
V4 best's calibrated ranker-to-win log loss was 2.184496 versus calibrated market 2.009062. Good ranking NDCG is not automatically calibrated win probability.

### E3: CatBoost Top-3 Classifier

V4 best: `benter-rich-v1`, signed_log1p(S,W), rank(S,W), center(S,W), interaction(S,W); depth=4, iterations=772, learning_rate=0.06954736, l2_leaf_reg=8.20777743.
V5 best: `notebook-rich-v2`, same transforms; depth=5, iterations=1440, learning_rate=0.03104415, l2_leaf_reg=29.79650299. Horse+jockey discovery, count/mean/std, speed/beaten lengths, depth 1, 16 selected from 177 candidates.
Leading selections are centered jockey last-five/last-three mean speed and centered jockey fifth-lag speed. V5 Brier 0.166593 beats its constant baseline 0.188400 and shuffled control 0.189037, but the latest-champion increment is tiny and inconclusive.

### E4: CatBoost Recorded-Final-Odds Regressor

V4 first: `benter-rich-v1`, depth=7, iterations=120, no transforms. V4 best: `notebook-rich-v2`, rank/center(S,W,HJ), interaction(S,W); depth=7, iterations=723, learning_rate=0.15807038, l2_leaf_reg=12.48054615.
V5 best retains schema/transforms; depth=6, iterations=732, learning_rate=0.12099815, l2_leaf_reg=2.61505745. Horse-only discovery, count/mean/std, speed/beaten lengths, depth 1, 16 selected from 99 candidates. Leading signals include centered fifth-lag speed, 90-day mean speed, and centered third-lag beaten lengths.
This predicts **recorded final odds**, not a live pre-off quote. Do not use final odds as current-race input or interpret log-odds MAE as win loss, ROI, or an executable value bet.

### Shared Discovery and Speed Caveats

V5 champion specs share 90/365-day windows, 3/5 prior-start sequences, MI screening, correlation threshold 0.95, missingness limit 0.95, date-only-next-day availability, race-relative construction, recency decay 180 days, and adjusted-speed residuals enabled with shrinkage 5.
Persisted residual reports describe forward-block OOF Ridge(alpha=10) adjusted on distance, carried weight, class, horse age, and draw. They establish a fitted residual construction, not its causal benefit. No top-three catalog selections inspected were residual features; the audited fold-001 selected catalog entries did not identify a residual/adjusted feature by those names. Do not claim that enabling the flag caused the champion gain.
There is no separate speed-target champion in the inspected V4/V5 objectives: speed was a predictor family. V6 must enforce source-specific units and individual timing, never substitute batch/winner trial time for an individual's observation.

## Statistical and Coverage Lessons

Exact V5 champion evidence files: `trials/<V5 champion attempt>/paired-feature-report.json`. All compare 3000 scored races after adaptive development selection, not untouched final confirmation.

| Candidate | Comparator | Paired Objective Delta | Race Bootstrap 95% Interval | Verdict |
| --- | --- | ---: | --- | --- |
| B | E1 champion `attempt-e36cb7c0...` | +0.015132 | [+0.001908, +0.027669] | reject versus boosted, not versus B baseline |
| E1 | `attempt-cb88a39fbe2283d951cc8649e1106207e70943b1f137ae11db98b3c5848ff690` | -0.001796 | [-0.008455, +0.004458] | inconclusive |
| E2 | `attempt-4b4151e292738370daf39aff691fa0d5121637b6c8a6e9b224fee62a6fa584d9` | -0.000401 | [-0.002815, +0.001885] | inconclusive |
| E3 | `attempt-3693834bd0b0a00fdcb4bbd0b3c5d5240ce0f9b47690e395a0215f18a3bb5391` | -0.0000355 | [-0.000313, +0.000238] | inconclusive |
| E4 | `attempt-0bf9eefb862890bf4c64b7832f44b526890fd689eee894748508f64d0311ad1e` | -0.000441 | [-0.001591, +0.000828] | inconclusive |

Meeting-equal-weight sensitivity intervals likewise cross zero for E1/E2/E3/E4. The E1 comparator also changed discovery depth/selected cap and boosted parameters, so even this paired comparison is not a pure discovery ablation.
Fold-001 synchronized within-race generated-family permutations worsened objective in both recorded repeats: E1 +0.036434/+0.030148 log loss; B +0.037984/+0.036853; E2 +0.006455/+0.009684 negative NDCG; E3 +0.000943/+0.000969 Brier; E4 +0.041528/+0.039813 log-odds MAE. This supports model reliance on the generated family, not independent statistical confirmation or individual-feature causality.
MI reports explicitly say training labels were used for screening; inner proxy diagnostics are not independent confirmation. Duplicate/correlated generated columns were rejected by Feature-engine screening. More definitions is not itself more information.

Known main-V5 failures:

- `attempt-db54e9a727ba3287f23de073cd9bcd538c750fe2dee165ffca7b92dd2e9275cc`.
- `attempt-8ef5c502c269e55296f17b1d893fcaca0e9a73f358ab980ecd978f486df5b72a`.

Both: `ValueError: Explicit discovery transforms exceed the selected-feature ceiling`. Budget explicit transform dependencies inside the selected cap before launching expensive preparation.
Legacy effective-training reports repeatedly lack days-since-trackwork/trial, individual trial speed/placing, sectional time/position, season stakes, start-season rating, starts-past-10-meetings, and veterinary/movement recencies. A named feature or an availability flag is not evidence that the underlying historical values existed.
V6 strict official manifest records 104723 observations, all prospective-capture tier, but **zero usable historical runner-event views and zero verified coverage records**. Missing historical event values are expected, not a pipeline failure and not verified zero activity. Keep retrospective occurrence hypotheses on separately labelled datasets/champions; never fabricate publication time.

## Cost, Time, and Operational State

V4 recorded trial-duration sum: 417851.11 seconds, about 116.1 job-hours. V5: 2080893.43 seconds, about 578.0 job-hours at the stable 469-completed/2-failed snapshot. These are overlapping elapsed trial durations, not CPU-hours, wall campaign duration, or a hardware bill.
V4 decisions: 47 files, 36 with numeric `planner_usage.total_cost_usd`; known sum $0.75003574359, not proof of full billing coverage. V5: 124 decisions, 109 with numeric cost; known sum $2.3557712428 agrees with campaign reported spend. Fifteen files lack numeric cost; absence is not a paid-call bill of zero. No compute-dollar estimate is available.
V5 best manifests report feature-generation wall seconds around 6036 B, 3874 E1, 4191 E2, 4166 E3, 3877 E4. Some exceed the associated trial-duration field: differing stage/cache boundaries make them non-additive. Treat large preparation cost as a reason to reuse exact-workload caches, not proof that a longer search is worthwhile.

Parent preserved `/home/imaopt/research-v2/ops-parent/v5-stop-20261004/`, including read-only audit sources `ledger.sqlite`, status/config/identity, STOP and `stop-receipt.json`.
Ledger backup SHA256: `e9ea47f520bc4db03a03bfce675a0fa465e652a2f59ab00015f491218b5ec93d`.
Graceful STOP was requested around 13:02 UTC. Parent cancelled only the V5 unit at 13:06:10 UTC after eight long jobs remained draining. At 13:06:55, unit is `inactive/dead`, MainPID 0, ExecMainStatus 15, systemd Result success; this is **operator cancellation**, not five/eight new successful fits. Last status remains `draining` at 13:06:08 and SQLite still has 469 completed, 2 failed, 8 running records representing interrupted work. Do not relabel those eight as completed or silently resume them. STOP remains present. This audit made no service/campaign/backend changes.

## Concrete Matched V6 Tests for Luna

Current observed V6 campaign: `agentic_v6_luna_session_20261004`; revision `376964fab8ba18f4e493824a3fa7702f9e4848e2`; at 13:11:19, 5 completed, zero pending tells, **one pending tracking upload**. This is not yet a fully drained backend acceptance checkpoint.
Dataset is `dataset-19aaa959e948d6e145ce62c8160413087b46ba3c2578e0de15c452054d9a83db`, SHA `2321218048ffb1f2b9d9ea9a357ec8a753775420f9808385dc323741ad1795a1`. It has 171782 rows/13924 races; official controls used a fresh 9000/500/500, latest-three-fold contract, actually scoring 18075 rows/1502 races. This is not the legacy 3000-race population. Refit hypotheses; never import legacy objective thresholds or champion scores as matched V6 evidence.

Persisted Luna results: best B `attempt-f69d14f6f03715a45a747b58dce0e9ef3b864bb219e4a744fd41e830c6cb5237` = 2.176621019625; B ablation `attempt-4758bdcbffc8cb579d7bbe551e94914249be7324962342eaeaa297b33ab3a065` = 2.179126283755; boosted `attempt-6eeedf920b47dcb6a380374aba4c34ba5fc3b347c43e7ce2ebb9c8b500b6c6b0` = 2.181631825911; other B results 2.176952905336 and 2.183282744248. All five declare the same scored population hash `5b0a1fbf4848b77192c1bf08336f6ce678a1446ffe6d13af432b9c6ff729bd76` and protocol file SHA `0de67fa0864a07d739f567327001e41d619faa32c23efe620c88b87b4583f2b1` as C1. C1 best B `attempt-ba90d53a335338040332ee584171dd2d247073ac420cbe69e8a44c4663d8a024` = 2.176278241358. Luna boosted improves on C1's boosted 2.188141543854 but not on C1's best linear model. These are observed development results, not parameter-held causal estimates. The following is a proposed future test menu, **not an approved additional budget**.

1. **Re-establish reference recipes, no discovery.** On the same strict immutable dataset and target-specific protocol, run V4 B transform bundle and V4 E1 rank+interaction bundle using V6-eligible columns only. Freeze parameters, seeds, scored keys and exclusions. Missing legacy horse-age/rating/profile columns must be declared, not reconstructed from current profiles. Compare to V6's current seed on exactly the same evaluated population.
2. **Isolate past-speed effects before expanding search.** Fixed boosted recipe: base; add usable past-race speed mean/support; add variability; add within-race rank/center; then add speed x prior-form interaction. Keep support/coverage-only controls separate. Use manifest-eligible numeric predictors and explicit units. These are candidate hypotheses, not guaranteed incremental gains.
3. **Small matched discovery ablation.** Fixed V4 boosted parameters: discovery off versus horse-only count/mean/std, 3/5 starts and 90/365 days, 16 selected; then 24 selected. Change depth only after a matched benefit. Do not change model depth/learning rate simultaneously. Account for preparation, selected-feature dependencies, actual memory bounds, and fresh admission telemetry.
4. **Test jockey histories on linear/ranking/placing, not everywhere automatically.** Horse-only versus horse+jockey with the same selected cap and model parameters. Split support count, mean speed and variability families; legacy selections suggest all three but do not isolate which carries skill versus opportunity/missingness.
5. **Adjusted-speed residuals off/on.** Hold generated definitions and final model fixed; use chronological OOF auxiliary fitting, explicit coverage, and available covariates only. The legacy flag is a construction precedent, not evidence of residual efficacy. No current result/time/market feature may enter predictors.
6. **Keep ranking, top-k, odds and win objectives distinct.** LambdaRank tests optimize positive NDCG@3 via the stored negative objective; top-3 CatBoost tests use race Brier; final-odds models use log MAE. Evaluate any ranker-to-win calibration separately. Odds outputs can support research scenarios but not live quote or wagering claims.
7. **Separate strict and retrospective event studies.** Strict past-event coverage is presently zero. Do not spend a canary retuning all-null event families or convert unknown into zero. Explicit retrospective occurrence-tier event experiments need a separate comparison contract, population identity and champion bucket. Prospective capture should accumulate before publication-safe historical event claims become possible.
8. **Require incremental evidence, not only a new minimum.** Save paired race-level deltas and meeting-weight sensitivity; use shuffled-family diagnostics as exploratory checks. Record all adaptive tries. Keep final confirmation inaccessible to planning; leave it untouched until a predeclared bounded confirmation decision. A paid-planner canary is operational evidence, not scientific confirmation.

## Retrieval Map and Relay Boundary

- Reconstruct each champion from main-campaign ledger payload plus the exact attempt table above; do not use an old globally labelled `best` across mixed objectives.
- V5 `reference-champions.json` carries five V4 references. `evidence/cycle-0123.json` and subsequent `cycle-0124.json` expose objective/protocol/model-specific family champions and recent feature evidence; `decisions/cycle-*.json` retain hypotheses, budgets, usage and rejections.
- Per champion, inspect `discovery-manifest.json`, `discovery-selection-fold-*.json`, `discovery-diagnostics-fold-*.json`, `discovery-residual-fold-*.json`, `paired-feature-report.json`, `predictions.csv`, `protocol.json`, and `package/recipe.json` where present. Do not unpickle model files merely to inspect them.
- `agentic_v5_discovery/readback.json` is an early 14-result replay receipt, not a final 469-result audit. Its model replay success cannot stand in for full-campaign backend coverage.
- Audit coverage: all persisted main trial summaries ranked by contract; detailed first/champion artifacts and ten champion backend run/version records checked. Not every historical backend trace, model package, coefficient, or full source-publication lineage was audited. No pure parameter-held feature-refit ablation or final confirmation was performed here.
- Parent should relay this Markdown to the identified Luna orchestrator. This task did not send a cross-thread message or inject a document into the live planner, and did not authorize any V5 resume.
