# iMa Repository Understanding

## Executive Summary

iMa is an exploratory horse-racing machine-learning research workspace. Its apparent goal is to estimate horse finish probabilities, especially top-three outcomes, and use those probabilities to investigate handicap-racing wagering decisions.

The repository is not currently a runnable application or reproducible Python package. The working system is encoded as a loose sequence of Jupyter notebooks, CSV handoffs, and two committed scikit-learn model artifacts. The primary Hong Kong race dataset is now present under `track/`, but the notebooks still reference absolute paths from earlier macOS and Linux machines and no dependency environment is declared.

## Evidence-Based Workflow

```text
External JSON/CSV race records
        |
        v
dadaframer / records / new_features notebooks
        |
        v
Race and runner feature tables
        |
        v
TrackX_data_cleaning_preparation or Simplified Horse Dataset
        |
        +--> train/test feature CSVs
        +--> binary and multiclass label CSVs
        +--> race-group CSVs
        |
        v
Preprocessing and model experiment notebooks
        |
        +--> logistic regression, MLP, tree, boosting, SVM experiments
        +--> calibration, bagging, stacking, and AdaBoost experiments
        |
        v
logreg_poly.joblib / nn.joblib
        |
        v
Predictions notebook and betting-cost exploration
```

## Functional Areas

### Data ingestion and normalization

- `notebooks/dadaframer.ipynb` converts external JSON-style datasets into records, movements, veterinary, and trackwork CSVs.
- `notebooks/records.ipynb` and `notebooks/new_features.ipynb` derive historical and horse-level features from those tables.
- `notebooks/Spark Dadaset.ipynb` and `notebooks/spark_logistic_regression.ipynb` appear to be isolated Spark experiments rather than part of the main pipeline.

### Feature and label engineering

- `notebooks/TrackX_data_cleaning_preparation.ipynb` is the broad, 105-code-cell feature engineering and experimentation notebook.
- `notebooks/Simplified Horse Dataset.ipynb` is the clearest dataset assembly boundary. It reads race and runner data, creates train/test splits without shuffling, and exports feature, label, multiclass-label, and race-group CSVs.
- Labels include top-three binary variants and a multiclass finish-position formulation.

### Preprocessing

- `notebooks/Simple Horse Pipelines.ipynb` experiments with median imputation, sixth-degree polynomial features, numeric/categorical separation, and one-hot encoding.
- Numeric examples include horse age, weights, draw, odds, speed/history features, finishing time, and rating.
- Categorical examples include horse, race class, gear, country/type, jockey, and trainer identifiers.

### Model experimentation

- Binary classification notebooks compare logistic regression, multilayer perceptron, gradient boosting, random forest, SVM, decision tree, passive-aggressive, ridge, and stochastic-gradient models.
- Additional notebooks test calibration, stacking, bagging, and AdaBoost meta-estimators.
- The committed `logreg_poly.joblib` identifies itself as a scikit-learn `LogisticRegression` artifact created with scikit-learn 0.23.2.
- The committed `nn.joblib` identifies itself as a scikit-learn `MLPClassifier` artifact.
- Stored notebook outputs report binary-classification accuracies ranging roughly from 0.59 to 0.89, but these results are not independently reproducible from the checkout.
- Stored multiclass outputs are around 0.09 to 0.13 accuracy, indicating that exact finish-position prediction is much weaker than the binary task.

### Prediction and wagering exploration

- `notebooks/Predictions.ipynb` loads a manually prepared race CSV and the logistic-regression artifact, recreates polynomial preprocessing, and generates predictions.
- `notebooks/Cost of Betting.ipynb` calculates combinatorial double-trio and triple-trio ticket costs. It is analysis support, not an integrated bankroll or expected-value engine.

## Available Data Assets

### Primary model dataset

- `track/hkracing 2/races.csv` contains 6,349 Hong Kong races from 1997-06-02 through 2005-08-28, covering Sha Tin and Happy Valley.
- `track/hkracing 2/runs.csv` contains 79,447 runner records across 6,348 race IDs, with finish results, horse/jockey/trainer IDs, weights, draw, section positions and times, and win/place odds.
- These column names match the data contract used by the TrackX and Simplified Horse Dataset notebooks. This is the current canonical input candidate.
- One race has no corresponding runner records and should be identified during ingestion validation.

### Enrichment and alternate datasets

- `track/horse-racing-dataset-for-experts-hong-kong/` adds 14,212 race results, 3,940 horse profiles, and 648,016 trackwork observations. It may support training-condition, pedigree, trainer, and workout features, but uses a different schema and identity model.
- `track/koreahorseracedata/` contains annual Korean runner/horse data from 2005 through 2014. Most result-side `p` files are still iCloud placeholders, so this source is not locally complete.
- `track/hkrj_samples_xls 2/` contains HK racing sample spreadsheets for calendars, racecards, results, sectionals, dividends, horses, trials, and comments. These are useful schema references rather than a historical training corpus.
- `track/sya-horses-for-courses/` contains normalized market, horse, rider, weather, and condition tables, but forms, odds, and runners are currently iCloud-placeholder directories. It cannot yet support a complete relational pipeline.

These sources should not be merged directly. Each needs an adapter into a canonical race/runner/market schema, explicit provenance, and entity-resolution rules.

## Research Basis

- `research/1994-benter.pdf`, William Benter's *Computer Based Horse Race Handicapping and Wagering Systems: A Report*, supplies the central architecture idea: estimate fundamental horse probabilities, incorporate public/market information, evaluate expected value, and size wagers under bankroll constraints.
- `research/KellyCriterion2007.pdf`, *The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market*, motivates growth-optimal and fractional-Kelly bankroll sizing rather than choosing wagers from classification confidence alone.
- `research/Chance constrained optimization for parimutuel horse race betting.pdf` adds a portfolio-optimization layer that estimates outcomes and payouts under uncertainty and limits downside through chance constraints.
- `research/Track.pdf` is retained as a domain reference; its exact role should be documented once its title and intended use are confirmed.

The code currently implements mainly the first layer: feature construction and outcome classification. Market-probability combination, expected-value filtering, constrained wager allocation, and bankroll simulation are not yet implemented as one auditable pipeline.

## Live Race and Odds Integration

The project needs a separate live-data adapter before predictions can support real wagers. `Bobosky2005/hkjc-api` is a useful reference implementation and npm package for HKJC's GraphQL interface. It is MIT-licensed and exposes:

- Active meetings, venues, race numbers, post times, status, distance, going, class, course, and field size.
- Declared runners, reserves and scratches, horse code, barrier draw, handicap/current weight, rating, gear, jockey, trainer, and last-six form.
- WIN, PLACE, quinella, quinella-place, forecast, tierce, trio, first-four and multi-race pool odds.
- Pool status, investment, minimum ticket cost, last-update time, and odds movement fields.

The package is a thin TypeScript wrapper around `https://info.cld.hkjc.com/graphql/base/`; it is not an independent or guaranteed data service. A direct request from the current development machine returned `Internal server error - WHITELIST_ERROR`, both with a bare request and with normal HKJC browser-origin headers. Access must therefore be proven from the intended deployment location before this endpoint can be accepted as the production source.

The live boundary should be designed as:

```text
HKJC or alternate live provider
        |
        v
scrapper provider adapter
        |
        +--> immutable timestamped raw response
        +--> normalized meeting/race/runner/odds snapshot
        +--> validation and freshness status
        |
        v
feature assembler using only information available before post time
        |
        v
probability model + market probability comparison
        |
        v
wager optimizer and decision audit record
```

Provider responses should never feed the model directly. Every poll should be timestamped and normalized into stable internal records so the exact wager decision can be replayed later. Runner identity must map live HKJC horse codes to historical horse IDs, and scratches or material runner changes must invalidate and regenerate predictions. Odds snapshots need explicit race, pool type, combination, source timestamp, received timestamp, and sale status.

`scrapper/` now implements this boundary as an installable Python package. It includes an HKJC GraphQL adapter, per-runner horse-page collection, model-schema normalization, historical fallback enrichment, immutable raw/normalized/model snapshots, readiness reports, a command-line entrypoint, and recorded tests. Horse pages are live-verified for profile/form, trackwork, veterinary, and movement records. The GraphQL meeting/odds endpoint remains whitelist-blocked from the current network, so fixture-backed development works while production endpoint access still requires verification.

The target modular-monolith design and migration phases are documented in `docs/architecture/TARGET_ARCHITECTURE.md`.

## Current Architecture

The architecture is best described as an exploratory notebook pipeline with file-based stage handoffs. It is neither a modular monolith nor a service architecture yet because there is no stable executable boundary, package API, deployment unit, or automated workflow.

The natural future bounded contexts are:

1. Race data ingestion and normalization.
2. Historical feature and label construction.
3. Training and model evaluation.
4. Probability calibration and race-level prediction.
5. Betting strategy and backtesting.

These should remain modules in one repository unless real scaling or ownership constraints emerge. Microservices would add operational complexity without evidence of a need.

## Principal Risks

1. **Reproducibility is still broken despite the primary data being present.** The README under `notebooks/` refers to a missing `requirements.txt`; notebooks reference hard-coded paths such as `/Users/mk2/...`, `/home/goblin1732/...`, and `s3://dadadata/...`; and there is no canonical command or environment definition.
2. **Preprocessing is not bound to the models.** The prediction notebook reconstructs transformations separately, while the serialized artifacts appear to contain estimators rather than complete fitted pipelines. Training-serving skew is therefore likely.
3. **Evaluation methodology needs validation.** Several notebooks fit on one array and then call `cross_val_predict` on the held-out test array. Some pass race groups to ordinary integer `cv` splitters, which does not guarantee group-aware splitting. This can produce misleading metrics or leakage between runners in the same race.
4. **The domain objective and metrics are misaligned.** The repository aims at probability estimation and wagering, but much of the recorded evaluation emphasizes classification accuracy. Calibration, log loss, Brier score, race-level ranking, market baseline comparison, and simulated return are more relevant.
5. **Notebook duplication obscures provenance.** There are parallel TrackX notebooks, simplified variants, meta-estimator notebooks, and machine-specific copies with no declared canonical run order.
6. **Serialized model compatibility is fragile.** At least one artifact was produced with scikit-learn 0.23.2. Pickle/joblib artifacts are version-sensitive and should not be loaded from untrusted sources.
7. **Generated Megatect scorecard undercounts risk.** Its static scanner detects no Python module graph inside notebooks, so the generated `low risk` score means `no conventional source graph detected`, not that the research pipeline is low risk.

## Recommended Architectural Direction

Keep this as a single Python project and extract stable code from notebooks into a small package. The first useful boundaries are `data`, `features`, `training`, `prediction`, and `backtesting`, with notebooks retained as thin experiment drivers.

The highest-value first milestone is reproducibility:

- Declare a supported Python version and locked dependencies.
- Define configurable data locations and a small sample dataset or schema fixture.
- Establish one canonical end-to-end experiment command.
- Save preprocessing and estimator together as one fitted pipeline.
- Split by race and time before fitting transformations.
- Produce a versioned evaluation report centered on probability quality and wagering utility.

Only after that baseline exists would deeper modularization be justified.

## Evidence Limitations

This assessment reads notebook source, metadata, stored outputs, dataset schemas and counts, PDF metadata, and artifact string metadata. It does not execute the notebooks because their paths are not wired to the newly added datasets and no dependency environment is declared. The model files were not deserialized because joblib/pickle loading can execute code and the artifacts are version-sensitive. Several supplementary datasets are represented by iCloud placeholders and were assessed as unavailable rather than empty.
