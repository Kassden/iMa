# iMa Target Architecture

## Decision

Build iMa as a modular Python monolith with offline training and online race-decision workflows sharing stable domain schemas. Do not split it into microservices until independent scaling, ownership, or deployment pressure exists.

## System Shape

```text
                         OFFLINE
Historical providers --> ingestion adapters --> canonical race store
                                               |
Research notebooks --> feature definitions ----+
                                               v
                                      training + evaluation
                                               |
                                               v
                                   versioned fitted pipeline

                         LIVE
Meeting/odds provider --> race snapshot --------+
Horse pages -----------> form/profile ----------+--> feature assembler
Trackwork/vet/movement -> enrichment snapshots -+          |
                                                          v
                                                probability inference
                                                          |
Market odds ----------------------------------------------+
                                                          v
                                            value + wager optimizer
                                                          |
                                                          v
                                                decision audit record
```

## Modules

### `providers`

External integrations only. HKJC GraphQL, HKJC horse pages, recorded fixtures, and future commercial feeds implement provider interfaces. Provider payloads never enter model code directly.

### `domain`

Stable records for meetings, races, runners, odds, horse profiles, form runs, trackwork, veterinary events, movements, model features, predictions, and wager decisions.

### `ingestion`

Validation, identity resolution, timestamp normalization, raw snapshot persistence, and conversion from each provider schema into domain records.

### `features`

One versioned implementation of pre-race features used by both training and live inference. Every feature declares its source and availability timestamp to prevent post-race leakage.

### `training`

Race/time-aware splits, preprocessing, estimator fitting, calibration, evaluation, and artifact publication. Metrics should include log loss, Brier score, calibration, race-level ranking, market baseline improvement, and simulated return.

### `inference`

Loads a versioned fitted pipeline, validates its feature schema, predicts probabilities, and records model/data versions. It does not scrape or allocate wagers.

### `strategy`

Combines fundamental probabilities with market probabilities, calculates expected value, applies liquidity and freshness constraints, and performs fractional-Kelly or chance-constrained allocation.

### `operations`

Polling schedule, retries, freshness alarms, scratch handling, race lockout, observability, and immutable decision audit storage.

## Data Contracts

Every live race decision must retain:

1. Raw provider responses and page HTML hashes.
2. Normalized meeting, runner, odds, and horse-history snapshots.
3. Feature vector with schema version and per-feature observation time.
4. Model artifact version and probability output.
5. Market snapshot, strategy settings, proposed wager, and rejection reason if no bet is made.
6. Later race result and settlement for backtesting and monitoring.

## Safety Boundaries

- Never use race results, final odds, or post-time updates as pre-race features.
- Recompute after scratches, jockey changes, weight changes, or stale odds.
- Refuse inference when the fitted pipeline schema and live feature schema differ.
- Refuse wager generation when odds are stale, pools are closed, or required horse pages failed.
- Keep collection, prediction, and wager execution separate. Automated bet placement requires a later explicit decision and additional controls.

## Migration Plan

### Phase 1: Reproducible baseline

- Installable environment and canonical data paths.
- Live snapshot collector with fixtures and readiness reports.
- Extract the simplified notebook feature logic into tested Python functions.

### Phase 2: Reliable model

- Rebuild the historical dataset using the same feature code as live inference.
- Use chronological, race-grouped train/validation/test splits.
- Train and calibrate complete scikit-learn pipelines.
- Publish versioned artifacts and evaluation reports.

### Phase 3: Market and strategy

- Store odds time series and convert pools to market probabilities.
- Implement Benter-style fundamental/market probability combination.
- Add expected-value filtering, fractional Kelly, and chance constraints.
- Backtest with realistic timestamp, liquidity, dividend, and transaction assumptions.

### Phase 4: Operations

- Schedule race-day collection and retries.
- Add freshness, schema-drift, and provider-health monitoring.
- Add a review interface for predictions and proposed wagers.
- Consider automated execution only after sustained shadow-mode validation.
