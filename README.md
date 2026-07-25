# iMa

iMa is a horse-racing probability research project based on Hong Kong race data and the modeling/wagering ideas described in the papers under `research/`.

The repository now has four explicit areas:

- `track/`: historical race datasets and schema references.
- `notebooks/`: original data preparation and model experiments.
- `scrapper/`: live HKJC meetings, runners, odds, horse form, trackwork, veterinary, and movement collection.
- `ima/`: point-in-time datasets, race models, pool probabilities, market blending, strategy, ledgers, and model registry.
- `docs/architecture/`: current and target architecture evidence.

## Environment

Use Python 3.11-3.13. The existing machine has Python 3.11 available at `python3.11`.

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
```

Start Jupyter:

```sh
.venv/bin/jupyter lab notebooks
```

Run scraper tests:

```sh
.venv/bin/python -m unittest discover -s scrapper/tests -v
```

Run the complete test suite and historical benchmark:

```sh
.venv/bin/python -m unittest discover -v
.venv/bin/python -m scripts.train --output artifacts/models/latest
.venv/bin/python -m scripts.train \
  --canonical-runners data/processed/historical/runners.csv.gz \
  --output artifacts/models/canonical-2005-2025
```

Run the complete 1997-2025 parameter grid and generate the interactive results dashboard:

```sh
.venv/bin/python -m scripts.run_experiments \
  --output artifacts/experiments/full-history
```

Open `artifacts/experiments/full-history/dashboard.html` to filter model families,
rank parameter runs, inspect calibration metrics, and download the complete CSV or JSON results.

Run a fixture-backed scrape:

```sh
.venv/bin/ima-scrape \
  --race 1 \
  --fixture scrapper/tests/fixtures/race_snapshot.json \
  --output /tmp/ima-snapshots
```

## Live Collection

The default live workflow renders the public HKJC odds application in Chrome because direct GraphQL requests are rejected with `WHITELIST_ERROR`. It collects real WIN/PLACE prices and pool turnover metadata, then follows local Hong Kong runners with stable HKJC horse identifiers to collect:

- Horse profile and recent form records.
- Trackwork records.
- Veterinary records.
- Movement between Hong Kong and Conghua.

Each poll preserves raw and normalized snapshots and emits the exact feature columns used by the legacy all-attributes notebook model. Rows are marked `prediction_ready=false` if required data is missing or the runner is inactive.

The current Hong Kong mobile egress is accepted by the public browser application. Overseas simulcasts do not expose local HK horse-profile identifiers, so profile enrichment is correctly marked unavailable for those runners. Pass `--all-pools` to interact with every available pool tab and preserve its rendered odds table. Tierce, Trio, First 4, Quartet, Forecast, and Double combinations are also normalized where the page exposes stable matrix or top-combination structures.

Run a live check:

```sh
.venv/bin/python -m scripts.smoke --live --race 1
.venv/bin/ima-scrape --race 1 --all-pools --without-history --without-horse-pages
```

## Historical Acquisition

Official result collection is checkpointed and resumable:

```sh
.venv/bin/python -m scripts.bulk_history \
  --start 2018-07-01 --end 2025-12-31 \
  --output data/historical/hkjc-2005-2025 \
  --workers 12 --delay 0.1
```

For older seasons, use the reconciled runner archive as a known-meeting index instead of probing every calendar date:

```sh
.venv/bin/python -m scripts.bulk_history \
  --meeting-index data/processed/historical/runners.csv.gz \
  --output data/historical/hkjc-2005-2025 \
  --workers 12 --delay 0.1
```

Build all normalized historical tables:

```sh
.venv/bin/python -m scripts.reparse_official_history
.venv/bin/python -m scripts.build_historical
```

The builder keeps pre-race race cards and timestamped odds separate from final odds, results, and dividends. It also emits sectionals, horse snapshots, trackwork, barrier trials, runner comments, and steward/veterinary incident narratives. See `docs/data/source-registry.json` before redistributing any third-party archive.

## Current Benchmark

The full-history race-grouped benchmark uses 271,868 runners from 22,146 races
spanning 1997-06-02 through 2025-12-27. Its fixed held-out period contains 3,322
races from 2022-01-05 through 2025-12-27. The best tested fundamental model is
`boost-lr003-leaf15` at 22.97% Top-1 accuracy and `2.2367` race log loss; the
final WIN market reaches 30.89% and `2.0135`. Validation therefore assigns zero
fundamental weight to the tested blends. This is a valid abstention signal, not
evidence of a betting edge; forward timestamped odds are required for tradable
paper-trading claims.

## Model Compatibility

The committed joblib files were created with scikit-learn 0.23.2. Loading old pickle/joblib artifacts in a modern environment is both unsafe when provenance is unknown and likely incompatible. The supported path is to reproduce feature generation, validate race/time-aware evaluation, and retrain versioned pipelines that contain preprocessing and estimator together.

See [Target Architecture](docs/architecture/TARGET_ARCHITECTURE.md) for the proposed system.
See [Operations](docs/OPERATIONS.md), [Historical Data Research](docs/data/HISTORICAL_SOURCE_RESEARCH.md), and [Historical Coverage](docs/data/HISTORICAL_COVERAGE.md) for deployment and acquisition details.
