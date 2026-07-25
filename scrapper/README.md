# iMa Live Race Scraper

This package fetches an HKJC race, maps WIN and PLACE odds to each declared runner, follows every active horse to its HKJC pages, and writes the exact feature columns used by the legacy `all_train_set.csv` notebook workflow.

## Run

```sh
python3 -m scrapper.cli --race 1 --date 2026-07-25 --venue ST
```

The default output directory is `scrapper/snapshots/`. Each poll writes:

- An immutable raw GraphQL response.
- A normalized runner snapshot.
- Normalized horse profile, form, trackwork, veterinary, and movement records.
- A model-compatible CSV.
- A readiness report listing missing features and inactive runners.

Use a recorded response when HKJC blocks the current network:

```sh
python3 -m scrapper.cli \
  --race 1 \
  --fixture scrapper/tests/fixtures/race_snapshot.json \
  --output /tmp/ima-snapshots
```

## Model Readiness

The live API directly supplies runner identity, draw, carried/body weight, rating, gear, jockey, trainer, and current odds. The scraper then reads each horse's form page to derive prior speed, results, odds, weights, experience, and cumulative finish statistics. Trackwork, veterinary, and movement records are preserved as structured enrichment for future models.

Current horses do not overlap the repository's 1997-2005 numeric horse IDs. Do not fill missing history with zero. The generated CSV marks each row with `prediction_ready` and lists missing fields. A current historical-results provider and horse identity map are required before live rows can be passed safely to the existing model.

An optional identity map is a CSV with:

```csv
live_horse_code,historical_horse_id
H001,3917
```

## Verify

```sh
python3 -m unittest discover -s scrapper/tests -v
```
