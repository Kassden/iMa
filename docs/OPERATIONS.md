# Operations

## Runtime Boundary

Run collection and decision generation from a host whose public egress is accepted as Hong Kong by HKJC. Chrome or Chromium is required because direct GraphQL requests currently return `WHITELIST_ERROR` even from the accepted mobile egress.

Use Python 3.11 and keep raw snapshots, model artifacts, and the prediction ledger on persistent storage. Set `Asia/Hong_Kong` as the scheduling timezone. The repository defaults to paper mode.

## Initial Verification

```sh
.venv/bin/python -m unittest discover -v
.venv/bin/python -m scripts.smoke --live --race 1
.venv/bin/ima-scrape --race 1 --all-pools --without-history --without-horse-pages
.venv/bin/python -m scripts.train --output artifacts/models/latest
```

## Meeting Cycle

1. Collect timestamped odds repeatedly before each race and preserve raw snapshots.
2. Generate predictions and recommendations from one pinned model version.
3. Record the decision before post time and execute in paper mode.
4. After official declaration, collect results and dividends and settle the ledger.
5. Finalize the complete meeting, version the enriched dataset, and train challengers.
6. Compare challengers with the champion and market baseline on untouched recent races.
7. Promote only after metric gates and operator approval. Live approval is separate.

## Credentials and Live Wagers

HKJC account credentials must be retrieved at runtime from macOS Keychain or an external vault. They must not be stored in `.env`, source control, browser profile archives, logs, screenshots, or model artifacts.

The current `HKJCWebExecutor` deliberately blocks live submission. Enabling transactions requires an authenticated browser session, multi-factor handling, bet-slip confirmation parsing, idempotent receipt capture, account/race exposure limits, kill switch, and explicit operator approval. Collection and model processes must not hold transaction credentials.

## Scheduling

`deploy/launchd` targets this Mac. Replace `__IMA_REPO__` with the checkout path before loading the plist. `deploy/systemd` targets a persistent Linux host under `/opt/ima`. Retraining is scheduled after typical Wednesday and Sunday meetings; official-result reconciliation should precede it and incomplete meetings must be quarantined.
