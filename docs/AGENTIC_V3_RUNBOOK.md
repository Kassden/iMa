# Agentic V3 Research Campaign

`agentic_v3` is an opt-in research campaign. It never places wagers or promotes a
model for race day. The existing v17 controller, release, campaign, and MLflow
experiment remain separate.

## Scientific Policy

- Every five durable attempt reservations contain four Benter conditional-logit
  trials and one experimental trial. The experimental slots rotate E1, E2, E3.
- Benter trains race-level winner likelihood without current-race odds as
  features. It reports standalone, raw-market, calibrated-market, and combined
  scores on protected chronological folds.
- E1 trains LightGBM grouped ranking and reports NDCG@3. Its separate calibrated
  win-probability endpoint reports race log loss against calibrated market.
- E2 trains CatBoost for literal top-3 finish probability. It is not an HKJC
  paid-place probability until payout rules are modeled.
- E3 trains CatBoost to predict recorded final win odds from pre-race features.
  It is not a timestamped intra-race odds forecast or an executable price.
- The orchestrator may choose feature schema, ablations, transforms, search
  bounds, and total program budget within the assigned ID. Optuna chooses trial
  parameters. A missing program pauses its slot; no lane borrowing is allowed.

## Operator Commands

Use the dedicated `imaopt` account on `cortex-server` over Tailscale. The
versioned configuration is `config/agentic_v3.json`. Its dataset path points to
the existing rich-history file read-only. The campaign directory is
`/home/imaopt/research-v2/campaigns/agentic_v3`.

```sh
python -m scripts.optimize status --campaign /home/imaopt/research-v2/campaigns/agentic_v3
python -m scripts.optimize stop --campaign /home/imaopt/research-v2/campaigns/agentic_v3
```

Do not use the old v17 release path for v3. A release needs its own code revision
and research-only LightGBM/CatBoost dependencies. A bounded real-data canary
must complete B, E1, E2, and E3 and prove package reload plus MLflow readback
before starting continuous work. Resume only the same immutable code, dataset,
protocol, and portfolio identity; otherwise start a new campaign.

## Readback

- `campaign-identity.json`: dataset, code, protocol, environment, portfolio.
- `ledger.sqlite`, `trials.jsonl`, `decisions.jsonl`: durable reservations,
  results, and planner choices.
- `trials/<attempt>/package`: loadable model and recipe manifest.
- MLflow experiment `ima-agentic-v3`; registered models begin
  `ima-agentic-v3-candidates`. Filter runs by `experiment_id`, `target_kind`,
  and `portfolio_version`. Do not compare raw objectives across targets.
- A planner outage, spend cap, or resource pause must be visible in `status.json`.
  No status alone proves recovery; check a later natural trial and MLflow trace.
