# Feature Discovery V4 Operations

V4 is a research campaign, not a race-day or betting service. It runs as `imaopt`
in its own campaign directory and MLflow experiment. No candidate is promoted
to race-day use automatically.

## Identity

- Release: `/home/imaopt/research-v2/live-releases/ima-v4-current` points to an immutable release with a `REVISION` file.
- Campaign: `/home/imaopt/research-v2/campaigns/agentic_v4_features`.
- Config: `campaign/ops/openrouter-config.json`; dataset and protocol are pinned under `campaign/inputs`.
- Unit: `ima-feature-v4-supervisor.service` in the `imaopt` user systemd manager.
- Tracking: `ima-agentic-v4-features` MLflow experiment; registered models are research candidates only.
- The OpenRouter route is direct HTTPS from the server. Verify a real authenticated completion, not only DNS or a GET, before claiming independence from the laptop.
- Server DNS has shown a transient resolution failure. The supervisor retries a paused planner every five minutes; inspect `decisions/cycle-*.json` and `ops/supervisor.log` if no new trials appear.
- Keep the initial 16-worker ceiling until measured memory headroom justifies more. The first 25-trial v4 wave used 16 active workers, left about 44 GiB available at peak, and did not increase swap use.

## Preflight

1. Verify the release revision, dataset and protocol hashes, Python imports, OpenRouter credentials, and direct API completion.
2. Run a bounded v4 canary in a *different* campaign directory. Confirm completed attempts, finite per-race probabilities summing to one, feature-program artifacts, MLflow run/decision traces and planner cost.
3. Generate a development-only matched-race report with `python -m scripts.report_v4_programs --campaign CAMPAIGN`. A lower point estimate without a paired interval excluding zero is not evidence of a reliable gain.
4. Verify v3 is active and record its current decision, running attempt count, and last completed result. Do not edit its release, config, ledger or other users' services.

## Handoff

1. Install the v4 launcher in `/home/imaopt/.local/bin/` and user unit in `/home/imaopt/.config/systemd/user/`. Point `ima-v4-current` at the verified immutable release and run `systemctl --user daemon-reload`.
2. After the canary gate, request v3 STOP using `python -m scripts.optimize stop --campaign /home/imaopt/research-v2/campaigns/agentic_v3_fundamental`. STOP is checked between completed cycles. Do **not** signal/kill workers to force a boundary.
3. Poll until v3 `status.json` reports `stopped`, SQLite has zero running attempts, and `ima-fundamental-v3-supervisor.service` is inactive. Check for any rollover watcher before proceeding. Disable the v3 user unit only after those conditions hold.
4. Enable and start `ima-feature-v4-supervisor.service`. Verify its PID, service log, and first completed trial and decision. Watch two decisions/cycles, the direct planner route, costs, CPU/RAM and Cortex/solar service health.

For a long final v3 trial, `ima-v4-handoff.service` can perform steps 3-4 unattended. Its `--check` mode is read-only; the watcher requires v3 service inactive, `stopped` status, zero running/reserved attempts, and the pinned v4 revision before it switches units. Confirm its journal and the v4 run after it fires.

## Failure and Rollback

- If a canary fails, keep v3 running. Preserve artifacts and planner rejection reasons; do not switch services.
- If v4 fails after handoff, request its graceful STOP. Once its active cycle ends, inspect the failure and only then explicitly start the preserved v3 unit if research continuity is needed. Do not run both high-concurrency campaigns unattended.
- A stale `status.json`, a live process, and MLflow traces are different signals. Use SQLite and process/service readback to decide whether training is actually progressing.
- Keep the later held-out period sealed during proposal search. A separate, one-time promotion review must authorize its evaluation, and deployment still requires human approval.
