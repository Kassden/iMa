# Agentic Optimizer Acceptance Report

Snapshot: 2026-09-25 14:48 CST

## Deployment

- Branch: `fix/agentic-feedback-controller`
- Revision: `a6932b590a4c9c23beafc2d36e1ec1f9dd6d88a8`
- Release: `/home/imaopt/research-v2/releases/a6932b5`
- Campaign: `/home/imaopt/research-v2/campaigns/feedback-canary-v10`
- Config SHA-256: `fe164ed8d739633c96f37e1671c07627cd23807fa8ad2c362fe8319dd873d832`
- Dataset SHA-256: `ab437757562c5bf914bae1741a8c80d55b3c647a51f99bc08d40495e49221160`
- Protocol SHA-256: `d8895620a08a27e1a7e134231f857c03473a1beec4ba997e9c0ce26f82c431ce`
- Environment SHA-256: `2827ddd042f17d0f82dfb5473fddd8fff1ebd512694e24110b2b1290d6f539c4`
- Model route: `deepseek/deepseek-v4-pro-0813` through exact endpoint `baidu/fp8`, fallbacks disabled.
- OpenRouter request tier: `flex`; returned tier: `null`. Flex is not claimed as provider-confirmed on this route.
- Reasoning: `none`, leaving the bounded completion budget for strict recipe JSON.

## Real Provider Evidence

The corrected v10 campaign preserved monotonically increasing cycle IDs across a controlled stop and resume. The bootstrap decision hash remained `ba46b18f4155f08768f64501a70f9c0f18e087ea3739f584dda58f2b0616f11f` after restart.

- Cycle 1: transient TLS EOF, durably recorded as `degraded_local`; three approved-space recipes continued training.
- Cycle 2: DeepSeek/Baidu, evidence `1cf2f322c7c874e6`, three valid proposals, cost `$0.00187677876`, artifact SHA-256 `85507cc3be45db8d42bcd175ca182dadeede070e77a5f18b186486f12dd4aa57`.
- Cycle 3: DeepSeek/Baidu, evidence `2702337069c6c47c`, cost `$0.0020438814`, artifact SHA-256 `b235a349511426c65e9ffc814cce4e64de63133717ceac20b1aba8bd835001e5`.
- Provider responses reported zero reasoning tokens and `service_tier: null`.

The first six successful v10 attempts were:

1. `attempt-b83ceab60aaa9b92eaf2ca474c7fffabccb5f0a3b5a0c4c116dd66779c804120`
2. `attempt-1b938c84198b362a4544075a6fe4224012ce871409fe76a60718c4c76938e99c`
3. `attempt-aa4bfa1cc3d2fde081dc41a7edf879e5751f0d7232baa4af8c743516cf1723c2`
4. `attempt-03c10acf51ce489e86a1f97da647d80af442a88fbc04c3bf3b710e0ffbaee9e8`
5. `attempt-a11d272754e0c22c045714adba9a6a33fb60087051b5fb71749b9060fa107ace`
6. `attempt-b1d049e7fdf8fbb27898060b2357afc7526b739f6766c7c37d75ee54d4be8351`

At the snapshot, v10 had 10 completed trials and two running trials. Its best protected development race log loss was `2.0085626883063323`; this is research progress, not holdout proof or betting profitability.

## Registry Readback

MLflow is reachable at `http://100.95.24.121:5000`. Every completed v10 trial at the snapshot had durable run/version linkage and zero pending uploads. Registered version `42` was loaded through `models:/ima-agentic-candidates-win-probability/42` and reproduced 12,280 saved `fold-003` predictions with maximum absolute error `9.974659986866641e-17` (`numpy.allclose` true).

## Secondary Targets

The server-side deterministic probe `test_secondary_targets_train_score_controls_and_reload` passed for ranking strength, fixed top-k placing, and adjusted finish-time/speed. Each trained a real adapter, scored the shuffled control, loaded its fitted package, and reproduced target-specific predictions. Odds forecasting remains correctly blocked without timestamped pre-race/future snapshot pairs.

## Recovery And Resources

- The initial canary used two workers. The final unlimited resume used `--max-concurrent-trials auto`; measured admission selected three workers for the three-recipe batch.
- At the 34-second sample, the three workers used about 11.9 GiB RSS total and the host retained 111.5 GiB available memory.
- Stop markers drained admitted batches and were consumed. Resume preserved completed attempts, Optuna tells, MLflow links, and cycle files.
- The v9 campaign exposed cycle-file overwrite on process resume. Revision `a6932b5` fixes this by deriving the next cycle from durable decisions and artifacts; v10 proved the bootstrap hash remained unchanged.

## Shared Host Boundary

All optimizer writes stayed under `/home/imaopt/research-v2`; no optimizer command restarted or reconfigured shared services. Solar simulator, nginx, and PostgreSQL start timestamps remained unchanged. `cortex-web.service` independently deactivated successfully at 14:35:49 and systemd auto-restarted it at 14:35:54 (`NRestarts=4`); the bounded journal had no optimizer process or command associated with that event.

## Operator Commands

```bash
# Status
cd /home/imaopt/research-v2/releases/a6932b5
PYTHONPATH=. .venv/bin/python -m scripts.optimize status \
  --campaign /home/imaopt/research-v2/campaigns/feedback-canary-v10

# Graceful stop
PYTHONPATH=. .venv/bin/python -m scripts.optimize stop \
  --campaign /home/imaopt/research-v2/campaigns/feedback-canary-v10

# Resume indefinitely with measured admission
PYTHONPATH=. .venv/bin/python -m scripts.optimize run \
  --campaign /home/imaopt/research-v2/campaigns/feedback-canary-v10 \
  --config /home/imaopt/research-v2/config/feedback-canary-v10.json \
  --max-trials unlimited --max-concurrent-trials auto
```
