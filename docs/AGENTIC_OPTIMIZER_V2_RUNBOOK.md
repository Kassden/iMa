# Agentic Optimizer V2 Runbook

## Local Deterministic Test

```bash
.venv/bin/python -m unittest tests.test_agentic_optimizer_e2e
```

The test launches the terminal optimizer with the fixture planner, trains nine
real recipes, and verifies bootstrap plus two evidence-driven planning cycles.
It also verifies offline status and that preview mode creates no Optuna study.

## Local Dry Run

```bash
.venv/bin/ima-optimize run \
  --policy agentic \
  --campaign artifacts/agentic-learning/local-agentic-smoke \
  --max-trials 6 \
  --proposal-batch-size 3 \
  --dry-run
```

Dry run never calls OpenRouter, fits a model, or advances the persistent study.

## Campaign Configuration

Use a strict JSON file. Credentials stay in the environment.

```json
{
  "schema_version": 1,
  "policy": "agentic",
  "planner_mode": "openrouter",
  "model": "deepseek/deepseek-v4-pro-0813",
  "service_tier": "flex",
  "provider_endpoint": "baidu/fp8",
  "planner_reasoning_effort": "none",
  "max_total_cost_usd": 1.0,
  "max_output_tokens": 2400,
  "max_trials": 9,
  "proposal_batch_size": 3,
  "max_concurrent_trials": 2,
  "replan_every_terminal_trials": 3,
  "mlflow_tracking_uri": "http://100.95.24.121:5000"
}
```

The Cortex canary pins `baidu/fp8` with provider fallbacks disabled. A live
probe on 2026-09-25 confirmed that this endpoint served the requested DeepSeek
model. OpenRouter returned `service_tier: null`, so Flex is a request preference,
not a verified provider tier for this route. The explicit `none` reasoning
setting preserves the bounded completion budget for the strict JSON recipe;
without it, this model consumed the completion budget in hidden reasoning and
returned no usable content.

```bash
.venv/bin/ima-optimize run \
  --campaign artifacts/agentic-learning/feedback-canary \
  --config campaign.json

.venv/bin/ima-optimize status \
  --campaign artifacts/agentic-learning/feedback-canary

.venv/bin/ima-optimize stop \
  --campaign artifacts/agentic-learning/feedback-canary
```

`stop` is consumed on acknowledgement. Running the same campaign again resumes
from its ledger and Optuna journal. Dataset, protocol, code, environment, target
contract, and metric identity are immutable within a campaign.

## Cortex Canary

Use only the isolated `imaopt` checkout and a distinct tmux session. Inspect
shared services read-only before and after; do not restart them.

```bash
python -m scripts.cortex_optimizer_worker run \
  --remote-root /home/imaopt/research-v2 \
  --campaign artifacts/agentic-learning/feedback-canary \
  --config campaign.json \
  --max-trials 9 \
  --proposal-batch-size 3 \
  --max-concurrent-trials 2
```

Start at two workers. Observe CPU percent, load percent, available memory, disk,
and the owned process tree for at least 30 seconds before increasing the cap.
Defaults pause below 10 GiB disk and reserve the greater of 16 GiB or 20% RAM.
Operational overrides are `IMA_RESEARCH_MAX_WORKERS`,
`IMA_RESEARCH_RESERVE_MEMORY_GIB`, `IMA_RESEARCH_TRIAL_RSS_GIB`,
`IMA_RESEARCH_DISK_PAUSE_GIB`, and `IMA_RESEARCH_CPU_CEILING_PERCENT`.

Successful trials register target-specific models in MLflow under
`ima-agentic-candidates-<target>`. The campaign ledger stores the MLflow run,
model name, registered version, and URI. A tracking outage leaves
`pending_tracking` nonzero and retries on the next controller run.

Do not restart Cortex, solar simulator, nginx, postgres, or other workloads.
