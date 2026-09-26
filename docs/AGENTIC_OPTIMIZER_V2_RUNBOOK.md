# Agentic Optimizer V2 Runbook

## Local Deterministic Test

```bash
.venv/bin/python -m unittest tests.test_agentic_optimizer_e2e
```

The test launches the terminal optimizer with the fixture planner, trains nine
real recipes, and verifies bootstrap plus two evidence-driven planning cycles.
It also verifies offline status and that preview mode creates no Optuna study.

The current controller stores each planner direction in `search/programs.jsonl`.
Each program has a fixed target, structural recipe, bounded model-parameter
search space, parents, and trial budget. `search/program-journal.log` contains
one Optuna study per program. Every executed trial has Optuna parameters; the
planner does not compete with Optuna for individual parameter values. Evidence
files contain the recent recipes, comparable target leaders, program outcomes,
failures, and transform-input coverage. A failed planner call falls back to a
bounded local direction without changing the frozen dataset or protocol.

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

### Runs, Models, and Traces

- **Run:** one trained recipe attempt. Its searchable metrics include the
  objective, every fold, model/baseline/control comparisons, and mean/std/worst
  fold summaries under `summary.*`. Searchable `program_id`, `trial_id`, and
  `proposal_id` parameters link the run to the durable research direction.
- **Model:** the loadable package produced by that run, registered under the
  target-specific model name. Models are versioned independently of traces.
- **Trace:** one optimizer decision cycle. The root `optimizer-cycle-NNNN` span
  contains evidence identity and comparable per-objective outcomes. `planner-decision`
  records hypotheses, recipes, rejected proposals, token usage, and provider
  cost. Each executed attempt has a child `trial-*` span with its target,
  objective, status, duration, and metric summary.

Cycle linkage is persisted at `CAMPAIGN/traces/cycle-NNNN.json`; retrying the
same cycle reads that linkage instead of creating a duplicate trace. Trace
failure is added to `tracking_errors` and does not stop model training.

OpenRouter cost is never reconstructed from a price table. When the provider
returns `usage.cost` or `usage.total_cost`, MLflow receives that exact USD value
and standard token attributes. When the provider omits cost, the trace is tagged
`ima.cost_status=unavailable`, `total_cost_usd` is null, and the UI must not be
interpreted as a confirmed zero-cost call. Local, fixture, and Optuna-only
decisions naturally have unavailable LLM cost.

In MLflow, open experiment `ima-agentic-v2`, use **Runs** for model-attempt
comparison, **Models** for registered package versions, and **Traces** for the
planner-to-results narrative and cost.

Do not restart Cortex, solar simulator, nginx, postgres, or other workloads.

The ongoing v12 campaign at
`/home/imaopt/research-v2/campaigns/feedback-canary-v12` uses immutable release
`f54c604` in `ima-feedback-v2`. It is a legacy-search control, not a campaign
to migrate in place. Start hierarchical-search work only in a new release and
campaign directory; never reuse v12's ledger or journal.

The isolated workflow canary is
`/home/imaopt/research-v2/campaigns/hierarchical-canary-v2`, using
`canary-releases/hierarchical-v2` and code revision `00e4c6a`. Its six fixture
trials completed with two planner cycles, two cycle traces, and no pending
Optuna or MLflow updates. The second cycle trained both newly approved
programs. This small synthetic fixture verifies orchestration and tracking,
not improved race prediction. The live v12 campaign still uses its previous
search implementation until separately migrated.

## Historical Live Campaign

The 2026-09-25 Cortex acceptance campaign is
`/home/imaopt/research-v2/campaigns/feedback-canary-v10`, running immutable
revision `a6932b5` in tmux session `ima-feedback-v2`. It uses
`deepseek/deepseek-v4-pro-0813`, exact endpoint `baidu/fp8`, a `$1` planner
spend cap, unlimited completed-trial budget, and automatic worker admission.

```bash
cd /home/imaopt/research-v2/releases/a6932b5
PYTHONPATH=. .venv/bin/python -m scripts.optimize status \
  --campaign /home/imaopt/research-v2/campaigns/feedback-canary-v10

PYTHONPATH=. .venv/bin/python -m scripts.optimize stop \
  --campaign /home/imaopt/research-v2/campaigns/feedback-canary-v10

PYTHONPATH=. .venv/bin/python -m scripts.optimize run \
  --campaign /home/imaopt/research-v2/campaigns/feedback-canary-v10 \
  --config /home/imaopt/research-v2/config/feedback-canary-v10.json \
  --max-trials unlimited --max-concurrent-trials auto
```

See `AGENTIC_OPTIMIZER_ACCEPTANCE_REPORT.md` for provider hashes, trial IDs,
MLflow readback error, restart proof, resources, and the shared-service audit.
