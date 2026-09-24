# Agentic Optimizer V2 Runbook

## Local Deterministic Test

```bash
.venv/bin/python -m unittest tests.test_agentic_optimizer_e2e
```

The test launches the terminal optimizer in `--policy agentic --dry-run` mode,
resumes the same campaign, verifies recipe persistence, then saves and reloads a
research package.

## Local Dry Run

```bash
.venv/bin/ima-optimize run \
  --policy agentic \
  --campaign artifacts/agentic-learning/local-agentic-smoke \
  --max-trials 6 \
  --proposal-batch-size 3 \
  --dry-run
```

## Cortex Canary

Use only the isolated worker checkout:

```bash
python -m scripts.cortex_optimizer_worker run \
  --remote-root /home/imaopt/research-v2 \
  --campaign artifacts/agentic-learning/cortex-agentic-smoke \
  --max-trials 6 \
  --proposal-batch-size 3 \
  --max-concurrent-trials 2 \
  --dry-run
```

Do not restart Cortex, solar simulator, nginx, postgres, or other workloads.
