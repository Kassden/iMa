# V5 Implementation Baseline

Recorded 2026-10-01 before changing the live campaign.

- Branch: `feat/agentic-discovery-feedback`, created from main merge `6d64e374`.
- Server: Tailscale `100.95.24.121`; SSH identity verified `uid=1001(imaopt)`.
- V4 user service active: `ima-feature-v4-supervisor.service`. Live policy `feature_v4`, 16-job cap, proposal ceiling 260, planner `deepseek/deepseek-v4.1-flash:floor`.
- Host observation: 121 GiB total RAM, 96 GiB available, 8 GiB swap (2.3 GiB used); shared workloads remain protected. This is a timestamped observation, not a permanent admission guarantee.
- Existing controller submitted all Optuna asks before batch execution and waited for the last result before the next planning boundary. Within submitted batches `as_completed` already replenished pool workers.
- Existing v4 selected registered feature schemas and transforms over its pinned matrix; no Featuretools builder or fold-local generated-feature selection was installed.
- Merged mixed-objective trace preview fix is on main. It will ship with the new pinned successor; v4's revision remains unchanged.
- Remote v5-only dependencies are installed under `research-v2/v5-dependencies/site-packages`, using wheels downloaded on the Mac and transferred over Tailscale. Shared v4 Python site-packages were not modified. Woodwork additionally requires isolated `setuptools==80.9.0` because the shared environment no longer supplies `pkg_resources`.
- Implementation file ownership: `feature_discovery_specs.py` and `feature_program.py` own the declarative DFS/domain generator; `feature_screening.py` owns fold-local filters/selectors; `research_hypotheses.py` owns durable hypothesis records/champion projection; `research_scheduler.py` owns resource accounting; `research_v5.py` owns only v5 orchestration through the existing controller/ledger/search/executor/tracking APIs.
- The new controller is selected by `research_policy=discovery_v5`; there is one controller lock and one ledger/Optuna writer. This is not a second independently running trainer/planner pair.
- Local fixture canary completed ten attempts across conditional logit, boosted win prediction and LambdaRank, followed by a fresh decision containing completed evidence. Fixture planner is a test double, not proof of live agentic planning.
- Regression suites: 67 existing focused tests passed; discovery/state tests passed. Final full-suite and server proofs remain separate gates.
- Unrelated pre-existing worktree changes in v3 config, architecture graph artifacts, temporary files and legacy launchers are preserved and excluded from v5 commits.
