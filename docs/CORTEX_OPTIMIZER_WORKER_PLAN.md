# Cortex Optimizer Worker Setup

## GOAL
- Set up `cortex-server` as a heavy iMa optimizer worker while this Codex session remains the decision-maker, with results pulled back here for review.

## Acceptance Criteria
- [x] A dedicated remote execution boundary exists for iMa optimizer work, OS user `imaopt` with home `/home/imaopt`.
- [ ] Existing `cortex-server` users, repos, services, nginx, PostgreSQL, Tailscale, and `/srv/apps/cortex` are not modified or restarted.
- [x] The iMa checkout on the server is separate from existing cortex repos.
- [ ] The server can run `ima-optimize` dry-run as the isolated runner.
- [ ] The server can run at least one real optimizer trial as the isolated runner, or the exact blocker is recorded.
- [ ] Remote artifacts can be pulled back to this local checkout under `artifacts/remote-cortex/<campaign>/`.
- [ ] Every remote write is read back with `whoami`, `pwd`, ownership, command output, artifact listing, and no-write-outside-boundary checks.

## Out of Scope
- Restarting or editing `cortex-web.service`, `cortex-worker.service`, nginx, PostgreSQL, Tailscale, or any production deployment.
- Touching `/srv/apps/cortex` except read-only inventory.
- Running live betting or race-day estimator work.
- Making cortex-server autonomous. The server is a worker; this Codex session remains the operator/decision loop.
- Storing OpenRouter, MLflow, or HKJC secrets in Git.

## Research
- Built-in options:
  - Use the already implemented `ima-optimize` CLI and append-only campaign artifacts.
  - Use `rsync` for deterministic source/data sync because it is installed on both machines and can exclude `.venv`, `.git`, artifacts, caches, and secrets.
  - Use remote Python virtualenv and editable install rather than system packages.
- Off-the-shelf options:
  - Use SSH key authentication and `sudo useradd`/`install` only for the isolated OS user setup.
  - Use local SQLite MLflow on the server only inside `/home/imaopt/iMa/artifacts/mlflow` if enabled later.
- Official / standards sources:
  - POSIX user isolation and file ownership are the simplest boundary for a shared Linux host.
  - SSH key-only access is the current server access model.
  - Python venv keeps dependencies isolated from system Python and existing apps.
- Evidence from read-only probe on 2026-09-22:
  - Host `cortex-server`, Tailscale `100.95.24.121`, SSH as `cortex` works.
  - Existing non-system user: `cortex` only.
  - Existing production services running: `cortex-web.service`, `cortex-worker.service`, `nginx.service`, `postgresql@18-main.service`, `tailscaled.service`.
  - Existing production path: `/srv/apps/cortex`.
  - Server resources: `121Gi` RAM, `116Gi` available, `28` CPU threads, `/` has about `864G` free.
  - Remote tools present: `/usr/bin/python3` reports Python `3.14.4`; `git`, `rsync`, and `tmux` exist.
  - `sudo -n true` fails for `cortex`; passwordless sudo is not available through that login.
- Evidence from setup attempt on 2026-09-23:
  - Direct Tailscale SSH as `root` works, so scoped admin setup can be performed without using `cortex` sudo.
  - `imaopt` was created as UID/GID `1001`, home `/home/imaopt`, shell `/bin/bash`.
  - `/home/imaopt/iMa` exists and is owned by `imaopt:imaopt`.
  - Tailnet policy does not permit direct SSH as `imaopt`, so automation logs in as `root` and executes optimizer commands with `sudo -u imaopt -H env HOME=/home/imaopt`.
  - Repo/data sync into `/home/imaopt/iMa` completed, with post-sync `chown -R imaopt:imaopt /home/imaopt/iMa`.
  - Server outbound downloads to GitHub/Astral are very slow or stall; use China-accessible PyPI mirrors for Python packages where possible and prefer pre-cached or transferred runtime artifacts for Python itself.
  - Tailscale later reported `cortex-server` offline, last seen `2026-09-22T16:40:00Z`, while the minimal dependency install was in progress.

## SOTA, Standards, And Best Practices
- Current SOTA / prior art:
  - Keep heavy training near CPU/RAM and data; keep model-selection authority near the audit conversation.
  - Use least-privilege OS boundaries for shared servers before process managers or containers.
  - Use append-only experiment artifacts and pullback rather than remote mutable decisions.
- Official docs, standards, specs, or platform guidance:
  - Python venv per checkout, not global `pip`.
  - `rsync --delete` only inside the target checkout, never at home or system roots.
  - SSH `BatchMode=yes` for automation so password prompts fail fast.
- Mature libraries, frameworks, or built-in mechanisms:
  - SSH, rsync, venv, existing `ima-optimize`, existing unittest suite, optional MLflow.
- Best-practice constraints to integrate:
  - No remote write before boundary readback.
  - No sudo operations unless the exact command is scoped to `imaopt`.
  - No service restarts.
  - No secret copying except explicit environment forwarding at run time.
  - Limit server concurrency initially to `--max-concurrent-trials 1`; increase only after measured smoke.
- Rejected approaches:
  - Reject running campaigns directly in `/srv/apps/cortex` or any existing repo.
  - Reject using the existing `cortex` user as the long-lived optimizer runner.
  - Reject Docker/LXD as the first move because a simple OS user is sufficient and lower risk.
  - Reject pushing MLflow UI to Vercel; Vercel remains static dashboard only.
- Implementation decision:
  - Plan and scripts target `imaopt`, `/home/imaopt/iMa`, and `/home/imaopt/iMa/artifacts`.
  - Because passwordless sudo is blocked, implementation pauses before remote user creation. Temporary verification may use read-only probes only; no long-lived runner under `cortex`.

## Dependency and Tooling Preflight
- Required verification tools:
  - Local: SSH, rsync, Git, Python 3.11, iMa tests.
  - Remote: key-only SSH, sudo for user creation, Python 3.11 or compatible project Python, venv, pip, rsync, disk/RAM/CPU readback.
- Existing tooling found:
  - Local Vercel and MLflow are installed, but not needed for this worker setup.
- Remote has Python 3.14.4; compatibility with repo `requires-python >=3.11,<3.14` is not ideal for the canonical venv. The preferred setup is Python 3.11-3.13. If the server cannot download that runtime, a temporary smoke may use Python 3.14 with `--ignore-requires-python` only after tests pass.
- Install or repair commands:
  - Preferred remote runtime: install or transfer Python 3.12 into `/home/imaopt` and create `/home/imaopt/iMa/.venv`.
  - Mirror-aware temporary install path: use `PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple` and `PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn`.
  - Temporary Python 3.14 smoke command may use `.venv314`, minimal runtime dependencies, and `pip install --no-deps --ignore-requires-python -e .`; this is not the final canonical runtime unless verified and intentionally accepted.
- Browser/runtime binaries:
  - None for the worker product surface. This is a terminal/SSH workflow, not a browser UI.
  - Browser smoke, Playwright, screenshot, visual, pixel, canvas, and accessibility proof are not required for the remote worker itself.
  - The generated Megaskill dashboard is a documentation read view only; verification is static HTML generation/readback, not product browser QA.
- Real blockers:
  - `cortex-server` is currently offline on Tailscale, so remote install/run cannot continue.
  - Remote Python is 3.14.4, outside project metadata; Python 3.12 install/download remains unresolved.
  - Direct SSH as `imaopt` is blocked by tailnet policy, so wrapper must keep separate SSH user and worker user semantics.
  - Any evidence that a command would write outside `/home/imaopt`.

## Deterministic Real-User Test
- Entry point:
  - From local operator machine: `scripts/cortex_optimizer_worker.py`.
  - On server: `/home/imaopt/iMa/.venv/bin/ima-optimize` for canonical Python 3.12, or `/home/imaopt/iMa/.venv314/bin/ima-optimize` for explicitly verified temporary Python 3.14 smoke.
- User workflow:
  - Read-only `check` prints server boundary facts.
  - `sync` copies this repo and required data into `/home/imaopt/iMa`.
  - `remote-dry-run` executes `ima-optimize run --dry-run` as `imaopt`.
  - `remote-smoke` executes one local policy trial as `imaopt`.
  - `pull` copies `artifacts/agentic-learning/<campaign>/` back to local `artifacts/remote-cortex/<campaign>/`.
- Stable inputs:
  - Existing historical files: `track/hkracing 2/runs.csv`, `track/hkracing 2/races.csv`, `data/processed/historical/runners.csv.gz`.
  - Dry-run campaign `artifacts/agentic-learning/cortex-smoke`.
- User-observable assertions:
  - Remote `whoami` is `imaopt`.
  - Remote `pwd` is `/home/imaopt/iMa`.
  - CLI dry-run writes `campaign.json` and `dry-run.json`.
  - One-trial smoke writes `trials.jsonl`, `decisions.jsonl`, and `report.md`.
  - Pullback creates matching local files under `artifacts/remote-cortex/cortex-smoke`.
- Commands:
  - `ssh -o BatchMode=yes cortex@100.95.24.121 'hostname; whoami; free -h; nproc; df -h /'`
  - `ssh -o BatchMode=yes root@100.95.24.121 'sudo -u imaopt -H env HOME=/home/imaopt bash -lc "cd /home/imaopt/iMa && .venv/bin/ima-optimize run --campaign artifacts/agentic-learning/cortex-smoke --max-trials 1 --policy local --dry-run"'`
  - `rsync -a --delete root@100.95.24.121:/home/imaopt/iMa/artifacts/agentic-learning/cortex-smoke/ artifacts/remote-cortex/cortex-smoke/`
- Evidence to record:
  - Server read-only probe.
  - User creation readback or sudo blocker.
  - Remote dry-run output.
  - Remote one-trial output.
  - Local pullback listing and report readback.

## Fulfillment and Readback Proof
- Requested final user-visible result:
  - A remote worker that runs optimizer campaigns on cortex-server and makes artifacts available here.
- Required write/mutation operation:
  - Create `imaopt` user.
  - Create `/home/imaopt/iMa`.
  - Sync repo/data into that directory.
  - Create remote `.venv`.
  - Run optimizer commands as `imaopt`.
  - Pull artifacts to local `artifacts/remote-cortex`.
- Readback surface:
  - `getent passwd imaopt`, `id imaopt`, `ls -ld /home/imaopt /home/imaopt/iMa`.
  - Remote `whoami`, `pwd`, `python --version`, `ima-optimize --help`.
  - Remote artifact files.
  - Local pulled report.
- Expected behavior:
  - No writes under `/srv/apps/cortex`.
  - No existing service restart timestamps or process restarts caused by this work.
  - All optimizer artifacts owned by `imaopt`.
- Not-done conditions:
  - Runner uses `cortex` as the long-lived worker.
  - Remote setup requires Python 3.14 for iMa without an explicit passing smoke and documented temporary acceptance.
  - Artifacts are only generated remotely and not pulled/read back.
  - Any existing cortex service or repo is modified.

## Armageddon Mode
- Damage-scaled attack scope:
  - Systemic, because this touches a shared host even if intended as isolated.
- Edge cases to try:
  - `cortex-server` offline.
  - SSH works for `cortex` but not `imaopt`.
  - `sudo` prompts for password.
  - Existing `/home/imaopt` exists with unexpected owner.
  - Remote Python unsupported.
  - Remote data missing.
  - Disk low.
  - `rsync --delete` target path typo.
  - Campaign interrupted and resumed.
  - Multiple concurrent trials overload server.
- Failure modes to simulate:
  - Missing campaign.
  - Invalid concurrency.
  - No OpenRouter key.
  - Pullback missing report.
  - Permission denied reading artifacts.
- Optimization opportunities:
  - Add a reusable local wrapper script so future campaigns use the same safe check/sync/run/pull workflow.
  - Keep `--max-concurrent-trials 1` for first server run; raise to 4 only after resource readback.
- Must-fix threshold:
  - Any command writes outside `/home/imaopt`.
  - Any service restart.
  - Any secret committed or printed.
  - Any remote run not attributable to `imaopt`.
- Evidence to record:
  - Command outputs plus artifact readbacks.

## Generality Guardrail
- Existing mechanism to reuse:
  - `ima-optimize` for optimization, Megaskill ledgers for proof, SSH/rsync for remote execution.
- Similar existing use cases:
  - Future cortex-server campaigns, server MLflow, and model training smoke runs.
- Recurrence likelihood:
  - High. Remote training will recur.
- General mechanism to create:
  - A small wrapper script for remote check/sync/run/pull commands, with safe defaults and hard path guards.
- Specific implementation:
  - `scripts/cortex_optimizer_worker.py` encapsulates the remote worker workflow and refuses unsafe remote roots.
- One-off justification:
  - None; this is recurring operator infrastructure.

## Regression Guardrails
- Planned edit surface:
  - `docs/CORTEX_OPTIMIZER_WORKER_PLAN.md`
  - `scripts/cortex_optimizer_worker.py`
  - `tests/test_cortex_optimizer_worker.py`
  - `.gitignore` only if generated pullback artifacts are not already ignored.
  - Remote write surface after sudo is available: `/home/imaopt` only.
- Protected behaviors:
  - Existing optimizer CLI behavior.
  - Existing dashboard/Vercel deployment.
  - Existing cortex-server services and repos.
  - Existing local artifacts ignored by Git.
- Likely consumers:
  - This Codex operator session, future optimizer campaigns, MLflow/dashboard export workflow.
- Damage radius:
  - Systemic for remote setup; moderate for local wrapper script.
- Branch strategy:
  - Dedicated isolation branch: `audit/run-centric-dashboard-redesign`.
  - This is not `main` or a shared release branch, and it already contains the optimizer commits this worker setup depends on. Do not switch midstream because we need the unpushed optimizer code and plans together. Remote writes remain gated by readback and sudo availability.
- Proof plan:
  - Local unit tests for wrapper safety.
  - Megaskill plan check.
  - Read-only server probe.
  - If sudo becomes available: create user, sync, install, dry-run, one-trial smoke, pullback.
  - Final gate is `blocked` if sudo/user creation remains unavailable.

## Phase 1: Plan

### Subphase 1.1: Write and Validate Remote Worker Plan
- Commit: `docs(cortex): plan optimizer worker isolation`
- Tests: `python3.11 /Users/milkingthesun/.codex/skills/megaskill/scripts/mega_plan_check.py docs/CORTEX_OPTIMIZER_WORKER_PLAN.md`
- Success Criteria: Plan passes Megaskill checks and names the exact server boundary, blockers, and verification commands.
- Planned Touch Files:
  - `docs/CORTEX_OPTIMIZER_WORKER_PLAN.md`
- Checklist:
  - [ ] Write concrete plan.
  - [ ] Validate plan.
  - [ ] Generate dashboard.

## Phase 2: Server Discovery

### Subphase 2.1: Read-Only Shared Host Inventory
- Commit: `docs(cortex): record worker discovery evidence`
- Tests: Read-only SSH probe commands.
- Success Criteria: We know users, protected services, resources, Python version, rsync/git availability, and sudo state.
- Planned Touch Files:
  - `.mega/evidence.jsonl`
  - `.mega/state.jsonl`
- Checklist:
  - [ ] Probe Tailscale/SSH.
  - [ ] Probe users/services/resources.
  - [ ] Confirm sudo availability or blocker.

## Phase 3: Local Remote-Worker Wrapper

### Subphase 3.1: Add Safe Check/Sync/Run/Pull CLI
- Commit: `feat(cortex): add optimizer worker wrapper`
- Tests: `.venv/bin/python -m unittest tests.test_cortex_optimizer_worker -v`
- Success Criteria: Wrapper builds safe SSH/rsync commands, refuses unsafe remote roots, supports check/sync/run/pull, and does not execute unsafe paths in tests.
- Planned Touch Files:
  - `scripts/cortex_optimizer_worker.py`
  - `tests/test_cortex_optimizer_worker.py`
- Checklist:
  - [ ] Implement command builder and path guards.
  - [ ] Implement `check`, `sync`, `remote-dry-run`, `remote-run`, and `pull`.
  - [ ] Add unit tests.

## Phase 4: Isolated User Setup

### Subphase 4.1: Create `imaopt` User
- Commit: none for remote-only operation; record evidence.
- Tests: `ssh root@100.95.24.121 'getent passwd imaopt; id imaopt; ls -ld /home/imaopt /home/imaopt/iMa'`
- Success Criteria: `imaopt` exists, owns `/home/imaopt` and `/home/imaopt/iMa`, and no existing service/repo was touched. Direct SSH as `imaopt` is optional because current tailnet policy blocks it.
- Planned Touch Files:
  - Remote only: `/home/imaopt`
- Checklist:
  - [x] Obtain admin-side execution through direct `root` Tailscale SSH.
  - [x] Run scoped `useradd`/`install` commands.
  - [x] Read back user/home/authorized keys.
  - [x] Verify sudo-mediated `imaopt` command execution.

## Phase 5: Repo Sync and Remote Python Setup

### Subphase 5.1: Sync iMa and Build Remote Venv
- Commit: none for remote-only operation; record evidence.
- Tests: `ssh root@100.95.24.121 'sudo -u imaopt -H env HOME=/home/imaopt bash -lc "cd /home/imaopt/iMa && .venv/bin/python --version && .venv/bin/ima-optimize --help"'`
- Success Criteria: Separate checkout exists at `/home/imaopt/iMa`, dependencies install with supported Python, and CLI help works.
- Planned Touch Files:
  - Remote only: `/home/imaopt/iMa`
- Checklist:
  - [x] Sync repo/data into `/home/imaopt/iMa`.
  - [ ] Create `.venv` with Python 3.11-3.13.
  - [ ] Install editable package.
  - [ ] Run CLI help.

## Phase 6: Remote Smoke and Pullback

### Subphase 6.1: Run Remote Campaign and Pull Artifacts
- Commit: none for generated ignored artifacts; record evidence.
- Tests:
  - Remote dry-run command.
  - Remote one-trial smoke command.
  - Local pullback listing and report readback.
- Success Criteria: Campaign artifacts exist remotely and locally after pullback.
- Planned Touch Files:
  - Remote: `/home/imaopt/iMa/artifacts/agentic-learning/cortex-smoke`
  - Local ignored: `artifacts/remote-cortex/cortex-smoke`
- Checklist:
  - [ ] Run dry-run as `imaopt`.
  - [ ] Run one-trial smoke as `imaopt`.
  - [ ] Pull artifacts.
  - [ ] Read local `report.md`.

## Phase 7: Final Verification

### Subphase 7.1: Armageddon and Final Gate
- Commit: `test(cortex): verify worker wrapper`
- Tests:
  - Wrapper tests.
  - Megaskill scope check.
  - Megaskill final gate.
- Success Criteria: Done if isolated user and smoke pass; blocked if sudo/Python setup prevents user creation or supported venv.
- Planned Touch Files:
  - `.mega/evidence.jsonl`
  - `.mega/state.jsonl`
- Checklist:
  - [ ] Run local tests.
  - [ ] Record server blocker or success.
  - [ ] Run final gate with accurate verdict.
