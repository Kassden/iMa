# Research Expansion Validation

Evidence date: 2026-10-04 UTC. Scope: CI/CD/OPS on
`feat/agentic-research-expansion`; no staging, commits or branch changes. Remote
dependency setup under `v6-dependencies` was subsequently authorized; no V5 or
service mutations. Other agents own implementation/config and coordinate cutover. This report
does not certify the whole plan or V6 launch.

## Evidence Status

| Check | Actual result | Interpretation |
|---|---|---|
| Canonical megaskill plan check | Passed | Plan structure only |
| Parent-provided pre-change suite | 334 passed in 67.307s | Supplied baseline; not rerun at that revision by this role |
| Existing CLI `python -m scripts.optimize run --help` | Passed locally | `run --campaign PATH --config PATH` exists |
| Local `pip check` | Passed | Existing local environment only |
| Existing locks/extras pip resolution | Local `pip install --dry-run` passed, exit 0 | No installation; not a fresh Linux environment proof |
| Corrected targeted regression command | 29 passed in 0.160s | `tests.test_optimizer tests.test_research_resources tests.test_research_discovery_state` |
| Initial targeted command | Failed to import `tests.test_research_scheduler` | That planned test module did not exist; corrected command above |
| New launcher `bash -n` | Passed | Syntax, not deployed execution |
| Full working-tree suite | 334 passed in 67.147s, exit 0 | Collected while concurrent implementation proceeded; not all later files were included |
| Earlier successor/telemetry tests | 60 ran in 0.259s; eight failures, five errors; exit 1 | Historical in-progress snapshot, superseded below |
| Latest successor/telemetry tests | 86 ran in 24.235s; three failures, zero errors; exit 1 | `2026-10-04T06:48Z`; real-CLI canary failure resolved, remaining IDs below |
| Workflow YAML and guardrails | Passed | Read-only token, existing test command, no secret/deployment references |
| Launcher failure cases | Passed | Unknown args/extra args rejected with 64; missing environment rejected with 1 before credential access |
| Embedded launcher preflight | One valid fixture passed; five adversarial fixtures rejected | Temporary local root substitution; revision, predecessor policy/campaign, unknown key, missing/relative input; no ledger created |
| Linux unit validation | `systemd-analyze verify --man=no` passed, exit 0 | Systemd 252 package installed in disposable Debian container; actual unit verified; container removed automatically |
| GitHub Actions | Workflow authored; not remotely run | No credentials, paid planning or deployment in workflow |
| Parent final config schema | Passed via `_load_config` and `CampaignConfig.validate()` | Working tree only; no campaign executed |
| Parent canary CLI `--help` | Passed | Local fixture execution/tracking now verified below; official/live runtime gates remain pending |
| Effective remote V5 `pip check` | Exit 1: missing JupyterLab, Graphviz, Plotly | Preserve V5; repair only private V6 dependency closure |
| Remote offline locked pip dry-run | Exit 1: no Pydantic 2.13.4 wheel in existing stores | Private locked environment not ready from existing wheels alone |
| Authorized private copy-mode setup | Offline install and `pip check` exit 0 | 120 original versions preserved; 189 effective distributions, 25 private module origins |
| Parent fixture canary | Parent CLI exit 0; readback mode complete, 10 completed, three decisions, pending tells 0, tracking 10 | Synthetic/offline; not paid or advanced dataset/graph acceptance |
| Parent fixture champion replay | Two families, prediction/objective delta 0 | Local artifact readback; unknown release/environment lineage, not pinned V5 replay |
| Parent local MLflow fixture canary | Read-only SQLite: five FINISHED runs, five READY linked model versions, four OK traces; status complete, five completed, tells/tracking 0 | Local fixture tracking only, not remote MLflow/paid/native USD UI proof |
| V6 live canary, replay, UI and sustained feedback | Not run by this role | Parent coordinates cutover; launch gates remain open |

## CD Preparation Refresh

Initial assignment was read-only remotely; source deployment and launch remain
held until parent gates are green. Exact current test failures are below.
Own-user venv/lock/extra inventory, actual OpenRouter Tailscale routes, frozen
snapshot transfer sizes and the prepared V5 STOP procedure are in the runbook.
No STOP, config, route or service action was performed. Later authorization
allowed only private dependency setup; its separate receipt is recorded below.

Local package verification exercised a scoped `git archive --format=tar` from
committed SHA `6ef7481dc93f406e77174d510b36d99155ef38b4`, 1,515,520 bytes.
Its committed source integrity passed; deployment readiness correctly failed for
14 missing required paths, including the then-not-yet-created successor config and
uncommitted controller/service/docs/tests. Temporary test archives were removed;
this is a verification receipt, not a prepared V6 release artifact. No source was
copied from dirty/untracked workspace files into the archive.

Archive adversarial checks passed: unknown commit, non-full SHA, modified content,
duplicate file, symlink, traversal path, omitted file and wrong PAX commit metadata
were rejected. Full V6 required-path success remains pending a parent commit.
The verifier outputs package SHA256/size only after complete source/readiness
checks; it does not certify model/data/recovery gates or extract/deploy anything.
CI now runs that committed-SHA package check after the regression suite, without
uploading/deploying a package. Hosted CI remains unrun.

Runbook schema example was accepted through the real `scripts.optimize.main`
config loader/constructor with `run_campaign` mocked to validate only. No campaign
was created, no planner called and no `config/agentic_research_expansion.json`
workspace file written by this role. Parent has since supplied the final config
and canary. Their schema/help checks passed; approved model is
`deepseek/deepseek-v4.1-flash`, cap $5, campaign
`/home/imaopt/research-v2/campaigns/agentic_v6_research`, ceiling 26 workers,
24 CPU threads, 100,000,000,000 bytes / 93.13225746154785 GiB. A ceiling is not
an actual admission count. Loader normalizes the 260-trial decision alias.
Parquet readers exist in profile and executor; actual preparation/training on
the frozen Parquet input remains a separate gate. The package verifier now
also requires the committed canary script. Committed readiness remains pending.
Launcher was corrected for the observed own-user Python executable symlink;
existing V5/acquisition runtimes were not changed. Bash syntax, CI YAML guardrails
and the canonical plan checker passed locally.

Interpreter path fixtures passed for a regular in-root executable and own-runtime
symlink; unapproved executable target, out-of-root entry point and escaping dataset
symlink were rejected. No fixture campaign ledger was created. Final own-user
readback at `2026-10-04T06:22:07.472087Z` matched all ten protected hashes,
V5 `training` / cycle 123 / 434 completed / two failed / eight running / zero
pending tells and tracking; V5 STOP absent, V6 unit absent, MLflow health 200.

CI/CD/OPS owned artifacts are ready for parent integration. Whole-plan verdict is
not done: successor tests failed at the later snapshot, hosted CI has not run and
V6 launch/readback is parent-owned. The megaskill whole-plan final gate with
`--verdict blocked` passed against the shared execution state; it is not an
acceptance pass. No `.mega` records were written by this role because they are
outside its allowed edit surface.

Later command: `.venv/bin/python -m unittest tests.test_research_expansion
tests.test_research_telemetry`. Observed failures include blank review/unallocated
reason validation, Benter-only graph lane classification, legacy recipe-hash
null-field parity, incomplete-identity champion grouping and common-population
comparison of different feature datasets. Errors include missing `search` and
`v6_proposal` fixtures on `PackageIndexAlignmentTests`. This is an in-progress
working-tree observation, not a verdict on the eventual implementation. Those
files are outside this role's ownership; no edits were made to them. Resolve and
rerun before launch; the earlier 334-test pass does not supersede these failures.

## Parent Handoff: Exact Current Failures

Latest focused rerun at `2026-10-04T06:48Z`, after concurrent implementation
changes: `.venv/bin/python -m unittest tests.test_research_expansion
tests.test_research_telemetry`, 86 tests in 24.235s, three failures / zero errors,
exit 1. Previous 06:35 run collected 82 tests and four failures in 30.589s.
The `_prepared_fold(... pins=...)` real-CLI canary failure no longer fails;
the original three blockers remain:

| Exact current test ID | Observed assertion | Test line |
|---|---|---|
| `tests.test_research_expansion.RuntimeFixtureTests.test_fixture_planner_respects_one_trial_operator_ceiling` | `item.trial_budget` is 2, expected <= 1 | 705 |
| `tests.test_research_expansion.RuntimeFixtureTests.test_missing_preparation_manifest_schedules_preparation_before_fit` | Expected `_prepare_program` once; called zero times | 694 |
| `tests.test_research_expansion.RuntimeFixtureTests.test_restart_resumes_reserved_attempt_before_asking_replacement_work` | Original `attempt-91b8fc59ecb2aa8fe45286bbefb2f1f7d44f887345fc277d215cc76abdbc9aee` absent; replacement `attempt-ce941d8000fe25fcb29ae6e81d46ae283a058eaa0bd8c53a4d1ef4b4ea7c8f1b` completed | 652 |

No claim is made that the whole suite or launch gates pass. This role did not
edit implementation/test files. The latest table is the active parent handoff;
historical details below are preserved to distinguish fixes from missing proof.

### Earlier Snapshot

Read-only test capture started `2026-10-04T06:10:24.344472Z`:
`tests.test_research_expansion` and `tests.test_research_telemetry`, 60 tests in
0.179s, nine failures / one error, exit 1. A unittest runner collected exact
failure IDs and tracebacks; fixtures execute in their local temporary directories.
The earlier eight-failure/five-error snapshot is historical. Missing test-fixture
attributes no longer failed in this refresh. Owners must rerun after their changes.

Every test prefix below is `tests.test_research_expansion.` except the two marked
`tests.test_research_telemetry.`. Lines describe this working-tree snapshot and
may move during simultaneous edits.

| Exact test suffix / module | Result and observed assertion | Test line |
|---|---|---|
| `DecisionApplicationTests.test_invalid_fixed_control_extension_does_not_partially_apply_new_program` | Failure: unexpected program `980ce35dd584cd53` remains after rejected extension | 391 |
| `MeetingBoundaryTests.test_variable_meeting_sizes_do_not_score_the_same_race_twice` | Failure: eight scored entries / six unique race IDs | 513 |
| `PackageIndexAlignmentTests.test_invalid_probabilities_cannot_pass_vacuously_on_nonoverlapping_index` | Failure: expected ValueError not raised | 541 |
| `PlannerDecisionTests.test_no_op_requires_nonblank_review_reason (reason='   ')` | Failure: expected ValueError not raised | 80 |
| `PlannerDecisionTests.test_positive_remainder_requires_reason (reason='   ')` | Failure: expected ValueError not raised | 121 |
| `PortfolioCompatibilityTests.test_benter_only_graph_preserves_benter_lane` | Failure: actual `experimental`, expected `benter` | 280 |
| `PortfolioCompatibilityTests.test_legacy_recipe_hash_keeps_pre_expansion_null_field_semantics` | Failure: actual `3e56d8c782e0f515`, expected `ce66143e3edf472c` | 303 |
| `tests.test_research_telemetry.ChampionTests.test_missing_protocol_identity_does_not_group_unrelated_attempts` | Failure: one grouped champion, expected two | 170 |
| `tests.test_research_telemetry.ComparisonKeyTests.test_different_feature_datasets_can_compare_on_same_score_population` | Failure: comparison keys differ for `dataset-1` and `new-features-same-runners` on the same evaluation population | 93 |
| `PackageIndexAlignmentTests.test_valid_race_probabilities_do_not_depend_on_dataframe_index` | Error: ValueError `Packaged probabilities must sum to one by race`, raised at `ima/research_model_package.py:148` for valid `[0.2, 0.8, 0.3, 0.7]` | 537 |

Deployment remains held. This role has no callable parent-agent message channel;
this shared owned report is the concrete parent handoff. No implementation/test
files were modified to resolve these failures.

## Protected V5 Baseline

Parent-supplied observation at `2026-10-04T05:48:00Z`: cycle 123,
432 complete, two failed, eight running, tracking pending zero.
`memory.current=90095411200`, `memory.peak=103083323392`,
`MemoryMax=107374182400` bytes. These are historical samples, not V6 results.

Read-only refresh at `2026-10-04T05:52:37.745783Z`: status `training`,
cycle 123, ledger 434 completed / two failed / eight running,
zero pending tells, zero pending tracking and no tracking errors. A separate
SQLite connection opened with `mode=ro` confirmed the attempt status counts.
Actual scheduler snapshot: eight reserved CPU threads and 32 GiB reserved RAM,
eight workers, 24-thread CPU budget and 80 GiB RAM budget. Source and scheduler
agree: one native thread / 4 GiB reservation per job. Reservation is not RSS.

| Protected identity | Value |
|---|---|
| Own account | `imaopt`, uid 1001 |
| Host | `100.95.24.121` over Tailscale |
| Campaign | `/home/imaopt/research-v2/campaigns/agentic_v5_discovery` |
| Release alias | `/home/imaopt/research-v2/live-releases/ima-v5-current` |
| Alias target | `/home/imaopt/research-v2/live-releases/ima-v5-0915b61` |
| Revision | `0915b619f6a40dcc390c3044107831293f826660` |
| Dataset SHA256 | `8f33d989cb258ecb45f16b49363e5a1b791b95a8ca9a40b8e22db7a8a64feb6c` |
| Protocol identity hash | `121ee03b31fd583072df255581aa21e530c4ff55ebfd82a88382f81519363ff8` |
| Environment identity | `5ee4ca2ef9503545b5cb5c15` |
| Portfolio | `feature-discovery-v5-fundamental` |
| Metric contract | `protected-development-v3-fundamental-v1` |
| Target contract | `research-targets-v3` |
| Service PID / restarts at refresh | 792999 / 1 |
| Effective high / max bytes | 103079215104 / 107374182400 |
| MLflow | `http://100.95.24.121:5000`, `/health` 200 OK, `/version` 200 `3.16.1` |
| MLflow experiment search | API 200, experiment `6`, `ima-agentic-v5-discovery` |

Later read-only cgroup sample: current 95,895,584,768 bytes, peak
103,083,323,392. Current-instance `memory.events`: high 787, max 0, oom 0,
oom_kill 0. Memory-pressure avg10/60/300 zero, cumulative some/full stalls
9,327,190 / 9,327,020 microseconds. Do not erase historical OOM incidents or infer
safe V6 capacity from these counters. Disk sample: 756,200,480,768 bytes available
on the own deployment filesystem. Dynamic counts, PID and memory can advance;
immutable identity and unit definitions are the protection comparisons.

## Protection Digests

SHA256 below refers to exact file bytes, unlike the canonical protocol identity
hash above. Read these files before and after any parent cutover; preserve them.

| File under own account | SHA256 |
|---|---|
| `research-v2/campaigns/agentic_v5_discovery/campaign-identity.json` | `1cde1d418a741dcf2c6c2d4fdea6ae14d681f8e1b4d2b03b7d4e024363df7e2e` |
| `research-v2/campaigns/agentic_v5_discovery/inputs/protocol.json` | `ba9e8bd670c3d18ee90d9b3ce006bc0254017c018b5248a8ff3483c9276bb4ea` |
| `research-v2/live-releases/ima-v5-current/deploy/systemd/ima-discovery-v5-supervisor` | `7a7d6f88a6d99da19b1e96097a0e9d57067d53edec96219b0e61cc4ad6d43f0e` |
| `.config/systemd/user/ima-discovery-v5-supervisor.service` | `ae05a36667cdafd7c4de2da5254981a6d36017c8cec3e049348d8189828bbf4b` |
| `.config/systemd/user/ima-hkjc-acquisition.service` | `4c62251ec90b498746b8e7f183b9e3886493b3e17ced06e9f465612604fa869d` |
| `.config/systemd/user/ima-hkjc-snapshot.service` | `ac86ca879d5ed6957f2a96901d1e7062fc7b0f027d63b8076b6b366090f60c6c` |
| `.config/systemd/user/ima-hkjc-coverage.service` | `1bb7904d2a4a4ae582099ba5d7003245eaa87efe7f776336fe7daa5bbfe832d5` |

Acquisition was active/running, coverage was activating and snapshot was inactive
at the initial refresh. Timer transitions are expected; compare definitions and
record any intentional own-user change. No other users' repositories or services
were inspected. Cortex/Solar health is therefore unverified by this role, and no
claim of a before/after end-to-end check of those systems is made.

Final role readback at `2026-10-04T06:04:02.217566Z`: all seven file digests above
matched, release alias target unchanged, V5 still `training`, cycle 123,
434 completed / two failed / eight running, zero pending tells/tracking. This
proves protected own-user files stayed unchanged during this role's audit; it
does not certify a later parent cutover.

Additional read-only protection baselines, captured after that comparison:

| Own V5 file | SHA256 |
|---|---|
| `campaigns/agentic_v5_discovery/ops/openrouter-config.json` | `4ca079b6c12faa6944d4fffe32e64b98a5722c5adf5769bd1f01cd8d730f353c` |
| `live-releases/ima-v5-current/ima/research_v5.py` | `52d872b85167358c376ce2bed8325ef563aa676927098f683bfb5a2fb8b3f1aa` |
| `live-releases/ima-v5-current/scripts/optimize.py` | `422263ddef8d159d142c2baf1c075cf6f56781e51a92dac1a620914bba248919` |

Paths in this additional table are relative to `research-v2`. Config bytes were
hashed without displaying its contents or credentials. Use these extra digests
for the parent's next before/after cutover readback.

## Dependencies And Deployment

### Immutable Transfer Preparation And Tests

Parent release SHA is not assigned/ready. No dirty source archive, source upload,
remote extraction, V5 STOP or service mutation was performed. Supervisor success
receipt now includes the exact `committed_tree` mode/blob map derived from
`git ls-tree`. Runbook packaging uses the verifier FROM that committed SHA,
not an uncommitted workspace launcher; SHA/receipt arguments are validated and
the future receiver independently verifies hashes, PAX, blobs, executable modes,
file set and post-extraction content before read-only release metadata.
Target read-only `git --version` returned 2.53.0; no target Git repository is
needed. Transfer shell `bash -n`, supervisor syntax and diff whitespace pass.

Local mechanics test uses committed baseline
`6ef7481dc93f406e77174d510b36d99155ef38b4`: 1,515,520-byte archive, 178 committed
files. The actual supervisor reports source integrity passed but required V6
readiness failed (15 missing paths). First baseline attempt including newer
uncommitted doc pathspecs failed Git archive with exit 128; test scope was then
limited to existing COMMITTED entries, never filled from the workspace.
Receiver tests use an explicitly synthetic readiness flag only to exercise
transfer/extraction, with a canonical temporary root substituted for the remote
root. This does NOT certify a V6-ready archive or permit transfer.

Final mechanics result: one valid local extraction/read-only metadata fixture
passed; 11 rejected fixtures: bad receipt SHA, archive SHA, existing destination,
failed readiness, PAX revision, blob content, executable mode, symlink, traversal,
omitted file and duplicate file. Rejections occurred before destination creation
(existing destination sentinel preserved). Fixture artifact/test-results path:
`/private/var/folders/wq/tn0d0r417cb3kghxltyqhjt80000gn/T/ima-v6-transfer-test-bkjr36wu`.
Initial harness argument-count/macOS `/var` symlink-path errors were corrected
without weakening the receiver. Positive complete-V6 committed package and
actual Tailscale upload/extraction remain pending the parent's SHA/assignment.

### Guarded Build: Read-Only Feasibility

At `2026-10-04T07:23:58.863161+00:00`, own user manager was running, systemd
259.5, CPU/memory/pids delegated. Host physical memory 130,391,482,368 bytes,
MemAvailable 29,046,710,272 (includes reclaimable cache), MemFree 5,423,894,528;
28 CPUs/load averages 9.64/9.84/9.45. V5 current 98,719,813,632, peak
103,083,323,392, max 107,374,182,400. V5 remains active PID 792999/invocation
`0e74db4db0fa4f05bf8a308e200b022c`. Own manager ancestors' memory/cpu maxima
are unlimited, so no observed ancestor ceiling prevents a sibling build unit.
Historical manager-ancestor OOM/kill counters 16/7 are nonzero; host full memory
pressure avg10/60 was 0.06/0.08. No transient unit was created as a probe.

Runbook proposed limits are MemoryHigh 6,000,000,000 / MemoryMax 8,000,000,000
decimal bytes, swap 0, CPUQuota 200%, native threads 1, runtime 3600 seconds.
Admission reserves another 8,000,000,000 host bytes plus possible V5 growth
to its own max. At the initial sample the required available total was
24,654,368,768, leaving 4,392,341,504 above that allowance. Thus provisionally
feasible, NOT proof a full build fits 8 GB or is safe later.

Exact documented guard was subsequently executed read-only over SSH: two samples
five seconds apart, exit 0. MemAvailable 30,869,491,712 / 30,692,278,272;
required 24,742,957,056 / 24,740,732,928; V5 current 98,631,225,344 /
98,633,449,472; host full-pressure avg10 0/0. Both were admissible at sampling.
Require a fresh two-sample pass immediately before any assigned build.
Local `scripts.build_research_dataset --help` passed; exact request/registry/
source-snapshot/raw-manifest flags are in the runbook. Committed source,
approved request/output and Linux import/readiness proof remain prerequisites.
Actual transient-unit creation/limits enforcement and official build exit,
peak, immutable output/coverage/protocol validation remain UNTESTED. Neither
V5 mutation nor campaign launch is authorized by this feasibility result.

### Authorized Private Setup Receipt

Completed `2026-10-04T06:49:26.590593+00:00`, only under
`/home/imaopt/research-v2/v6-dependencies`. No sudo, V5/acquisition modification,
direct server download, source upload, STOP or service action. Actual changes:
fresh `venv/` generated with own Python 3.12.12 `--without-pip`; 27,197 base
files and 3,990 overlay files copied without hardlinks/old scripts; generated
`preserve.constraints`; locally downloaded `wheels/` (95 files, 59,588,314 bytes)
transferred with `scp` over Tailscale; private pip installed only JupyterLab
4.6.2/Graphviz 0.21/Plotly 7.1.0 plus missing closure. Target no-index dry-run,
install and pip check all exited 0. All original 120 effective versions match;
189 effective distributions means 69 additions, not upgrades. Pydantic remains
2.13.5/core 2.46.5: approved copy mode, explicitly not research-lock parity.

Copy integrity verified per-file SHA256 before repair: base ordered path/content
hash `cfae0f9b9772cb39732624ea113a4f15ddb55844ca3888218c29cbc491f49837`,
overlay `915f228d12b79d8a9e769e305a04c7a6381da6e51c5d867ba4cdc8c0c1099c1f`.
All 25 tested research/features/repair/native module origins and all effective
metadata paths are private; prefix is private and sys.path has no V5/acquisition
package directories. Executable resolves to the unchanged own managed Python.
Setuptools/Woodwork emitted its existing deprecated `pkg_resources` warning;
no upgrade was made. Ten protected hashes match; V5 active, PID 792999,
invocation `0e74db4db0fa4f05bf8a308e200b022c`, STOP absent.

Exact private content inventory: `evidence/environment-files.json`, 46,947
file/link entries, 1,466,981,855 regular bytes across venv/overlay/wheels and
constraint file. Generated cache and evidence receipts are separately scoped
inside this root; nothing was written to another remote path.

| Evidence under private `evidence/` | SHA256 |
|---|---|
| `setup-receipt.json` | `3a108b4002b5c8f17dce43912f15e09ac41dae2bd6e78b8bda76185c3fb90ed6` |
| `environment-files.json` | `2dd0482463a56e33c84dc11fe2174469a5cc2451efaa7c0ecd4bad0d9311ab4c` |
| `resolved-effective.json` | `11d9d339cc1903a37763eb0aed7d52ea422507db284464d0f2b31a7ada045092` |
| `resolved-effective.lock` | `87794eb50cc8e4d39c37c505632177889be470f7ed8a26ae3b46a47a151cfc19` |
| `import-origins.json` | `0c7c4570931fcc7c3b77ce2f2aa7d7dedd5cf6a273ccef7c1d557babe654d811` |
| `wheel-manifest.json` | `d91a28c1fa3c8ed768c2366f6097639f4b16662940eacdcf00e4ff8dddc4c33f` |

`copy-integrity.json`, baseline/post protection JSON, offline dry-run/install
reports/logs, `pip-check.txt`, `pip-freeze.txt` and baseline effective JSON also
exist; their checksums are indexed by `setup-receipt.json`. This is a ready
private dependency base, not full acquisition/browser extras, code deployment,
replay on official races or a running V6 service. Preserve this resolved identity
for later committed-release canary gates; do not silently apply partial locks.
Local receipt readback at the runbook's artifact root verified all 15 indexed
checksums and the setup receipt SHA256. Bash syntax, CI YAML/read-only permissions
and decimal service limits passed after these updates. No exec sessions from
dependency setup remain running.
After compaction, read-only remote refresh again returned `No broken requirements
found.` with exit 0. Existing `setup-receipt.json` SHA256 remains
`3a108b4002b5c8f17dce43912f15e09ac41dae2bd6e78b8bda76185c3fb90ed6`;
`pip-check.txt` receipt SHA256 remains
`9261363b733079a641c2e4cc9bc46ffa1d8336945a87f807b6cf68847dbc9b09`.
Refresh created no remote files or source/service mutation.

### Parent Fixture Readback: Not Launch Acceptance

Parent reported new canary CLI exit 0 and two workers. This role read
`.tmp/v6-canary-integration-3/campaign/status.json`: update
`2026-10-04T06:44:59.212117+00:00`, mode complete, three cycles/decisions,
10 completed, zero pending tells, 10 pending tracking (expected offline).
Program completed counts total eight Benter/two boosted. No advanced generated
feature, paid planner, official dataset, uploader/UI or live recovery proof is
implied. Parent separately reported 32 legacy-controller tests passing after the
cache helper fix; that exact 32-test command was not independently rerun here.
Readback of the updated canary code confirms complete mode/exact count,
zero failed attempts/pending tells, and zero pending tracking when MLflow is
configured are now enforced. This does not imply offline entries were uploaded.

Subsequent local tracking canary `.tmp/v6-canary-tracking-1` uses its own
`mlflow.db`, not the remote pinned MLflow service. This role read status JSON
and both SQLite databases with URI `mode=ro`: status updated
`2026-10-04T06:52:09.433022+00:00`, complete, five completed, three cycles,
pending tells 0, pending tracking 0. MLflow database has five FINISHED runs,
five READY model versions (all five join to their run IDs), four OK traces.
Status SHA256 `10cc8eb64735da19f4045b5de61138fe588b4ce1ee93f4ebceb1f25105f24fc6`.
This verifies the observed local tracking/registration/trace endpoints and empty
pending outbox, not paid planner costs, native USD UI, publication-safe official
datasets, advanced graph/features, committed Linux execution or live recovery.
V5 remains outside these fixture paths. No live launch is authorized yet;
official dataset/controller gates remain held.

`.tmp/v6-replay-smoke/replay-report.json` readback has two records with Benter
and boosted package manifests, each 120 scored runners, prediction parity true,
prediction/objective deltas 0; parent reports rtol 1e-10. Report SHA256
`6f89b522a5c30edb8fb556373f2b1bc361542af876989be9a8db05caefc4527a`.
Source is `.tmp/v6-canary-integration-2/campaign`, not the pinned live V5.
Replay lineage code/environment is `unknown` and automatic promotion false;
this is fixture parity only, not committed-release or V5 champion replay proof.

### Earlier Read-Only Inventory

Read-only refresh `2026-10-04T06:26:24.957593Z` confirms Python 3.12.12/Linux
x86_64. Effective 120-distribution version/metadata-path manifest SHA256:
`97dd8c60d6edda3180a9ae0a586c6d98a1f6c837f8f0ae2f5570da9608bf5bd9`.
Resolved base purelib under release `2868bec` has 27,197 files / 1,150,969,295
bytes, no links/hooks; V5 overlay has 3,990 files / 28,858,761 bytes, no links,
only the standard setuptools `.pth` shim. The advertised `f54c604` venv shares
that base; release names are not environment isolation.

Actual effective remote `pip check` exit 1 exact findings:

```text
ima-racing 0.1.0 requires jupyterlab, which is not installed.
catboost 1.2.10 requires graphviz, which is not installed.
catboost 1.2.10 requires plotly, which is not installed.
```

Read-only offline `pip install --dry-run --no-cache-dir --no-index` with both
current research/overlay locks and existing research/V5 wheel stores exited 1:
Pydantic 2.13.4 was not available. Its required core is 2.46.4, versus effective
V5 2.13.5/core 2.46.5. Missing acquisition extras are Scrapy/pypdf in the effective
research environment; pypdf was not found in the audited acquisition wheel store.
Playwright is a separate UI gate, not headless runtime. The runbook records exact
future wheel preparation, Tailscale copy and isolated no-index install commands.
Those strict-lock/acquisition commands have NOT run. Subsequent user assignment
instead authorized a fresh private copy of the audited effective environment,
preserving all 120 versions; prefix and metadata-origin checks passed. Its only
permitted repairs are JupyterLab/Graphviz/Plotly closure, downloaded locally and
transferred through Tailscale. No source release upload or service launch.
Local repair candidates JupyterLab 4.6.2, Graphviz 0.21, Plotly 7.1.0 are observed
local versions, not target compatibility receipts. Require a complete Linux
wheel closure, own manifest/hash/import-origin readback and successful private
`pip check` before acceptance. Never repair this drift in live V5.

At `2026-10-04T06:26:26Z` own env/routing refresh still showed public models HTTP
200; OpenRouter addresses 104.18.2.115 and 104.18.3.115 via `tailscale0` table 52,
source 100.95.24.121 uid 1001. Exact own env source is
`/home/imaopt/.config/imaopt/openrouter.env`, mode 0600 uid 1001, key present
without value/length output. No authenticated paid call was made. Parent gates
and explicit setup assignment remain prerequisites; launch is explicitly held.

V5 interpreter: `/home/imaopt/research-v2/releases/f54c604/.venv/bin/python`,
Python 3.12.12. Overlay: `research-v2/v5-dependencies/site-packages`.

| Dependency | Remote installed version |
|---|---|
| NumPy / pandas / SciPy | 2.5.3 / 2.3.3 / 1.18.1 |
| scikit-learn / Joblib | 1.9.1 / 1.6.0 |
| MLflow / Optuna / psutil | 3.16.1 / 4.5.0 / 7.2.2 |
| Pydantic | 2.13.5 |
| CatBoost / LightGBM | 1.2.10 / 4.7.0 |
| Featuretools / feature-engine / Woodwork | 1.31.0 / 1.9.4 / 0.31.0 |
| httpx | 0.28.1 |

Pinned V5 `requirements-research.lock` SHA256:
`2c961dae392342503f3ab7053e0f27e21160d7beafe9dac5b38667332fcec310`;
overlay lock SHA256:
`1de397463e2dd4d3a827a20ac613bc97501394195bd57f8ad980920b60e34126`;
`pyproject.toml` SHA256:
`98dd1654d763e75daa70a336d3496bc751b3a57b2b25e36e5b105c7732401523`.
Existing research lock pins Pydantic 2.13.4; live V5 reports 2.13.5. Record this
drift; do not fix it inside V5. Existing lock files are partial and do not lock
all transitive dependencies. CI installs their explicit pins plus existing extras
and prints resolved versions. A V6 release still needs its own resolved manifest,
`pip check`, environment identity and replay evidence. No blanket upgrades or new
distributed scheduler are needed for this role.

## Dataset And Protocol Comparability

Read-only official pointer:
`/home/imaopt/acquisition/hkjc-20261002/snapshots/latest_verified.json`.
At refresh it pointed to `snapshot-20261004T050721Z-122062e8`, verified at
`2026-10-04T05:25:04.142934Z`, parser `official-corpus-v14`:
14,488 races / 178,718 runners; 173,884 previous baseline runner keys retained.
Input digest `84f6b3eb8c2315eda49f2d41d4acb0df6e127f4d3eeef3bc4fe603d5beb9ecb6`
identifies acquisition inputs at job start; immutable dataset identity is its
manifest and lineage. Latest is a pointer, never a campaign identity.
Independent readback reports 14,471 feature races and 138 attrition rows;
raw retention does not imply every raw race enters training. Its verdict is
sampled roster readback, not exhaustive website coverage or a temporal/leakage audit.

V5 protocol is `min_train_races=17000`, `calibration_races=1000`,
`score_races=1000`, `max_folds=3`. Even the raw 14,488-race count cannot provide
the minimum training block; the first complete scoring fold needs 19,000 races.
No silent minimum reduction, replay claim or cross-contract score promotion is
valid. Freeze a separate V6 protocol against eligible whole-race counts and date
boundaries; report exclusions, coverage and common race intersection. Preserve
original V5 champion replay on its original dataset/protocol/environment. Label
retraining or scoring controls on the V6/common population as new comparisons.

## Remaining Launch Gates

| Requirement | Required concrete proof before acceptance | This role's result |
|---|---|---|
| Events and speed sources | Publication/coverage-aware joins; race/workout/trial units; whole-race quality audit | Not independently verified |
| Dataset lifecycle | Request -> validated immutable artifact; failed build cannot promote; baseline retention | Official baseline captured; V6 lifecycle unverified |
| Features and selection | Novel formula reuse; >32 eligible selected; trained-only selectors; leakage rejection | Unverified |
| Graphs and performance models | Chronological OOF, controls, scale/identification, valid joint outcomes and packaged replay | Unverified |
| Controller and budget | Planning during busy fits; continuous refill; idempotent restart and retirement | Legacy scheduler observed; V6 unverified |
| MLflow | Paid plan and linked snapshot; native USD field API/UI; one physical call charged once; current comparable champions | Health/experiment API only |
| Paper EV and Kelly | Quote modes, source-backed units, exact settlement, joint portfolio/solver/no-lookahead controls | Unverified; no bets submitted |
| Performance | Equal concurrency cold/warm repeated full-history benchmark, cgroup peak/PSS/USS/I/O/pagefaults | V5 resource baseline only |
| Recovery | Builder/controller/worker/uploader restart drills; no duplicate spend/allocation/registration | Not run |
| Deployment | Exact release/data/protocol/environment identities; multiple real decisions and completed graph; protection readback | Parent-owned, pending |

For each gate record command, revision, UTC timestamps, exit code, artifact path
and readback. Fixture success cannot certify historical coverage, native USD UI,
full-history throughput or live recovery. Missing historical publication evidence
and pre-race exotic quotes remain source limitations when observed; unavailable
controller paths or failed tests remain implementation gaps. Keep those separate.

### Parent Integration Checkpoint 2026-10-04

The committed implementation passed 629 local tests in 121.491 seconds after the
final dead-heat quarantine guard. The preceding full pass was 628 tests; earlier
failed runs exposed real restart, publication, identity, JSON null and taint
propagation faults and are not release evidence. Focused actual executor/data
integration also passed 39 tests. Registered feature definitions build verified
successor datasets and replay from packages; these remain provenance fixtures,
not proof that the complete official corpus has been built or trained.

An actual read of the latest official runners file found 31 multiwinner races
and zero missing-winner races. The current conditional-logit contract rejects
multiple winners, so the registry now explicitly excludes complete dead-heat
races and retains original evidence upstream. All publication, quality,
population and confirmation exclusions must be read back after the full build.
The registry protects confirmation keys before filtering, binds numeric and
categorical eligibility to dataset identity, and records unsupported delayed
outcome publication rather than inventing historical availability.

Paid API receipts are authoritative for cost, including rejected/repair calls;
unknown cost freezes further paid planning. Queue-aware native-thread allocation
avoids reserving four cores for every fit when a large trial queue is ready.
Actual host/cgroup admission and progressive capacity still govern concurrency.
No V5 stop, V6 launch, real betting, or other-user modification has occurred at
this checkpoint. Remote Linux execution, actual official dataset readback,
champion replay and paid planner/trainer/MLflow feedback remain rollout gates.
