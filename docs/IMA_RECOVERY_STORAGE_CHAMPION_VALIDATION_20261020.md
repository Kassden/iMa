# Recovery And Champion Implementation Record

Date: 2026-10-20. Task: `ima-recovery-storage-champion-20261020`.

## Current Status: 2026-10-28

**Implementation and qualification continue; the full plan is not complete.**
Obligations 1.1, 2.1, 3.1 and 3.2 are qualified. The first reclaim removed 320 selected
derived payloads and freed 55,674,880,000 measured bytes; twelve held payloads
remain intact. Twenty empty temporary entries have since been moved into
receipt-backed quarantine. The initial shared inventory registered 76 valid
caches; new qualification caches are accounted for automatically as they grow.
The pinned retained-cache policy and bounded control-intent store are installed.

Actual-champion reproduction completed under sealed code `6e66a31`, using the
saved recipe, original dataset and original protocol. Its fundamental race log
loss was **2.1494157522885775**, compared with the saved **2.149415373890447**
(difference +0.000000378398130607). It took 263.742 seconds and peaked at
4,843,667,456 private bytes. This is close reproduction, not an improvement or
independent confirmation. The saved champion package remains unchanged.

An integrated native-controller recovery check preserved five accepted fits
through an injected storage interruption and completed five more on resume,
without redispatching an accepted fit. Actual MLflow SDK readback in a private
qualification store found ten finished runs and twelve traces, including two
error traces retaining the injected storage and scratch-configuration failures.
Tracking, optimizer-tell and control-intent queues drained to zero. A separate
active-process probe verified inherited policy, private scratch, and the kernel
aggregate memory ceiling of 99,999,997,952 bytes (page-rounded below 100 GB).
The cross-program refill test also passed, with budgets independent of capacity.

The scratch failure exposed a startup configuration requirement: launch now
requires a canonical own-user private TMPDIR on the admitted filesystem before
SDK initialization. Normal research, feature exploration and acquisition remain
held pending the complete restart gates. OpenRouter authentication succeeded
through Tailscale. The initial zero-credit blocker was resolved by a top-up;
a subsequent read-only check reported US$16.784208275 available. A bounded
real-data controller qualification uses DeepSeek Flash, a US$1 planner cap,
four completed fits maximum and two concurrent fit slots. This qualification
does not replace the unlimited-service handover gate.

Its first launcher mistakenly used loopback for MLflow, which listens on the
Tailscale address. Readback established zero decisions, zero attempts and no
evidence file before that qualification unit alone was stopped. Records were
preserved. The successor uses the healthy Tailscale endpoint and sealed code
`dded2d0`; normal services and old campaigns were not restarted.

The independent phase-3 review supports original-format writer admission;
bounded native Scrapy/Queuelib, SQLite/Optuna and cache fixtures cover the
noncontroller paths. These canaries do not claim a physically full production
filesystem test. The qualification joins 83 hash-bound inputs in each phase-3
outcome. Other cache classes and objects-v2 remain separate qualifications.

The additive object-lifecycle foundation passed 29 tests across archive,
snapshot and dataset consumers. Review findings for symlinked shards and
temporary-name collisions were fixed and tested. The planner holds every
object while MLflow/history/pending reference adapters remain incomplete; no
production shared-object deletion or external archive is qualified.

The event-evidence audit now separates schema-default false identity flags
from evidence of rejection. A live readback found 104,723 observations, 77,397
full horse IDs and 27,326 unresolved IDs. All observations have source links,
stored body hashes and capture timestamps; none has a verified publication
timestamp. Stored hash presence does not mean original response bodies were
rehash-verified in this audit. Unknown publication is not backdated.

The dated sections below retain the incident history, including superseded
access blockers. Read the later checkpoints and `.mega` task outcomes for
current acceptance; old checkpoint tables are not a fresh live-state claim.

## Completed Component Study: 2026-10-28

Three fixed controls used the champion's unchanged dataset, outer protocol and
scored population. Independent readback verified supervisor nonces, result
hashes, exact archived source files and paired reports. The original saved
champion package remains unchanged.

| Pipeline | Stored development objective | Paired race-mean delta against reproduced champion |
|---|---:|---:|
| Reproduced 35% Benter / 65% boosted | 2.1494157523 | Reference |
| Pure Benter | 2.1667156197 | +0.0172804659 |
| Pure boosted | 2.1550922950 | +0.0056875961 |
| Equal-weight pool | 2.1501587033 | +0.0007385202 |

Positive deltas are worse. Both pure controls were worse in the exploratory
race bootstrap and meeting sensitivity checks; the equal-weight result was
inconclusive. Stored objectives average fold metrics, while paired reports use
the full race population, so their aggregation differs slightly. These are
adaptively selected development comparisons, not independent confirmation.

All fits completed before the harness's final stdout serialization failed
(`json.dumps` received an unsupported `flush` argument). The result file was
already durable. This harness error is preserved separately from accepted model
outcomes. Its bundled source inventory included 171 entries outside the archive;
those entries are unqualified. Independent verification instead used exact Git
archive members, all of which matched the running release.

A separate learned-pool qualification is fitting simplex weights on inner
forward prediction blocks, with later outer evaluation. It uses explicit
`development_selected` lineage; nested model-selection independence remains
unqualified. No score or improvement is claimed before accepted readback.

## Latest Offline Qualification

The complete committed suite at `48993ee` passed 2,080 tests and 792 subtests,
with seven local-evidence/data skips. Subsequent exact-index tests qualified the
transformed graph-transfer correction (20 tests), study seals and existing
confirmation consumers (30 tests), and lifecycle admission/collision protection
(31 tests). These overlapping sets must not be summed as unique coverage.

Graph transfers use eligibility-filtered feature views, preserve fixed registered
transforms, and route a bundle only to its declared estimator. The test completes
a transformed graph reference, matched control/candidate, null and leave-one-out
fits, with bit-exact original/control predictions. Adaptive discovery and unfrozen
definitions remain rejected for fixed transfers.

Study seals bind supplied exposure inventories and immutable prospective locks,
wait for labels and exercise actual later evaluation. Scientific identifiers
remain explicitly caller asserted; this foundation does not establish complete
historical exposure or training independence. Phases 4 through 9 remain open
unless their separate task outcomes qualify the requested full scope.

## Historical Verdict: 2026-10-20

**Not complete. Live reclaim and rollout are blocked.** This execution record
does not replace or reduce the canonical nine-phase, sixteen-obligation plan.
Its SHA256 remains
`b0384addf49223492c75d6d5319e3efaf597511007924d60036a6dbd0d9013ae`.

Implementation is isolated in branch `feat/ima-storage-champion`, worktree
`/Users/milkingthesun/Projects/iMa-storage-champion`, based on `ed81321`.
The original dirty checkout was not reset, stashed or overwritten. Necessary
existing dirty planner/resource/discipline/confirmation prerequisites were
carried into this worktree; they are not all newly authored in this execution.
No release was deployed, no new campaign started, and no cache payload or
historical snapshot deleted. Protected other users, Cortex and Solar were not
modified.

## Live Recovery Evidence

At `2026-10-20T04:26:11Z`, Tailscale SSH succeeded as imaopt, UID 1001. The
filesystem had **zero user-available bytes**, despite 41,841,614,848 free bytes
unavailable to that account. Free blocks are not usable admission capacity.

The named own-user feature, snapshot, coverage and billing-reconcile writers or
timers remain inactive/failed. Main V7 was already failed. MLflow was left
running. No root reserved-block or system-wide service change was made.

The feature unit had 11,483 restart retries; the recorded immediate exception
was `RuntimeError: Not an own-user cgroup`. Its budget helper expected a live
main cgroup when the main unit had no process. The local fix handles a strictly
inactive main unit and budgets 92 GB future main plus 8 GB feature work, not
another 100 GB. Startup serialization and retry/backoff remain unqualified.

Own process maps for systemd, sd-pam and the SSH transport were unreadable.
`sudo -n -l` returned `user imaopt may not run sudo on cortex-server`.
No process-name exemption was used. Safe selected reclaim requires complete
privileged mapping/quiescence evidence for the exact payloads, or an equally
complete alternative proof. This is not authority to touch other repositories.

Local evidence: `.mega/live-evidence/preflight.json`, `blocker-current.json`,
`package-before.json`, database-capture/readback JSON and `fitness-availability.json`.
Backup data and these operational receipts are not source-code deliverables.

## Preserved Databases

All three healthy local copies passed integrity and application readback:

| Copy | Readback | SHA256 |
|---|---|---|
| Ledger | 1,035 attempts: 974 completed, 60 failed, one stale running | `b40b73a9aa7cb852c0e9f13a1ed39968939cea546ee51c0751dbbf7b9c2d9012` |
| Decisions | 255 decisions | `76781f44f418ea7669e06b7a72c79a955550a3d21e14744190032f252b2f13cc` |
| MLflow | 26,739 runs; 7,449 model versions; 17 experiments | `1ca4ba072a06fa0f212197f4eeb406c7214df9ae53e5d1e01bd04588521afd17` |

MLflow's verified copy is 4,693,680,128 bytes. The first timed-out transfer is
retained separately as `.incomplete`; it is not a valid backup.

Ledger/MLflow used SQLite online backup. Decisions had a three-byte SHM and
failed normal readback. A stopped-writer, whole-file-locked DB/WAL/SHM capture
preserved that triad; only the separate healthy copy reconstructed SHM. No
original database was repaired/replaced. Healthy SQLite readers can update
ordinary source SHM bookkeeping, so byte-immutable sidecars are not claimed.
Original damaged SHM SHA256:
`709e80c88487a2411e1ee4dfb9f22a861492d20c4765150c0c794abd70f8147c`.

Copies were staged in owned `/dev/shm/ima-recovery-20261020-1001` and pulled over
Tailscale to healthy local storage. RAM staging does not reclaim root disk.

## Database Restore Runbook

This is a recovery procedure, **not authorization to restore now**. Healthy
ledger and MLflow originals do not need replacement merely because backups
exist. Decisions needs a separately verified healthy operational destination
before its writer can resume. No original triad is removed or repaired in place.

Authoritative frozen recovery inputs are the local verified files under
`.mega/live-evidence/database-backups/`, with the hashes and application counts
in the table above, not the `.incomplete` transfer or an unrelated WAL:

| Input | Original Operational Destination |
|---|---|
| `ledger.sqlite` | `/home/imaopt/research-v2/campaigns/agentic_v7_openrouter_recovery_20261007/ledger.sqlite` |
| `decisions.sqlite` | `/home/imaopt/research-v2/campaigns/agentic_v7_openrouter_recovery_20261007/decisions.sqlite` |
| `mlflow.sqlite` | `/home/imaopt/iMa/artifacts/mlflow/mlflow.db` |

1. Recheck own-user identity, units, PIDs, open descriptors, byte/inode headroom
   and the current database generation. Account for any newly committed writes:
   never overwrite a newer valid database with this frozen older backup. Hold
   the affected writer and its timers; preserve productive fits by draining.
   If an MLflow restore is actually necessary, stop its own service and all iMa
   uploaders first. Do not stop other users' services.
2. Preserve the current matching original DB/WAL/SHM with a consistent online
   backup or a verified quiescent, locked triad. Record hashes/ownership/paths.
   The original damaged decisions triad is also copied to healthy local storage
   at `database-backups/decisions-original-triad/`, with every member matching
   its capture receipt. Do not mix sidecars from different captures.
3. Only after storage admission/cache qualification and sufficient writable
   headroom, create a private, own-user recovery destination beneath
   `/home/imaopt/research-v2/recovery/`. Transfer the chosen frozen healthy copy
   over Tailscale into a new stage; refuse existing files/symlinks. Check exact
   content hash, SQLite integrity and application counts before publication.
   Never use root reserved blocks or place a candidate in another user's tree.
4. Keep original operational files untouched. Prefer an explicitly reviewed own
   unit/config switch to a new healthy database/campaign directory. A main
   campaign switch must carry the complete consistent operational state, not
   just a ledger stripped of decisions, hypotheses, search or artifact links.
   For MLflow, retain its referenced artifact store and validate stored URIs;
   the SQLite backup is not a backup of all external model artifacts.
5. Inspect restored attempt/decision identities, outbox states, pending tells,
   registered models, artifact paths and the actual champion package. Pending
   uploads/calls are not blindly replayed: reconcile durable IDs first to avoid
   duplicate runs or paid requests. No unqualified archived object is assumed
   available simply because the database row exists.
6. Bind any changed scientific code to a sealed new release/campaign identity.
   Resume only after all-writer admission and bounded-cache gates pass. Start
   the one affected own service, verify its actual imported files/PID/cgroup,
   database destination, readback and headroom, then observe multiple real
   cycles before enabling related timers. MLflow restore requires native
   run/model/artifact readback in addition to HTTP health.
7. On failure, hold the affected own writer again and preserve the failed stage
   and new forensic evidence. Roll back the unit/config only if the prior
   database remains healthy and no newer valid commits would be lost. Otherwise
   leave that writer stopped; never restart against a known-damaged original.

Current restore gate: **held**, because available storage is zero and core
storage qualification is unfinished. No live restore/config switch was applied.

## Supplemental Baseline Inspection

At 08:13 UTC the actual saved champion package was replayed again with identical
package/source hashes and all fourteen probabilities. Own process mapping
inspection was still incomplete, sudo was still denied, disk availability was
still zero, and named writers remained held. See `package-refreshed.json` and
the refreshed `blocker-current.json`.

`baseline-supplement.json` records 111 protected boundaries with ownership,
device/inode, resolved path or explicit absent/unfollowed-symlink state. Every
account path is held recursively by default; those observations do not approve
any deletion exception. Current campaign, dataset, trial, source, release and
MLflow artifact roots are included, not only the selected package.

The ledger has zero pending model uploads and zero pending Optuna tells. Billing
has four unresolved decisions: D000236, D000239, D000247 and D000248. Known spend
is USD 14.2744826737; total spend is unknown. The billing service journal proves
ENOSPC while flushing its local reconciliation receipt. Provider-credit
exhaustion is not established by that failure. No provider request was issued
by this inspection and no unresolved cost was replaced with zero.

The trace outbox separately failed direct source readback with `disk I/O error`.
That initially made its backlog unknown; ledger upload counts did not resolve
the trace queue. With the main and feature writers stopped, a matching locked
DB/WAL/SHM capture was preserved and only its healthy copy reconstructed SHM.
The healthy RAM copy passed integrity and showed 5,110 delivered traces and no
pending trace IDs. Its hash is
`77a7ca48bbdc0c0635687dc8d3cf9a7ff726bd31253c47c15ea073f73b1c39e0`.
The first 180-second uncompressed transfer timed out; its partial is not a valid
backup. Compressed Tailscale transfer, local integrity/count readback and every
original-triad content hash subsequently passed. The fourth healthy local copy
is verified; original triad members are preserved as checksum-verified gzip
files in `database-backups/trace-original-triad-compressed/`. Its receipt is
`trace-outbox-capture.json`. The source outbox remains unrepaired; no trace replay
or affected writer restart is allowed yet.

The same restore runbook applies to the trace outbox, whose original destination
is `/home/imaopt/research-v2/campaigns/agentic_v7_openrouter_recovery_20261007/trace-outbox.sqlite`.
Its pending/delivered identity must be preserved and checked before any tracing
consumer is resumed. Recovered delivered traces must not be blindly recreated.

The preserved trace SHM is 4,096 bytes, below the helper's 32,768-byte minimum;
copy-side reconstruction succeeded without modifying that source sidecar. This
supports the diagnosed sidecar failure; it does not imply loss of the 5,110
stored trace records. No claim is made that every disk failure has this cause.

A redundant combined full-integrity scan was cancelled without a passing result.
The final combined readback rechecks all four content hashes and application
counts and reuses the already successful full-integrity checks bound to those
identical hashes. It does not claim that the cancelled scan passed.

## Champion Readback

Fresh best development fundamental race log loss: **2.149415373890447**.
Attempt:
`attempt-09e8faabf8bcc228e8e4dce7d54b628502a894767ff4b54cfa1c90cec1087d9c`.
Saved package: `7ac12999d91540b5`, graph 35% Benter / 65% boosted fundamentals.

The actual saved package loaded and scored a complete eligible post-cutoff
14-runner race. Repeated probabilities were bitwise equal, finite and normalized.
Module paths/package hashes were captured. This is inference proof, not full
historical refit, post-reclaim cache rebuild, independent gain or betting edge.

## Local Changes

- Shared own-user filesystem admission, atomic extension/reconciliation,
  byte/inode headroom and pending cache-growth accounting. Guarded Joblib writes
  extend the shared reservation before excess bytes reach disk.
- Cache hard/soft/target policy of 50/45/40 GiB, stable-key LRU, existing-hit
  accounting and atomic reserved-growth publication. Executor pins protect
  consumers; a stable companion pin covers non-atomic lock downgrade. Neither
  stable lock inode is removed. Legacy cache format remains readable.
- Exact-content/inode/UID selected retirement, explicit authority/canary gates
  and fail-closed own mapping inspection. No actual retirement applied.
- SQLite capture publishes only validated private-stage copies. Symlink
  destinations/ancestors/sidecars reject. Failed stages cannot appear as final
  healthy backups and do not prevent a fresh retry.
- Opt-in immutable partition objects, checksum-verified old/new readers and
  explicit CSV export. No historical migration or object GC enabled.
- Opt-in multilayer parent-plus-delta feature datasets with exact key, label,
  order, unit and availability guards. Registry, lifecycle, executor, campaign
  feature-profile and feature-build tracking consumers support this format.
  Estimates account for decoded numeric widths, not pointer or dictionary size.
- Durable reference-specific merit views and fixed complete-bundle transfer,
  destination, paired/null/ablation mechanisms.
- Explicit boosted-node search and forward-prediction simplex pools. Unsupported
  self-attested nested selection rejects; development selection stays labelled.
- Identical execution snapshots no longer create research traces every five
  minutes. Local status heartbeat is separate from scientific progress.
- Durable known-exposure receipts and terminal `exposure` command, not a full
  independent confirmation implementation.

Admission currently wraps preparation, model execution, feature builds and
tracking snapshot outbox entrypoints. **Not all writers are integrated.**
Acquisition, snapshot/registry, OOF/publication/uploads and runtime paths still
need complete growth/publication handling. Estimates are not a hard filesystem
quota. Storage pressure also needs complete infrastructure retry integration
instead of consuming scientific failure budgets. Normal writers must not resume
with this partial coverage.

## Historical Fitness Audit

The pinned dataset's verified event file contains 104,723 normalized observations:
`dataset-19aaa959e948d6e145ce62c8160413087b46ba3c2578e0de15c452054d9a83db`.
Event SHA256:
`86d0bebb79163b75579c5c85e029a7c11efcf97814f4f7a082e376fb05856900`.

None is marked verified point-in-time; all are observed prospective. All have
`physical_identity_verified=false`. Season starting September 2026 trackwork:
33,596 observations, 6,357 exact horse identities, 27,239 unresolved identities.
These are observation counts, not cutoff-eligible historical predictor coverage.
Do not backdate capture to occurrence. Source review and per-cutoff exclusion
waterfalls remain necessary. No new scraping was attempted on the full disk.

## Review And Tests

Independent review reproduced destination symlink, stale reservation,
unregistered-hit, delta startup, decoded-size, cache publication and pin-handoff
bugs. They were fixed and regression-tested. Final re-review found no residual
failure in these fixes; it independently forced cold/warm downgrade gaps.
Permanent commit/publication and forced-gap tests were then added locally.

A separate review rejected attempted V2 confirmation APIs for incorrect
weighting and incomplete threshold, power, exposure, scheduled-post, coverage,
label-hash and supersession contracts. Those APIs were removed, not qualified by
happy-path tests. Only the exposure-receipt facility was retained and accepted.
It does not enforce quarantine or establish full exposure reconciliation.

Recorded checkpoints: broad relevant regression 402 passed, 2 skipped,
171 subtests; focused implementation suite 95 passed; later storage/delta/
confirmation checks 61 passed, 4 subtests; permanent operator/pin workflow
5 passed. These overlap and must not be summed into a unique test count.
Final current-tree run: **575 passed, 3 skipped, 215 subtests passed** in
229.15 seconds. Source hashes were unchanged across the run. JUnit reports zero
failures/errors. Receipts: `.mega/final-regression.json`, `.xml`, `.stdout` and
`.stderr`; task test event `evt-73b2f3e197ad40c38ad4da3fb19bb4c3`.

Committed local recovery units: `61619af` (storage/cache/recovery), `1507940`
(operator CLI/workflow), `8b7d8e2` (stopped-main budget helper). Other research and
format integration changes remain uncommitted in the isolated worktree. Nothing
was pushed or merged. Scope qualification passed with explicit canonical touch
paths plus recorded carried-prerequisite/actual-consumer extensions.

Owned children were natively closed with actual receipts. Runtime task binding
and prompt/tool invocation were observed; Stop coverage remains unverified.

## Remaining Obligations

| Subphase | Boundary |
|---|---|
| 1.1 | Scope, protected roots, four consistent copies, package replay and restore procedure independently approved; no cleanup/restart qualification implied. |
| 2.1 | BLOCKED: cortex SSH works as of 2026-10-21, but sudo requires interactive authentication; process-map proof remains unqualified and zero bytes reclaimed. |
| 3.1 | Retry and managed publication mechanisms locally tested; all-writer growth/admission, startup policy and independent phase qualification remain open. |
| 3.2 | Local cache/pins/build-once tested; global startup inventory, pressure and live cross-program refill qualification pending. |
| 4.1 | Objects/exports locally tested; normalized-versus-capture trigger and real snapshot qualification pending. |
| 4.2 | Delta consumers locally tested; preparation-wide projection, real package parity and memory/space benchmark pending. |
| 4.3 | Reference GC and verified archive/restore not implemented; archive destination unqualified; history held. |
| 5.1 | Merit/reference mechanisms present; actual next-planner consumption not demonstrated. |
| 5.2 | Transfer mechanism locally tested; real champion/control outcomes pending. |
| 6.1 | Binding/search/controls tested; real attribution and slice studies pending. |
| 6.2 | Pool mechanism tested; real forward pools and nested-selection evidence pending. |
| 7.1 | Metadata audited; official publication/identity recovery and cutoff waterfall pending. |
| 7.2 | Existing fitness/speed regressions passed; clean successor and matched studies pending. |
| 8.1 | Exposure receipts only. Power-planned >=30 meetings, race-weighted meeting-block inference, >=95% coverage, post-time seals, supersession and quarantine unfinished. |
| 9.1 | Local trace/consumer tests; live native usage/cost and matching planner/operator verification pending. |
| 9.2 | No rollout, multiple live cycles or whole-plan qualification. |

No obligation was silently marked complete. Administrator-assisted exact mapping
proof is the immediate live recovery dependency. The canonical reclaim -> all
writer admission -> bounded cache sequence still gates normal restart. Local
development is a checkpoint, not execution of the whole plan.

The whole-plan final gate returned **continue**, not a completion certificate:
only 2.1 has a recorded external blocker; independent unfinished obligations
remain open. Neither passing tests nor this operational blocker is represented
as completion of those remaining local implementations.

## Recovery Retry Follow-Up: 2026-10-21

Local changes preserve a logical trial allocation across typed storage pressure,
including raw ENOSPC/EDQUOT and SQLite FULL when the shared storage authority is
enabled. Non-storage errors retain their original classification. Operational
episodes, deadlines and diagnostics are retained in the SQLite ledger; delayed
allocations remain visible rather than disappearing from pending counts.

Dataset/program/fold operations have separate durable retry identities and do
not create model trials or Optuna tells. Program and fold controller injections
now complete one scientific allocation after a storage retry without retiring
the program or making another planner call. The fold injection exposed a
duplicate backoff timer; storage now uses the operation ledger's timer, while
genuine preparation errors retain their existing circuit delay.

Workers receive the admitted physical nonce explicitly and reject a stale
queued assignment before invoking model code. This does not yet solve managed
result publication: workers still write results and lifecycle feedback before
supervisor acceptance. Nonce-scoped staging, acceptance/restart reconciliation,
all-writer growth coverage and independent qualification remain open. No
normal restart is authorized by these local retry tests.

The first combined proof attempt on this follow-up is invalid: a test source
changed during the run. Its event is not completion evidence. A frozen-source
rerun passed (`evt-b7c67ac3e24348ad80fa9f2e725480f0`). Earlier 575-test results describe the previous source tree,
not these later edits.

The additional publication reviewer performed source review only, made no
changes and confirmed the remaining publication gaps. Its requested new test
file did not exist (pytest exit 4); this is not a passing verification. The
reviewer was natively closed and its actual receipt recorded.

### Subsequent Managed-Publication Implementation

Managed fit workers now use nonce-scoped `.fit-episodes` directories at the
same depth as `trials/<attempt>`, preserving shared preparation cache roots.
The supervisor verifies nonce, finished status and current deadline under its
completion/runtime locks, then publishes the canonical result with a hashed
acceptance receipt. Controller recovery consumes an accepted result before
resubmitting its logical allocation, including the case where result publication
succeeds but a later runtime write fails. Legacy files without acceptance are
not silently treated as accepted results.

Managed lifecycle workers stage feedback intent only. Shared feedback delivery
is controller-owned, accepted-result-checked, retryable and visible in
`pending_feature_feedback`; the scientific result remains unchanged. Direct
unmanaged lifecycle execution retains its previous behavior.

The focused publication suite passed **7 tests**, including actual supervised
training, saved-package loading and probability replay at rtol=1e-10/atol=1e-12.
The fundamental-only test explicitly disables market blending so it compares
the same probability basis. An earlier test compared a selected/blended package
output against fundamental probabilities and failed; that was not claimed as
a model regression or hidden by widening tolerance.

A preceding combined run passed **103 tests and 16 subtests**, but predates
the final publication crash-window adjustment and real-package test. The wider
current-source suite and new independent review are pending. Neither that
count nor a focused passing test completes 3.1 or authorizes deployment.

Local commit `d45be69` contains the separately verified stale queued-worker
nonce guard. Subsequent storage retry and publication integration is uncommitted.
No code has been pushed, merged or deployed in this follow-up.

### Fresh Administrator Evidence

`.mega/live-evidence/admin-access-20261021.json` records successful UID 1000
`cortex` login and membership in `sudo`, but `sudo -n true` fails with
`interactive authentication is required`. Available user bytes are still zero;
free privileged/reserved bytes are 41,849,716,736. No sudo rights were granted,
reserved filesystem settings changed, cache payloads deleted or writers
restarted. This replaces the previous Tailscale identity-check blocker, not the
remaining privileged-inspection requirement. Task blocker event:
`evt-065a4d05f1cd4d23a63d21471b151b61`.

### October 22 Local Storage Follow-up

At the user's request, Terminal was opened with interactive cortex SSH and a
`sudo -v` prompt. Opening it is not evidence of successful authentication and
does not transfer sudo credentials to independent SSH sessions. No cleanup,
service restart, deployment or sudo policy change was performed here.

Managed publication has since passed twelve focused tests and a narrow
independent source review. This approves that component only, not all of 3.1.
The own-user shared resource slice/retry policy is committed as `7d2b2de` and
passed actual Linux offline systemd parsing; it is not installed or proven
enforced on the live server. Current-session hook readback observes prompt,
tool and Stop handlers; the earlier unverified-Stop statement is historical.

Bounded export work now covers actual model CSV/Joblib replay, dataset and
feature Parquet builds, object snapshots, official corpus capture callbacks,
snapshot construction and forward-OOF NPZ publication. Storage guards are
cooperative, not an OS filesystem quota. Native SQLite/WAL, Scrapy queues,
discovery matrix writers and tracking delivery still require qualification.
The explicit remaining inventory is
`.mega/storage-writer-coverage-20261022.md`.

An immutable-source regression passed 653 tests, three skipped and 217
subtests before independent review corrections. That review then reproduced
cross-stream reservation release, concurrent feature-definition overwrite
and excess inode reservation. All three have new reproductions and local
fixes; 51 tests/four subtests passed in the affected storage/preparation set.
The OOF/pipeline set passed 48 tests/nine subtests. Counts overlap and are not
additive. The 653-test snapshot predates these fixes and is not current-tree
final qualification. Subsequent independent re-review approved the three fixes
and bounded NPZ export only, including an additional growth-denial/readback
probe. The current-source wider run passed **660 tests, three skipped, and
217 subtests** in 255.33 seconds. Source, revision and installed environment
remained unchanged during that run; JUnit reports zero failures/errors. The
native review is preserved in `.mega/close-curie-exports.json`, and the full
regression inputs/readback are in `.mega/final-regression.json` and XML/stdout
siblings. This remains component evidence, not full 3.1 qualification.
A prior local identity regression was caught and
fixed: narrative-only changes reuse the semantic definition without rewriting
its stored payload.

Obligation 3.1 remains open. The privileged reclaim step remains blocked, and
neither these tests nor the offline parser authorize normal writer startup.

### Tracking, Discovery And Controller Recovery Checkpoint

Tracking JSON/report exports, transformed-payload enqueue sizing, independent
outbox delivery and private/shared discovery producer admission have now been
implemented locally. Read-only WAL connections can create sidecars: a review
reproduced this, and admission now covers their entire connection lifetime,
including controller count reads. A blocked count remains unknown, not zero.

Further independent fault injection found startup sequence initialization and
synchronous trace-enqueue failures still escaped the retry/planning gate.
Local corrections now retry sequence initialization, pause paid planning,
preserve exact failed execution payload/sequence, and retry infrastructure
pressure without consuming the normal three-error trace circuit. Tests cover
four pressure failures followed by successful delivery, startup recovery and
byte-equivalent synchronous payload retries.

The latest frozen-input regression passed **687 tests, three skipped and
217 subtests**, zero failures/errors, in 266.59 seconds. Source, revision and
installed environment were unchanged. Receipts use
`.mega/storage-controller-regression-20261022.*`. The deliberately interrupted
earlier expanded run is not passing evidence. The 685-test checkpoint predates
the final controller fixes; the older 660-test receipt is also preserved.

**Not crash-safe yet:** a rejected pre-enqueue intent currently survives only
in controller memory. Normalizing and retrying that payload is not a durable
emergency-control mechanism. An independent design review recommends a bounded
two-slot SQLite control store feeding the existing outbox, with allowance
inside normal headroom and no new paid dispatch on uncertain accounting.
[The supplemental design](IMA_STORAGE_CONTROL_RESERVE_DESIGN_20261022.md)
records implementation/proof order, alternatives and required failure tests;
it does not change the canonical plan or authorize provisioning/rollout.

Sizing evidence comes from the verified frozen backup, not live guesses:
5,110 complete payloads, decision maximum 1,282,920 bytes and execution maximum
714,213 bytes. The backup hash was unchanged before/after read-only analysis.
These maxima are historical observations, not future envelope guarantees.

Latest runtime hook readback observed prompt/tool events but did **not** verify
Stop coverage in its current effective context. Earlier verified receipts are
historical; no current enforcement claim is inferred from their presence.
No cache deletion, live unit installation, writer restart or privileged
recovery occurred in this follow-up. All original obligations remain intact.

### October 22: Control Integration And Privileged Inspection

The earlier memory-only warning above describes that historical checkpoint.
The bounded SQLite control component and explicit digest-bound bootstrap CLI
are now committed. Current local controller integration reserves before planner
dispatch and retains complete normalized envelopes for restart replay. Unknown
control state freezes paid planning and leaves total spend unknown while
preserving known subtotals. Infrastructure storage failures do not exhaust the
normal trace/model-upload retry circuit. Operational diagnostic history is
bounded to 100 entries of at most 8,192 characters.

The initial controller test timeout was caused by a noncanonical macOS `/var`
fixture path. Resolving the fixture dataset, protocol and campaign paths fixed
it without changing production path validation or increasing that test timeout.

Independent failure injection then found three real integration defects:
normal decision receipt failure could strand an already received response;
ordinary final enqueue failure could return a clean stop despite a complete
pending intent; and unreadable control state could leave billing marked known.
Corrections retain the response in the frozen envelope, publish before the
normal decision JSON receipt, refresh final pending state for every exception,
and reconcile cost again after that refresh. Fallback billing uses immutable
control/outbox usage only when normal/physical receipts are absent, without
double counting. These corrections remain subject to final independent review
and the new frozen-input regression; older test receipts do not certify them.

Interactive sudo was verified, and a privileged read-only process inspection
completed at `2026-10-22T09:36:54Z`: zero scan errors, zero observed open
descriptors/mappings under research-v2, and zero user-available filesystem
bytes. The receipt is `.mega/live-evidence/root-proc-readback-20261022.json`.
This is a dated point-in-time observation, not producer quiescence or reclaim
approval. Named iMa producers/timers were also refreshed as inactive/failed;
Cortex, Solar and the running MLflow service were not changed.

A proposed saved-root-UID inspection bridge was rejected after independent
review reproduced unsafe credential failure paths and incomplete reference
matching. Its queued synthetic RAM canary was cancelled. The draft and its
historical tests were removed from runtime/test discovery and retained only
under `.mega/rejected-drafts/`; no successful Linux qualification or production
reclaim is claimed. Decision `own-uid-privileged-inspection-20261022` records
the rejection. Any successor must isolate a permanently unprivileged mutation
process from privileged inspection, qualify namespace/inode visibility, and
retain all selected-manifest, quiescence, lock, rebuild and package gates.

Obligations 2.1, 3.1 and 3.2 remain open. Actual SDK/native SQLite growth,
remaining control-plane writes, global cache qualification, and live startup
still need proof. No scientific score, reclaimed capacity, model promotion,
server deployment or writer restart is attributed to these local corrections.

The subsequent frozen-input run passed **745 tests and 217 subtests, with three
skipped**, zero errors/failures, in 417.28 seconds. The code revision, all
captured source inputs, and installed interpreter environment were unchanged.
Receipts: `.mega/control-durable-regression-20261022.*`; source/environment and
consumer readback: `.mega/control-durable-regression-20261022-readback.json`.
Independent integration approval and the $0.07 receipt-failure/restart probe
are preserved in `.mega/close-poincare-control-integration.json`. Two additional
real local MLflow package logging/reload tests passed; these do not establish
SDK scratch/native allocation coverage under storage pressure.

This is a local, uncommitted integration checkpoint on the dedicated feature
branch, not a completed implementation, pushed release or deployment. The
canonical task still has fifteen unmet obligations. All owned reviewers have
been closed with native receipts; the rejected privilege draft has no live
replacement. The installed server `setpriv` tool and its upstream documentation
are being checked for an isolated, permanently unprivileged mutation process;
that successor has not been implemented or qualified.

## October 25 access refresh and local publication checkpoint

The newly authenticated cortex Terminal session returned UID 0 from
`sudo -n id -u`. The privileged read-only process scan at
2026-10-25T14:26:18.681939Z covered 445 processes without scan errors or
unscanned active processes and observed zero open descriptors or mappings
under `/home/imaopt/research-v2/`. At that instant normal-user available space
was 130,744,320 bytes. This is a point-in-time observation, not a guarantee of
future quiescence or approval of any cache payload. Receipt:
`.mega/live-evidence/root-proc-readback-20261025.json`. User-level iMa producer
services and iMa timers were inactive/failed; MLflow remained running. No
cache was deleted and no server service was changed during this refresh.

MLflow SDK upload admission is now committed as
`4507ce871323bb3dc8e210b5584dd9aba617b0fe`. Fresh isolated-package tests passed
40 tests before commit and 40 after commit; these are overlapping checks,
not 80 distinct tests. Receipts:
`.mega/model-upload-staged-canary-20261025.*` and
`.mega/model-upload-committed-canary-20261025.*`. Registered models were loaded
and predictions read back; denial, scratch/artifact destination checks,
wrapped native pressure and flush lifetime were exercised. Independent scoped
approval is preserved in `.mega/close-locke-model-upload.json`. Sampled copy and
cleanup boundaries are not a hard peak bound or production capacity proof.

The controller fallback dataset export now admits before loading source data,
uses a bounded gzip stream and removes a failed private temporary export.
One reusable staging slot bounds orphans across interrupted retries; all three
production callers hold the campaign lock. The parent directory is fsynced
after publication, and rejected staging symlinks remain untouched.
Explicit input and cache-hit paths remain read-only. The initial focused run
passed 25 tests including storage-core tests. This change is not yet a full
controller qualification. After the review corrections, 27 focused tests
passed independently. The wider controller/dataset run passed 40 tests. An
isolated package check initially failed because the archive omitted committed
test fixtures; that failure receipt is retained. After including those fixture
inputs, the isolated staged-package check passed 40 tests in 86.28 seconds
without dirty-tree imports. Receipt:
`.mega/controller-dataset-staged-canary-20261025-v2.*`. The pre-existing delta
consumer hunk was excluded from this staging unit and remains uncommitted.
Independent scoped review and closure:
`.mega/close-confucius-controller-dataset.json`. Overlapping counts are not
summed. The SDK and dataset checks do not establish all-writer qualification.

Independent source review identified remaining campaign JSONL, status/summary,
owner metadata and database-growth gaps. The bounded emergency trace store
does not cover those paths. Obligations 2.1, 3.1 and 3.2 remain open; original
scientific and unattended-qualification obligations are preserved. No score
improvement, deployment, reclaim or campaign restart is claimed here.

### Later October 25 staged JSON publication checkpoint

The shared controller JSON publisher now has a staged enabled-storage branch
with normal admission, bounded writes, file/directory fsync, preservation of
the preceding JSON on pre-replace failure, and staging cleanup while the lease
remains held. A shared cleanup helper retains the primary exception and adds
a cleanup-error note if unlink also fails; the fallback exporter uses it too.
The final focused integration run passed 38 tests. The isolated committed-base
plus three-file staged-overlay check passed 50 tests, with unchanged index and
base: `.mega/controller-json-staged-canary-20261025-v2.*`.

These latest changes remain staged and uncommitted. Reviewer Mill reported one
intermittent failure of
`test_controller_reserves_before_planner_and_clears_completed_intents`: one
trial completed, one tracking item remained, the mode was `blocked_tracking`,
and no tracking error or blocked-storage operation was recorded. Isolation and
the subsequent four-file rerun passed, but the cause is unknown. Passing
reruns do not resolve that concern. The exact final native reviewer response
and closure are retained in `.mega/close-mill-controller-json.json`.

The proposed inode-aware process-inspection executor made no changes because
its higher-priority turn instruction limited it to review/status work. It was
closed with the actual receipt `.mega/close-jason-reclaim-inspection.json`.
No new inspection module or qualified privilege/deletion bridge exists. The
old credentials blocker for 2.1 was cleared using actual sudo evidence, not
replaced by a claim that reclaim has passed. Original obligations remain open.

### October 27: actual selected first-wave reclaim and consumer readback

The earlier inspection/execution limitation is resolved. Commit `3b378bd`
adds an observer-only privileged process scanner and an isolated eviction child
permanently running as UID/GID 1001, with empty supplementary groups and all
capability sets zero. Native credential-drop, real-payload inspection and
RAM-only held-FD/held-lock canaries passed. The isolated committed package
passed 64 tests. The exact reviewed production launcher rejects mixed UID
tuples and binds its manifest, source buffers and prerequisites by SHA-256.

The approved manifest selected 332 regenerable fold-cache payloads, totalling
57,781,940,224 allocated bytes. The production operation removed **320**, then
stopped when three bounded process scans could not obtain a stable process
set. **Twelve payloads were held**, including the refused item and eleven
unattempted items. This was a fail-closed stop, not successful deletion of all
332. Every held payload retains its original directory identity and all three
files with identical SHA-256, inode, ownership, size and modification time.
All 332 stable key-lock files remain; no source/model/raw dataset/history
deletion is authorized by this operation.

Measured available/free-space gain during the operation was **55,674,880,000
bytes**. The 320 removed payloads accounted for 55,674,974,208 allocated bytes;
the small difference remains visible rather than equating allocation with the
whole filesystem delta. Available space rose from 8,622,080,000 to
64,296,960,000 bytes; available inodes rose from 58,835,478 to 58,836,753.
These are the actual operation-time measurements, superseding earlier full-disk
snapshots without attributing unrelated changes before the operation to this
cleanup.

Post-operation readback found 320 removed paths, twelve intact held paths and
no surviving recorded actor/child. All 111 protected boundaries retain their
recorded identities, all selected retained dataset/config/identity hashes
match, and `ima-mlflow.service` remains active. The original champion package
`7ac12999d91540b5` loads and produces exactly the same fourteen probabilities
with identical package and inference-source hashes. Normal iMa producer units
remain inactive/failed. No Cortex/Solar restart or other-user operation ran.

Evidence is in `.mega/live-evidence/reclaim-events-20261027.jsonl`,
`reclaim-filesystem-readback-20261027.json`, `held-after-reclaim-20261027.json`,
`protected-after-reclaim-20261027.json`, `package-after-reclaim-20261027.json`
and `selected-reclaim-readback-20261027.json`. The last receipt binds the exact
source inputs and frozen manifest. Independent outcome review passed; obligation
2.1 is qualified, with test and readback proofs covering every source input.
Review receipt: `.mega/close-meitner-reclaim-outcome-20261027.json`.
The held twelve remain held; fresh
inspection and authority-bound selection are required for any later operation.

### October 27: atomic JSON checkpoint and remaining storage gates

Commit `4a3a03c` now contains the previously staged atomic JSON/cleanup unit.
An exact isolated committed-base plus three-file overlay passed **37 tests**;
its source-bound independent review and committed-blob verification are in
`.mega/close-wegener-controller-json-20261027.json` and
`.mega/controller-json-readback-20261027.json`. This is component qualification,
not completion of 3.1.

The intermittent controller test had a twelve-second success deadline.
Independent diagnosis identified expiry as the mechanism consistent with a
completed trial and pending upload, but did not reproduce the intermittent
failure or establish why those runs crossed the deadline. The success fixture
now allows one minute; a separate test advances only the controller clock
after a completed fit and proves pending upload durability, no new upload after
expiry and `blocked_tracking`. All thirteen trace-control tests passed. These
test-only changes do not alter production campaign deadlines.

Campaign JSONL/status-summary/owner writes, native database growth, complete
writer/resource qualification and shared cache lifecycle remain open. Normal
restart still requires both 3.1 and 3.2 plus dependent database recovery gates.
The broader data/research/confirmation/telemetry/unattended work remains in
the original canonical plan; this reclaim is not whole-plan completion.

### October 27: staged campaign log/report publication checkpoint

The next isolated unit routes both periodic and final Markdown summaries
through the same admitted atomic byte publisher as JSON. JSONL appends reserve
before creating their file or missing parent directories, validate the actual
file again after acquiring its inode lock, and keep that lock through buffered
stream closure and failed-append rollback. The rollback length is captured
after locking, so a complete competing append is preserved. Hardlinks created
while waiting are rejected. No crash-atomic append or protection against an
arbitrary malicious same-user process is claimed.

Atomic JSON/text staging records its opened inode and checks the path before
publication; a replaced staging path is held rather than published or removed
by cleanup. Regression fixtures replace it with both regular files and symlinks
for JSON and Markdown consumers, preserving the preceding accepted artifact.
Missing-directory inode demand is included for both append and atomic paths.

The exact clean committed-base plus four-file staged-overlay check passed
**56 tests** in `.mega/controller-publication-staged-canary-20261027-v5.*`.
The first harness omitted the committed research memo, producing 48 passes
and one fixture-input failure; that original receipt is retained. Subsequent
49/52/56-case checks overlap and are not summed. Independent review approved
the exact source under its serialized private-staging contract. The unit is
committed as `549afae`; `.mega/controller-publication-readback-20261027.json`
verifies every committed blob against the isolated tested overlay. Native
review/closure: `.mega/close-arendt-controller-publication-20261027.json`.
None of these local checks qualifies the remaining SQLite/WAL,
metadata/control, process startup or aggregate-resource paths.

### October 27: committed rollback-SQLite and controller-owner components

Commit `2fca33d` adds a shared admitted transaction owner and integrates
`HypothesisMemory`. It reserves before directory/database creation, serializes
cooperating writers, reconciles growth under the stable lock, reads back the
native main-page cap, and preserves primary storage errors through cleanup.
Rollback-journal peak and main growth are separate from retained-journal
settings. WAL and possible hot-journal recovery remain explicitly unsupported;
neither mode is silently migrated or repaired. Each hypothesis record receives
a fresh conservative growth envelope covering all bound fields and index
splits; this is not a permanent database-size limit.

The isolated exact-index check passed **38 tests**, including native
`SQLITE_FULL` rollback, long-ID record growth, shared-capacity contention,
readback/replay and legacy controller integration. Its first 37-case check
found a real disabled-mode compatibility issue for symlink-parent paths; that
was fixed and a new regression added. Independent approval and committed-blob
readback are `.mega/close-sagan-sqlite-review-20261027.json` and
`.mega/sqlite-component-readback-20261027.json`. This qualifies one metadata
owner, not all SQLite/WAL consumers or an OS filesystem quota.

Commit `612c508` admits campaign directory creation and stable controller-lock
metadata before their writes. The locked inode remains stable. A duplicated
descriptor retains the lock throughout campaign execution while the metadata
reservation/context is released before yielding. Setup error classification
ends before the campaign body, so training errors keep their original identity.
The status lock probe opens read-only/no-follow and creates no missing lock.
The isolated exact-index check passed **35 tests**; independent approval and
committed readback are `.mega/close-pascal-owner-review-20261027.json` and
`.mega/controller-owner-readback-20261027.json`.

These counts overlap earlier checks and are not summed. Research-ledger,
decision/outbox/WAL, acquisition-native outputs, remaining control metadata,
production peak envelopes and aggregate-resource qualification are still open.
Normal writers remain held until both 3.1 and 3.2, plus dependent recovery gates,
qualify. No model-training or scientific improvement is claimed by these
storage components.

### October 27: ledger, decision, outbox and runtime checkpoints

The first-wave reclaim remains qualified separately: 320 exact derived payloads
were removed, 12 were held intact, and measured available space increased by
55,674,880,000 bytes. A fresh read-only server check at this checkpoint reports
64,294,862,848 available bytes and 58,836,605 available inodes. The system
`ima-mlflow.service` is active; this is distinct from the inactive user unit
with the same name. Normal research writers have not been restarted.

Commit `8d44787` admits ResearchLedger writes and retains infrastructure retry
deadlines durably; its status reader avoids migrations and writable SQLite
opening under storage pressure. Commit `3e9dcb5` adds admitted DecisionStore
transactions and action receipts. Existing WAL databases are refused in enabled
mode, rather than implicitly repaired or migrated. Their isolated checks and
native independent-review receipts are bound in
`.mega/ledger-component-readback-20261027.json` and
`.mega/decision-component-readback-20261027.json`.

Commit `6f1431f` adds a native outbox SQLite connection that retains its own
capacity lease until close, refreshes its main/journal envelope before writes,
and commits before network flushing. Frozen control intents are acknowledged
only after exact outbox readback. Its exact-index canary passed 124 pytest tests
and 15 subtests. The original success fixture's 64 MiB capacity was insufficient
for the deliberately independent artifact and native SQLite leases; it now
provides 128 MiB, with explicit denial tests retained. This is conservative
cooperative admission, not a physical filesystem quota.

Commit `a2f43da` admits worker-runtime metadata, preserves legacy disabled-mode
behavior, and keeps failed staging cleanup inside the stable-lock interval.
An adversarial test proves another updater cannot acquire that lock during
cleanup. Its isolated check passed 49 pytest tests and 5 subtests. Native
independent review and committed-blob readbacks are
`.mega/close-goodall-outbox-runtime-20261027.json`,
`.mega/outbox-component-readback-20261027.json`, and
`.mega/runtime-component-readback-20261027.json`.

These checks overlap and are not summed. A separate fault injection confirmed
that Optuna can persist trial creation before recipe metadata, leaving a RUNNING
orphan that consumes budget. Its recovery repair and service launch enforcement
are still in progress. The verified production Optuna version is 4.5.0, versus
4.9.0 locally; compatibility must be checked before rollout. Obligation 3.1 and
the whole task remain open.

### October 27: journal recovery, cache headroom and fit publication

Commit `24d22eb` adds admitted Optuna journal storage and complete ask-envelope
admission. Incomplete native asks retain a durable infrastructure reason and
are pruned without consuming the research budget; fully formed asks remain
eligible for ledger recovery. Recovery never reconstructs an executable recipe
from a half-written parameter set. Native append rollback and reader replay
share the stable data inode lock, preventing another reader from observing a
complete log line that is subsequently truncated. Post-lock identity checks
and an explicit reader-at-lock barrier qualify this failure case. Exact-index
checks passed 30 tests against both Optuna 4.9.0 and the isolated 4.5.0 wheel.
The server's actual 4.5.0 interpreter also passed create/ask/tell/reopen in RAM.
Review/readback: `.mega/close-fermat-journal-review-20261027.json` and
`.mega/journal-component-readback-20261027.json`.

Commit `78d48da` admits feature-queue schema/updates and metadata startup paths.
Read-only queue access honors native SQLite writer locks and closes handles;
it does not use SQLite's `immutable` flag on a changing database. The isolated
check passed 34 tests, with native independent review/readback in
`.mega/close-harvey-feature-queue-review-20261027.json` and
`.mega/feature-queue-component-readback-20261027.json`.

Commit `60448f8` serializes package publication, publishes its manifest last,
and preserves public accepted packages by default. The three executor save
calls explicitly permit retrying their private fit output. Retry validates the
entire allowlisted directory before removing any payload; unknown, linked or
foreign entries hold the operation. Lock setup sync failures close every
duplicated descriptor. Legacy opt-out saves retain overwrite behavior and create
no lock sidecar. The isolated check passed 35 tests plus 15 subtests, including
real fit/package/MLflow readback. Its independent source review and committed
blob readback are `.mega/close-aristotle-package-review-20261027.json` and
`.mega/model-package-component-readback-20261027.json`.

Commit `fdaddb9` adds physical-byte/inode headroom trimming alongside the shared
cache quota. Existing key/pin locks protect active artifacts. Cache-hit demand
matches the subsequent 1 MiB/eight-inode reservation; a producer winning between
trim and reserve is denied atomically rather than overcommitted. Executor pins
transfer while the preparation lock is held. The isolated check passed 73 tests
plus 29 subtests. `.mega/cache-capacity-component-readback-20261027.json` binds
the exact source and independent review. The original operated core storage
source remains unchanged, so the qualified 2.1 outcome remains reproducible.

Commit `c4c5df6` admits FIT/FOLD outputs before directory creation. Native storage
pressure propagates without writing a scientific failure result. JSON uses the
shared atomic publisher; CSV is streamed through an admitted file, synced and
published with identity checks and owned cleanup. Metadata-only table estimates
do not enable a new dataset representation. Its isolated check passed 57 tests
plus 22 subtests; `.mega/fit-storage-component-readback-20261027.json` binds the
independently reviewed committed unit.

The sealed server canary in
`.mega/live-evidence/native-storage-ram-canary-v4-20261027.json` ran committed
SQLite, Optuna and preparation modules on Python 3.12.12/SQLite 3.50.4/Optuna
4.5.0. It verified native FULL rollback, short outbox transactions, lease release,
cross-root cache quota, pin preservation, and bit-exact cold rebuild. The scaled
quota fixture observed 385,024 available bytes reclaimed; its final allocated
footprint was 802,816 bytes, which is a final readback rather than a peak measure.
It used private RAM fixtures under an explicit 32 MiB reservation and removed
them. It did not force a physical-full filesystem or mutate production data.

These counts overlap and are not summed. An earlier broad working-tree diagnostic
reported 1,875 tests passing, seven failures and seven skips while edits were
still in progress. Its failures were investigated: stale budget/native-FULL
fixtures, invalid test probabilities, a real missing-recipe legacy admission
bug, and a stale single-message planner-memo assertion. Focused repairs passed;
that changing-tree run is not release qualification. Bounded native Scrapy
queue/log persistence, budget-control writes, remaining build/export paths and
normal restart/unattended gates are still open. No normal research restart or
whole-phase completion is claimed.

## Continued Native Writer Qualification — October 28

Commit `924bb65` admits feature-service budget metadata and preserves its fixed
100 GB aggregate partition. Storage denial now exits an `ExecCondition` with 2;
unexpected reserve failures map to 255. The native systemd documentation supports
the resulting skip/failure distinction and `ExecStopPost` cleanup. The exact
condition shell was executed for four exit cases, and the isolated suite passed
19 tests. Independent review and committed readback are recorded in
`.mega/close-plato-budget-review-20261027.json` and
`.mega/budget-component-readback-20261027.json`. Actual systemd execution remains
a deployment qualification requirement.

Commit `c179ee4` guards registry and feature-lifecycle directory creation, locks,
original-format Parquet builds, and provenance exports. Native directory/lock
creation reserves byte and inode allowances before mutation. Failed builds retain
their storage failure if cleanup or status publication also fails. The sealed
index suite passed 69 tests plus 10 subtests, including actual parent/successor
readback and injected partial Parquet output. Independent review and committed
readback are `.mega/close-popper-dataset-review-20261028.json` and
`.mega/dataset-component-readback-20261028.json`.

A read-only server check at this checkpoint reported 64,292,950,016 available
bytes, UID 1001, active system MLflow, and no running own-user research service.
These two commits do not qualify all of 3.1 or 3.2, restore damaged databases,
or start a normal campaign. Native Scrapy persistence and snapshot/report writer
qualification continue separately. No experimental progress is inferred from
these engineering tests.

Commits `800e3b2` and `0f27c09` add native acquisition guards and their tests.
They preserve the installed Scrapy 2.19.0/queuelib 1.10.0 persistence formats,
including the default downloader-aware priority queue. A real-reactor fixture
found an incompatible priority-state assumption that smaller tests missed; it
was repaired before commit. Five no-network reactor cases now verify normal
capture and enqueue/scheduler-close/spider-state-close/receipt storage failures.
Shutdown failures are retained even when Scrapy suppresses handler exceptions.
The sealed six-file suite passed 66 tests. Source and tests were committed
separately to satisfy the host's two-top-level-area commit gate without an
override; the final combined tree matches the independently reviewed six-file
canary. See `.mega/acquisition-component-readback-20261028.json`.

The actual server acquisition interpreter then passed the committed native
consumers in `.mega/live-evidence/native-acquisition-ram-canary-20261028.json`:
raw gzip and SQLite inventory readback, original queue resume, original duplicate
filter readback, bounded log rotation, and release of every reservation. Its
private tmpfs fixture was removed. Sampled allocations were 45,056 then 65,536
bytes; these are samples rather than a measured peak. No HTTP crawl, production
source installation, database mutation or normal service restart occurred.
Arbitrary hostile code running as the same UID is outside the cooperative writer
contract; pathname identity checks do not provide isolation against that actor.

Commit `1fbc885` admits feature-definition root creation and immutable
publication. First-writer-wins behavior intentionally also applies when storage
admission is disabled; this is a policy change from the old overwrite behavior,
not a claim of unchanged legacy semantics. The sealed two-file suite passed
50 tests; `.mega/feature-definition-component-readback-20261028.json` records
the independent source review and committed hashes. Native lifecycle and
publication integration, snapshot/report qualification and normal restart gates
remain open.

### Continued qualification and host-restart recovery, 2026-10-28

The original-format snapshot/report unit (`30386df`), accepted physical-fit
publication (`c8a45d4`), program-preparation admission (`0f26e19`), guarded
Featuretools production (`a3f880a`), own-filesystem worker guards (`501d4e9`),
and infrastructure-wait identity helpers (`b561f06`) are committed components.
Their separate sealed-canary/readback receipts remain under `.mega`; none alone
qualifies the complete writer inventory or authorizes a normal campaign restart.

After the host interrupted the parent, the pending reviews were no longer
available through native agent lookup. Fresh reviews replaced the unfinished
object, cache and graph checks; no unavailable review was counted as approval.
Fresh Tailscale SSH works without further authentication. Read-only evidence
`.mega/live-evidence/health-recovered-20261028.json` records UID 1001, active system
MLflow, no running own-user research service, and 72,759,558,144 filesystem bytes
available. This is a point-in-time capacity observation, not durable quota proof.

Immutable object foundation commit `be6bac7` passed 11 isolated tests and a fresh
independent review. Published objects survive post-publication errors so another
writer's accepted reference cannot become dangling. Only writer-owned temporary
names are cleaned up; this implementation has no garbage-collection authority.
See `.mega/object-component-readback-recovered-20261028.json`.

The combined graph/search tree (`c88a990` and `57bbdcd`) passed 70 tests plus
39 subtests against a sealed index overlay. Forward pools enforce chronological
prediction lineage and reject incompatible contracts. A tiny actual package
fitted and serialized using committed old code loaded under the new graph code
with bit-exact predictions across 12 rows, unchanged canonical recipe/hash, and
matching package/graph identities. This fixture proves compatibility only, not a
new scientific gain or independent champion confirmation. Proof is linked in
`.mega/graph-component-readback-recovered-20261028.json` and
`.mega/graph-compat-recovered-20261028/compatibility-result.json`.

Independent cache review identified a validation-to-path-deletion race; the
descriptor-bound fix is being implemented before release. The cache core used
for the historical reclaim remains byte-identical. No additional production
cache payload was deleted and no normal campaign was started at this checkpoint.

### Controller, native consumers and complete cache survey, 2026-10-28

The integrated controller tree (`0eb4308`, `16c55b2`, `41ba3df`) passed 203 tests
plus 87 subtests in an archived, exact 15-file overlay. Two absent local audit
fixtures were explicitly skipped. Independent review found and drove fixes for
three crash/admission windows: worker-peak headroom was being overwritten; a
saved paid response could remain stranded behind a reserved trace intent; and
startup accepted-fit reconciliation could exit on ledger pressure. Tests now
exercise those windows, including both before/after decision application without
another paid call. Storage-wait IDs are included in degraded summaries. The
source-bound receipt is `.mega/controller-component-readback-recovered-20261028.json`.

Committed old/new scientific package compatibility is also checked against the
actual server champion, not only a generated fixture. All 14 probabilities from
package `7ac12999d91540b5` are bit-exact under the streamed committed code, and
all package-file hashes remain unchanged. This is saved inference parity, not a
full retraining or independent scientific confirmation. See
`.mega/live-evidence/champion-committed-memory-recovered-20261028.json`.

Cache cleanup commit `ce94b7e` passed 71 tests plus 19 subtests. A fresh native
server canary also passed SQLite FULL rollback, native outbox lease retention,
Optuna ask/tell/reopen, cross-root cache quota/pins and bit-exact cold rebuild.
The exact reviewed preparation overlay is now committed; its source hash is
bound in `.mega/live-evidence/native-cache-recovered-20261028.json`. Root
filesystem fullness was not forced. Commit `c360349` adds fail-closed inventory
across explicitly supplied roots; 14 tests verify all-or-nothing metadata import,
unknown/protected holds, root identity and native empty-lock formats.

A production-path survey read 13 own-user cache roots without registering or
deleting any artifact. It found 76 structurally valid fold artifacts occupying
11,278,737,408 allocated bytes (about 10.5 GiB), plus 18 empty native build
directories and two zero-byte owner-temporary files. The 12 previously held
payloads remain excluded from any removal. Survey and diagnosis are in
`.mega/live-evidence/cache-inventory-survey-recovered-20261028.json` and
`.mega/cache-temporary-plan-recovered-20261028.json`. A quarantine plan is only a
proposal at this checkpoint; no move has been applied.

Commit `0703510` separates hash-pinned retention holds from quota accounting.
The sealed suite passed 55 tests plus four subtests. Invalid policy files,
non-regular/FIFO inputs, duplicate JSON keys and ambiguous schema types fail
closed before cleanup. Native Linux verification passed held-byte preservation,
held-only quota refusal, changed-policy refusal and complete lease release in a
private tmpfs fixture. Two earlier fixture harnesses lacked lazy dependencies;
their failure receipts are retained, and the complete committed-source loader
passed in `.mega/live-evidence/native-cache-holds-recovered-v3-20261028.json`.
This is not a production policy-installation receipt.

The existing server MLflow 3.16.1 SDK emitted one real local-fixture trace through
committed telemetry. SDK and SQL readback agree on 137 input tokens, 29 output
tokens, 166 total tokens and USD 0.001234567. Reservations remained held through
flush/readback and were all released; SQLite integrity is `ok` and page count
214 stayed below the explicit 2,048-page limit. Sampled fixture bytes reached
925,499; this is an observed sample, not a bound on every transient allocation.
See `.mega/native-mlflow-recovered-20261028-receipt.json`. Production tracking
destinations, concurrent uploads and the dashboard are still separate gates.

Additive snapshot commit `c6a7d9b` passed 20 tests and independent review.
Legacy exports remain the default. Objects-v2 and explicit CSV export are opt-in;
the snapshot captures/hashes both object-storage and capacity implementation
sources. No production dataset migration, normal service restart or new
statistical improvement is claimed by these engineering checkpoints. Obligations
3.1 and 3.2 remain open pending live integration and workload qualification.

### Actual cache-control handover checkpoint, 2026-10-28

The twenty reviewed empty native temporary entries were moved, without deleting
payloads, into a unique private same-filesystem quarantine. Atomic no-replace
rename, pinned directory descriptors, per-entry intent/completion receipts and
native inode readback protect the operation. Independent Zeno review approved
the exact actuator/proposal after thirteen local and thirteen Linux fixture
tests. The first driver invocation refused before any move because its process
substring check matched the streamed Python command itself; the corrected
driver checks command argument basenames. Both failure and success receipts
remain preserved. See `cache-quarantine-applied-recovered-v2-20261028.json` in
`.mega/live-evidence` and `dec-empty-cache-quarantine-20261028`.

Canonical `/home/imaopt/research-v2/storage-control` now contains a hash-pinned,
mode-0400 retention policy for the twelve protected caches, and a complete
inventory of 76 native artifacts totaling **11,278,737,408 allocated bytes**.
All 36 retained files were hashed before and after the installation against
their saved receipts. Automatic eviction was refused for all twelve. Final
reservations were zero. Goodall independently read the actual host policy,
inventory and all twenty quarantine receipt pairs; that review deliberately
did not repeat payload hashing. See
`.mega/live-evidence/cache-inventory-installed-recovered-20261028.json` and
`.mega/close-goodall-live-cache-review-recovered-20261028.json`.

Policy environment inheritance by successor services remains a release gate.
No normal research service has been restarted. Other users and services were
not changed. The shared policy reserves 50,214,916,915 bytes of filesystem
headroom at the observed filesystem size; cache hard quota is 50 GiB. These are
disk controls, separate from the shared 100 GB decimal RAM ceiling.

Committed `40105063` also passed a native sealed controller canary: ten actual
model trials on an explicitly synthetic fixture completed in 55.55 seconds,
with 8,724,480 final allocated bytes. The 512 MiB qualification reservation
covered extraction and fixture work and was released. The first archive omitted
the mandatory research memo and failed before training; its receipt and release
are retained, and the corrected archive includes the exact committed memo.
See `.mega/live-evidence/sealed-controller-canary-recovered-v2-20261028.json`.
This proves a bounded synthetic execution path; it is neither real-data science
qualification nor a concurrency benchmark, production MLflow readback, or
permission to mark all of obligations 3.1/3.2 complete.

The follow-up synthetic benchmark ran twelve completed model trials at each
configured cap of 1, 2 and 4. Elapsed times were 174.11, 174.00 and 173.56 seconds;
sampled aggregate private USS peaks were 495,935,488, 526,102,528 and
701,923,328 bytes. Sampled CPU busy fractions were 2.31%, 2.39% and 2.41% of
the host's 28 logical CPUs. Final output allocations were approximately 7.2 MB
per fixture. This low-volume synthetic workload did not establish saturated
concurrency or a production-optimal cap; no production worker limit was changed
on that basis. All reservations were released. The native status files retain
cgroup observations separately; the sampling driver did not claim a cgroup
peak bound. See `.mega/live-evidence/controller-worker-caps-recovered-20261028.json`.

The canonical native two-slot control-intent store has also been explicitly
bootstrapped. Its maximum envelope is 1 MiB, SQLite limit is 1,024 4-KiB pages,
and complete allowance is 80 MiB including conservative rollback-journal and
metadata limits. Actual allocated bytes were 28,672 and pending intents were
zero. The first bootstrap driver successfully created the store but failed its
readback assertion by comparing the API's empty tuple with an empty list; the
idempotent corrected readback passed. Both receipts are retained. No provider
request was dispatched, and predecessor billing evidence was untouched. See
`.mega/live-evidence/control-bootstrap-recovered-v2-20261028.json`.

Projection commit `a619610` passed 45 tests plus 15 subtests against the exact
index package and bounded independent review. It drops only unused additions
from the top delta layer before Parquet decoding, retains parent/replacement
columns and transitive availability bindings, and uses the same loader for
training and inference replay. Graphs, transforms, adaptive discovery and
performance-distribution recipes retain the full-read path. Fold-row projection
and a real memory/copy benchmark remain open parts of 4.2.

Launch guard commit `0aedf70` passed 52 tests plus 32 subtests and independent
review. Both retention-policy environment variables are required before the
production interpreter starts; the preflight validates the actual nonempty
hash-pinned policy. Those source checks are not a receipt for installation of
the successor service environment.


### Native paid-planner and resource readback, 2026-10-28

The isolated combined builder/research runtime now imports the actual HTTPX
transport and passes `pip check`. The original research and acquisition
environments were not installation targets. Source startup preflight runs in
the actual launch interpreter before campaign/request state is created; an
import or incompatible transport version fails before an HTTP request.

After account funding, the bounded real-data campaign at
`paid-controller-e7611e74611e-20261028/campaign` completed four accepted Benter
fits and preserved one failed Gaussian-probit fit. It exited successfully with
no pending tells or tracking uploads. Four native MLflow runs read back as
`FINISHED`, registered as versions 1371–1374. The best new fundamental race
log loss was **2.1659502442341783**; this does **not** improve the reproduced
35/65 reference at 2.1494157522885775. Six costed planner decisions consumed
389,640 tokens and **USD 0.059303952** in total. All 29 recorded traces were
readable through the native tracking API; paid decision traces expose
`mlflow.trace.tokenUsage` and `mlflow.trace.cost`, including the rejected calls.
The final execution snapshot's receipt cost is unknown because it is not an
LLM call; aggregate known spend is separately present and fully resolved.
See `.mega/live-evidence/paid-controller-readback-recovered-final-20261028.json`.

This live test exposed two consumer/validation gaps. Paper actions assumed
predictions lived directly in `trials/ATTEMPT`, whereas accepted supervised
fits keep large artifacts in `.fit-episodes`. The corrected reader verifies
the existing canonical supervisor receipt, admitted request, runtime nonce,
all accepted scientific result fields and artifact checksums. Extra ledger
bookkeeping is permitted; missing accepted fields, redirected artifacts and
replaced runtime state are rejected. A read-only Linux replay of two actual
completed fits passed paper preflight and four hypothetical pool calculations
for one development race, without prices, wagers, model fitting or paid calls.
See `.mega/live-evidence/native-paper-acceptance-recovered-20261028.json`.
Stale proposal/dataset evidence is now checked inside the existing two-call
planner repair envelope instead of failing only after the paid response returns.
The exact-index combined acceptance/publication/planner suite passed 80 tests
plus 62 subtests. Historical rejection and billing records remain unchanged.

The native fold-loading benchmark preserved exact artifact IDs, arrays and
row keys across three full protocol folds, but disproved a universal speedup.
On the actual 171,782-row, 311-column single-row-group Parquet file, the old
path took 28.82 s cold / 17.18 s warm versus 42.18 s / 27.87 s for repeated
fold predicates. Sampled USS was also higher for the latter path. The default
therefore retains full-read preparation for this layout; the predicate path
requires multiple row groups. This condition does not prove that every
multi-row-group dataset is faster. The fallback/projection/delta suite passed
48 tests plus four subtests. The benchmark is retained at
`.mega/live-evidence/fold-row-native-benchmark-recovered-20261028.json`.

The first real learned-pool fit failed because the earliest forward training
block had no speed-trend observations. The corrected serializable transform
state freezes such stateless inputs as inactive and masks their later values;
it does not learn from future blocks. Quantile clipping still refuses an
empty training distribution, and previous nonempty/infinity behavior remains
unchanged. Transform/graph/pool checks passed 59 tests plus 19 subtests.
The sealed retry is a separate attempt and is not yet a successful result.

Linux legacy/objects-v2/CSV-opt-in builder, unchanged refresh, dataset registry
and restored-object consumers passed on small fixtures. Their copied receipts
are hash-indexed under `.mega/live-evidence/native-builder-20261028/`. Independent
review identified a remaining changed-raw-capture retention/provenance readback
gap; that qualification is still open. Full-corpus migration, complete shared
reference adapters, independent confirmation, normal service handover and
whole-plan acceptance are not implied by these bounded checks.
