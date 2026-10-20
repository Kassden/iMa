# Recovery And Champion Implementation Record

Date: 2026-10-20. Task: `ima-recovery-storage-champion-20261020`.

## Verdict

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
| 1.1 | Scope, copies and package inference captured; complete recovery qualification pending. |
| 2.1 | BLOCKED: incomplete own maps and denied privileged access; zero reclaimed bytes. |
| 3.1 | Primitives tested; all-writer integration and infrastructure retry unfinished. |
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
