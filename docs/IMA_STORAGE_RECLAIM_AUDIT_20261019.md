# iMa Storage Reclaim Audit

Date: 2026-10-19. **Audit only: nothing purged, moved, deduplicated or restarted.**
Evidence capture: 2026-10-19T11:07:34.287699+00:00 through 2026-10-19T11:07:49.525290+00:00; detailed payload capture 2026-10-19T11:15:37.602011+00:00.
Scope: readable own-user iMa acquisition, research, metadata and MLflow consumers. No other-user cleanup.

## Recommended Reclaim
The first-wave candidate proposal identifies **153.509 GiB** of allocated blocks: reproducible fold-preparation payloads plus conditional unpublished incomplete-build outputs. Abandonment of the latter is not certified. This is NOT deletion readiness or permission to delete now. Own auto-restart/timers must first be quiesced, open files/locks rechecked, deployed-source/package behavior verified and source/model references preserved. All manifest entries have approved=false.

| Disjoint category | Allocated GiB | Allocated bytes | Interpretation |
| --- | ---: | ---: | --- |
| Fold-preparation artifact payloads | 152.007 | 163,216,367,616 | First-wave reproducible intermediates; retain stable root lock/owner files |
| 20 conditional incomplete `.building` outputs | 1.502 | 1,612,935,168 | Unpublished, no collected reference/open consumer; abandonment unverified, recheck producer/preservation |
| Other derived cache/program artifacts | 5.954 | 6,392,770,560 | Conditional only, excluded from first-wave recommendation |
| 315 historical snapshot directories | 324.885 | 348,842,921,984 | Conditional archive/reconstruction review, NOT certified disposable |

The first two rows sum to 164,829,302,784 allocated bytes (153.509 GiB). Including the conditional snapshot/archive scenario would raise that to 478.395 GiB, but that larger figure is NOT safe-purge space. Archive must be on a different filesystem/storage target and hash-verified; moving files elsewhere on this same full filesystem frees nothing.

## Why The Fold Caches Qualify
- They contain derived numeric matrices, fitted preprocessing, selected-feature state and manifests, not the final fitted horse models. Existing cache-miss code rebuilds them from retained datasets/recipes/code/environment.
- The scan found no matching own-user open file or mapped consumer, no external hardlinks or symlinks in these proposed payloads. This is a timestamped observation, not a guarantee against a future auto-restart.
- The saved ResearchModelPackage serializes its fitted model and feature replay state. Existing fixture tests passed for package predictions after synthetic fold-cache eviction: **2 passed, 24 deselected**, run locally, not on the server.
- Stable `<artifact_id>.lock` and root owner/control files are deliberately EXCLUDED. Future cleanup should use existing lock-aware eviction semantics, not delete cache roots or recreate lock inodes.
- Persisted preparation descriptors and scheduler leases require review/invalidation or rewarming before resuming dependent work. Full historical-package replay was not performed; do not generalize the fixture to every historical model without the final canary.

Payload counts: 3,573 exact artifact-ID directories across 14 cache roots. Retained root scaffolding allocation: 14,536,704 bytes. Kernel lock metadata check recorded 0 hits in the observed root-level files; no locks were acquired or changed by this audit.

Paths below are relative to `/home/imaopt/research-v2/`. The JSON manifest lists each exact payload path, not a deletion wildcard.

| Cache root | Reclaimable payload GiB | Payload bytes |
| --- | ---: | ---: |
| `feature-exploration/bb5dde3cbd8510d43a869172e0742ee58e297fb5/fold-preparation-cache` | 89.652 | 96,262,656,000 |
| `campaigns/agentic_v7_openrouter_recovery_20261007/fold-preparation-cache` | 32.408 | 34,797,727,744 |
| `campaigns/agentic_v6_reliability_0a50bf4/fold-preparation-cache` | 11.828 | 12,700,184,576 |
| `campaigns/agentic_v7_probit_runtime_efe1f6b/fold-preparation-cache` | 9.578 | 10,284,027,904 |
| `campaigns/agentic_v6_openrouter_ready_8d61102/fold-preparation-cache` | 2.933 | 3,149,570,048 |
| `campaigns/agentic_v6_openrouter_continuous_4864156/fold-preparation-cache` | 2.439 | 2,618,712,064 |
| `campaigns/agentic_v6_luna_session_20261004/fold-preparation-cache` | 1.224 | 1,314,381,824 |
| `campaigns/agentic_v6_official_controls_c1_parent/fold-preparation-cache` | 0.750 | 805,203,968 |
| `campaigns/agentic_v7_probit_runtime_verification_efe1f6b/fold-preparation-cache` | 0.518 | 555,687,936 |
| `campaigns/agentic_v6_openrouter_7da5b0d_r2/fold-preparation-cache` | 0.510 | 547,934,208 |
| `campaigns/agentic_v7_probit_runtime_verification_3b51e19/fold-preparation-cache` | 0.166 | 178,302,976 |
| `campaigns/agentic_v6_reliability_science_0a50bf4/fold-preparation-cache` | 0.001 | 659,456 |
| `campaigns/agentic_v6_reliability_science_e45cb21/fold-preparation-cache` | 0.001 | 659,456 |
| `campaigns/agentic_v6_reliability_science_f6ec04f/fold-preparation-cache` | 0.001 | 659,456 |

## Incomplete Snapshot Builds
These are `.building` directories, not the shared `raw-import` directory. There is no completed snapshot manifest, no collected reference and no observed producer/open-file consumer for them. Their stored copies can contain useful diagnostic evidence, so archive any required failure details and verify no producer will reuse them before removal. Current builder refuses an existing unfinished path; it does not automatically resume one.

| Directory under acquisition snapshot root | Allocated bytes |
| --- | ---: |
| `snapshot-20261016T024556Z-0ab327f1.building` | 1,131,995,136 |
| `snapshot-20261016T034817Z-7ab412bb.building` | 134,336,512 |
| `snapshot-20261016T065251Z-a5f7d123.building` | 22,581,248 |
| `snapshot-20261016T075500Z-5bd0948d.building` | 155,648 |
| `snapshot-20261016T085604Z-b963aeca.building` | 155,648 |
| `snapshot-20261016T095918Z-623b35f2.building` | 155,648 |
| `snapshot-20261016T105922Z-1b4e42fb.building` | 155,648 |
| `snapshot-20261016T120339Z-a94bf258.building` | 155,648 |
| `snapshot-20261016T141117Z-d703358f.building` | 155,648 |
| `snapshot-20261016T151500Z-51ce4c9c.building` | 155,648 |
| `snapshot-20261016T161905Z-a7958c7c.building` | 155,648 |
| `snapshot-20261016T172403Z-15c5821d.building` | 155,648 |
| `snapshot-20261016T182841Z-63279c8c.building` | 155,648 |
| `snapshot-20261016T213045Z-3fb413ab.building` | 114,688 |
| `snapshot-20261016T223125Z-879df8bf.building` | 155,648 |
| `snapshot-20261016T233223Z-98f21ec5.building` | 12,460,032 |
| `snapshot-20261017T003637Z-d3b6618a.building` | 19,607,552 |
| `snapshot-20261017T115849Z-cf8b173c.building` | 19,918,848 |
| `snapshot-20261017T171104Z-5e5a9143.building` | 155,803,648 |
| `snapshot-20261017T181127Z-32d9c028.building` | 114,405,376 |

## Historical Snapshots: Large But Not Blanket Garbage
There are 319 completed snapshots, occupying 328.708 GiB. The current acquisition refresh reads latest_verified and verifies that complete snapshot. Old base_snapshot fields are provenance after publication in the inspected read paths, but an explicitly requested future rebuild can still consume an old base.

The 324.885 GiB conditional category contains 315 older directories. These are not current direct optimizer source inputs in collected configuration/registry/ledger evidence. However, revisions are not all byte-identical, historical base references exist, and source manifests/old requests can depend on the retained anchors. Missing a reference in this collector is not proof that no hidden consumer exists. Preserve unique revisions off-filesystem or prove reconstruction before considering this category removable. Keep manifests, captured code, lineage, readbacks and raw evidence in the archive.

Conservatively retained full snapshots:
- `/home/imaopt/acquisition/hkjc-20261002/snapshots/snapshot-20261004T040303Z-a98503aa` (0.716 GiB).
- `/home/imaopt/acquisition/hkjc-20261002/snapshots/snapshot-20261004T050721Z-122062e8` (0.723 GiB).
- `/home/imaopt/acquisition/hkjc-20261002/snapshots/snapshot-20261016T004049Z-943994ec` (1.192 GiB).
- `/home/imaopt/acquisition/hkjc-20261002/snapshots/snapshot-20261016T014541Z-c5ec0bbf` (1.192 GiB).

Also retain `latest_verified.json`, readback receipts, `raw-import`, the configured baseline pair `imports/local-snapshot-008/{runners.parquet,manifest.json}`, and raw source archives/profile records used by the scheduled snapshot producer. Snapshot verification opens raw blobs outside individual snapshot directories; deleting shared raw-import breaks verification.

### Optional Deduplication Alternative
Snapshot manifests claim repeated exact file hashes corresponding to **51.242 GiB** of potential duplicate allocation. This is an alternative scenario overlapping historical-snapshot cleanup, NOT an additive saving. Payload hashes were not all independently recomputed, so verified deduplication savings remain zero. A future validated immutable coalescing design could preserve all historical paths; it requires byte verification, writer safety and qualified inode/link semantics. No deduplication was attempted.

## Keep: Still Used Or Scientifically Necessary
- `/home/imaopt/iMa`: this is NOT an unused old checkout. Its environment runs the LIVE MLflow server on port 5000. Database inventory: 26,739 runs, 7,449 model versions, 17 experiments. Preserve its database/WAL, artifacts, models and environment.
- `/home/imaopt/research-v2/campaigns/agentic_v6_research/datasets`: 1,061 registered datasets, with 1,059 descendants of the shared parent. The registry is used by current configuration and contains historical research lineage; per-dataset removal/reconstruction safety was not established. This is NOT a claim that every child is actively training. V6 in the path does not make the registry inactive. Its approximately 95 GiB is NOT included in recommended reclaim.
- Every campaign's `trials`, saved packages, predictions, decisions, ledgers, confirmation data and source manifests. A failed/stopped campaign can still hold the champion or its reproducibility evidence.
- Official raw corpus, snapshot raw-import, original acquisition/profile inputs, current and historically pinned code/dependencies. No source/release/virtualenv cleanup is certified by this audit.
- Discovery caches, graph-prediction caches and prepared-program descriptors remain conditional: they can be tied to resume, feature attachment or performance guarantees. Preserve useful probit warm starts; they are tiny and expensive to recreate.

Do NOT selectively remove CSV files from a manifest-bound dataset/snapshot merely because Parquet exists. Executors/verifiers hash every listed file; that can break consumers even when a column table seems redundant.

## Order Of Any Future Approved Cleanup
1. Fresh own-process/service/timer/lock readback; separately authorized quiescence, not this audit.
2. Preserve latest source/model/MLflow and run a saved-package inference canary independent of disposable caches.
3. Evict only eligible fold payloads through stable-lock semantics and remove independently rechecked unfinished builds. Keep all audit metadata and confirmation/source/model records.
4. Verify actual filesystem free bytes, model inference, registry integrity and safe resumption; rewarm invalidated descriptors as required.
5. Consider conditional snapshot archival or deduplication separately, with verified off-filesystem backup/reconstruction and a new approval.

Cleanup alone does not fix growth. The active snapshot timer runs hourly, recent large completed snapshots are about 1.19 GiB each, and preparation caches currently lack automatic retention. Future retention/admission work is covered by the pipeline improvement plan, not implemented here.

## Evidence, Limits And Readback
- [Machine-readable exact-path manifest](../.tmp/ima-storage-audit/manifest.json): all approved=false; integer allocated-byte totals and non-overlapping categories.
- [Captured inventory](../.tmp/ima-storage-audit/evidence.json) and [payload/lock details](../.tmp/ima-storage-audit/details.json).
- Source contracts: `ima/research_preparation.py` pin/evict; `ima/research_executor.py` cache construction/package save; `ima/research_model_package.py` self-contained replay; snapshot build/refresh/verify and dataset registry code.
- Deployed-source equivalence is NOT certified: `details.source_files` is empty because the queried overlay/base paths did not expose those modules. Local fixture tests support synthetic cache eviction/package independence only; no production-package inference canary or every-historical-model verification ran. This is an explicit prerequisite before action, not proof the deployed behavior differs.
- No read/write database connection, purge, service mutation, cache-lock mutation, remote script file, archive write or remote filesystem repair was performed. Acquisition/MLflow may continue their own activity independently. One timed-out own read-only audit collector was terminated after verifying its exact PID/UID/command/age; no training or service process was stopped.
- Byte values are allocated block estimates. Directory metadata, concurrent writers, open-unlinked files or hidden external links can affect actual free-space changes; remeasure at action time. Other-user usage is outside this proposal.
- The evidence is point-in-time. Independent review findings and manifest-validation results are recorded separately; this report does not authorize implementation of the older pipeline plan.
