# Consolidated Plan Review

Artifact: docs/IMA_RECOVERY_STORAGE_AND_CHAMPION_PLAN_20261020.md.
Reviewed SHA-256: b0384addf49223492c75d6d5319e3efaf597511007924d60036a6dbd0d9013ae.
Independent reviewer: Gibbs, session 01a15723-59db-7b70-87cc-04602ca3fd2b.
Date: 2026-10-20. Scope: planning only.

## Initial Findings
The reviewer compared the consolidated plan against the original champion plan, its independent review, and the storage audit. Three corrections were required:
1. Reservation qualification alone must not authorize normal restart before bounded-cache qualification.
2. Database recovery needs explicit consistent WAL-inclusive capture, integrity checks and application readback on healthy storage before affected writers resume.
3. Scheduler acceptance must explicitly prove identical preparation is built once and eligible queued work fills freed capacity across programs without budget/concurrency coupling.

The parent incorporated these changes, added an obligation crosswalk, and expanded admission integration to all controlled artifact writers. Earlier audit/plan evidence and unapproved candidate manifests were not changed.

## Final Native Review
> Qualified approval of planning only. I reread the current canonical file and found no remaining substantive omission, safety conflict, or scientific sequencing blocker.

The reviewer verified:
- Subphase 3.1 holds normal writers until 3.2 and dependent database recovery gates pass.
- Database capture rejects unrelated live DB/WAL copies and requires consistent healthy-copy verification.
- Subphase 3.2 includes build-once and cross-program refill tests.
- iMa-only boundaries, conditional 153.509 GiB reclaim, held historical snapshots, protected artifacts, all-writer admission, compatible data formats/deltas, matched comparisons, selection-exposure controls and prospective seals are retained.
- The dated champion reference is 2.149415373890447 and must be freshly resolved before execution.

The final hash above was independently computed by the reviewer from the current file.

This approves substantive planning, not implementation, current host health, deletion readiness, actual reclaimed space, or scientific improvement. No reviewer edits, SSH, service changes or children were created. The parent collected the result and is responsible for native closure and artifact readback.

## Parent Validation
The plan checker passed. The artifact contains nine phases and sixteen subphases with future checklists, explicit files/tests/commits/handoffs, a single dependency order, preserved source evidence and conditional cleanup boundaries. A generated dashboard is a read view, not an independent execution plan. No implementation checkbox is marked complete.
