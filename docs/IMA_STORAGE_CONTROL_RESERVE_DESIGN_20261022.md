# Bounded Trace Intent Control Reserve

Status: implementation design for existing obligation 3.1, not a replacement
goal, completed feature, qualified policy or rollout authorization. The
canonical plan remains unchanged. No live provisioning has occurred.

## Observed Gap

The controller currently retains failed pre-enqueue trace payloads in memory,
pauses paid planning, and retries their exact sequence and normalized payload.
One blocked execution snapshot prevents additional execution publications;
one blocked decision prevents another paid decision. This bounds outstanding
runtime intents, but not crash recovery. In-memory retention is not durability.

An independent design review is preserved in
`.mega/close-bernoulli-control-design.json`. Earlier concrete fault reviews are
in `.mega/close-hilbert-tracking.json` and `.mega/close-erdos-tracking.json`.
None approves all of 3.1.

## Measured Inputs

The verified frozen October 20 standalone outbox contains 5,110 rows. Complete
serialized payload maxima are 1,282,920 bytes for 344 decisions, 714,213 bytes
for 4,567 execution snapshots, and 571 bytes for 199 paper-betting traces.
Decision serialized MLflow trace bodies reached 2,885,305 bytes, but those
bodies belong to normal delivery, not emergency intent staging.

`.mega/trace-envelope-measurement-20261022.json` binds these measurements to
SHA256 `77a7ca48bbdc0c0635687dc8d3cf9a7ff726bd31253c47c15ea073f73b1c39e0`.
The file hash was checked before and after readback. This is a preserved
population, not a current live census or a bound on future payloads.

## Selected Direction

Use a small SQLite control store beneath the existing filesystem authority,
with two reusable slots, feeding the existing outbox. Do not add another
delivery engine or replay paid planner calls.

1. Keep normal headroom `H = max(20,000,000,000 bytes, 5% filesystem capacity)`.
   A fixed, measured control allowance `C < H` is included within H. Normal
   writers still require H, not H-C. Only allowlisted bounded control
   transactions may use C; no generic emergency writer bypass is permitted.
2. Hold a distinct stable controller-owner lock, not the authority lock for
   the controller lifetime. Otherwise normal worker reservations would
   deadlock. Exactly one controller owns the filesystem control slots.
   Successor ownership cannot discard another campaign's pending intents.
3. Slot one holds a decision intent. Slot two holds one operational intent
   whose role is execution, dataset or paper betting. Operational roles must
   serialize; a blocked execution must not silently discard a paper trace.
4. Use fixed policy ceilings for complete envelope bytes, database pages,
   peak rollback journal, metadata and recovery scratch. Test with scaled
   limits. Provisional engineering values must not become production policy
   without current full-payload measurements and actual filesystem tests.
5. Reserve the decision slot before paid dispatch. Before normal enqueue,
   commit the complete assembled intent: campaign/revision/policy identity,
   role, exact sequence, destination, schema, frozen payload, hash and execution
   signature. Missing source inputs or overflow block; never truncate them.
6. Commit the existing normal-budget outbox row, verify identity/payload, then
   clear its control slot. A crash between those steps leaves both copies;
   restart replays idempotently. Emergency capacity never emits MLflow spans
   or writes normal summary/model artifacts.
7. Recover pending control state before sequence initialization and paid
   planning. Include occupied intents in sequence maxima and pending status.
   Missing/corrupt/foreign state, destination conflicts and unknown reads
   block; they never mean zero pending.
8. Preserve native usage, receipt and generation identities. A dispatched call
   lacking durable response/accounting is a reconciliation problem. Do not
   claim exactly-once provider execution or retry the paid request blindly.

## SQLite Constraints

Use rollback PERSIST with synchronous FULL for the isolated store, subject to
qualification. PERSIST reuses the journal rather than deleting it after each
transaction. `journal_size_limit` constrains retained size after transaction
completion, not peak transaction growth; `max_page_count` limits database pages,
not all sidecars. Consequently payload/page transaction limits and actual
allocated journal/scratch measurements remain necessary. See the official
[SQLite PRAGMA documentation](https://www.sqlite.org/pragma.html#pragma_journal_size_limit)
and [page-count control](https://www.sqlite.org/pragma.html#pragma_max_page_count).

Bootstrap/preallocation is an explicit healthy-storage operation after
authorized reclaim, not an automatic response to zero free space. Creating a
sparse file or setting a PRAGMA is not proof that an emergency commit can work
when the filesystem is full. Validate returned PRAGMA values on every relevant
connection; unknown PRAGMAs can be ignored by SQLite.

## Alternatives

- Existing outbox emergency writes: fewer files, but its growing WAL, migrations
  and normal delivery writes defeat an independently proven small bound.
- New intent table in the research or decision ledger: reuses SQLite, but
  couples emergency recovery to growing scientific state and its own health.
- In-memory retries only: retained as interim behavior, never crash-safe proof.

## Implementation And Proof Order

1. Implement explicit policy validation, own-user/device/path checks, stable
   owner locking and healthy bootstrap. No automatic live setup or deletion.
2. Implement two-slot immutable intent commit/readback and bounded transactions.
   Admission of normal artifacts remains unchanged. Do not recursively grow
   the JSON authority ledger to record emergency capacity.
3. Integrate normalized trace submission, verified outbox handoff and slot
   clear. Replace the interim memory-only queue when storage mode is enabled.
4. Integrate startup recovery and sequence allocation, paid-dispatch slot
   admission, blocked diagnostics and auxiliary-role serialization.
5. Kill/restart after intent commit, outbox commit, receipt persistence and
   before slot clear. Prove identical payload/sequence and no duplicate spend.
6. Race controllers/campaigns. Test payload overflow, inode pressure, SQLite
   FULL, failed commit, corruption, changed destination, unknown count and
   repeated retries without growing histories/files or overwriting slots.
7. Independently review actual consumer behavior and all source inputs.
   Qualify allocated DB/journal/scratch on the target filesystem after reclaim.
   Keep all normal writers held until the remaining 3.1, 3.2 and database
   recovery gates pass.
