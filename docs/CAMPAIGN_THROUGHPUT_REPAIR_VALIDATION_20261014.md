# Campaign Throughput Repair: Live Validation

## Outcome

The qualified resumed campaign now runs independent fast fits alongside slow Gaussian-probit fits. This verifies scheduling progress, not a new statistically confirmed model improvement.

## Proven Cause

The previous controller restarted its capacity ladder at two fit slots. Its global largest successful private-memory observation demanded approximately 120 GiB to increase to four slots, exceeding its 100 GB decimal budget. At cap two, the derived expensive-fit cap was one; all composed graphs were classified expensive. Two funded programs were blocked by this count constraint, despite recent plain Benter/boosted fits requiring approximately 5 GiB.

Pending-experiment duplicate rejection also happened outside the planner's bounded semantic-correction path. The guard now detects pending signatures and duplicates within a decision before downstream authorization, preserving scientific rejection and two-call cost accounting.

## Deployed Identity

- Branch: `fix/workload-aware-throughput`.
- Sealed source revision: `12eda596f51a76d9b9c72dd639c4b9f346ca5c72`.
- Own user: `imaopt`.
- Own unit: `ima-v7-openrouter-recovery-20261007.service`.
- Campaign: `agentic_v7_openrouter_recovery_20261007`.
- Release: `/home/imaopt/research-v2/live-releases/ima-probit-feature-12eda596f51a76d9b9c72dd639c4b9f346ca5c72`.
- Activated: `2026-10-14T16:36:22.275543Z`.
- Request guard version: 6; throughput policy version: 1.
- Original scientific controller revision, dataset, probability objectives and per-job estimator remain unchanged.

## Safety And Limits

- Fit ceiling: 26, subject to each job's reserved CPU, private/shared memory, host/cgroup headroom and disk requirements.
- CPU budget: 24 threads; memory cap: 100 GB decimal, reported as 93.132 GiB.
- Emergency reserve: 8 GiB. Maximum simultaneous cold expensive workloads remains four.
- New/unqualified campaigns retain progressive qualification; five successful uncensored historical measurements qualify a resumed campaign's slot ceiling, not arbitrary job memory estimates.
- No active fits were cancelled. The preserved feature-discovery Benter/boosted fit `attempt-b9dcb1cf42bfaad28013b35f454d8339401a1104f54565959025e1cbb71624c4` completed normally.
- No other user's repositories or services were modified or restarted.

## Live Readback

At `2026-10-14T16:43:04Z`, two Benter/boosted and two Gaussian-probit fits were running concurrently. CPU busy was 26.3%, versus 8.1% in the baseline snapshot. The new policy reported `qualified_resume_per_job_admission`, cap 26. Cold-family admission, rather than a single global graph slot, limited additional dispatch at that snapshot.

At `2026-10-14T16:46:42Z`, a fast fit completed while the slow fits remained active:

- Attempt: `attempt-84a7e35d52de6369f832a37c52ada5cb9cc5089797e556822a8466768a9b1353`.
- Development objective: `2.1494450741188675`; not an improvement over the existing `2.149418422667394` champion.
- Cgroup current: 20.68 GB; peak: 24.43 GB; no OOM or OOM-kill events.
- Durable verifier emitted `LIVE_THROUGHPUT_VERIFIED` and exited successfully.

At `2026-10-14T16:49:43Z`: 774 completed, 42 failed, four running; pending tracking zero, tracking errors empty, latest planner error null. This includes two new fast completions after the preserved fit and handover. It is a short observed interval, not a throughput forecast.

MLflow upload linkage for another completed fast attempt:

- Attempt: `attempt-df6f048d694bf8930d6eaf461948e594a380216fe52917fa07c629d14ffb6f8f`.
- Run: `a97ce907630e44b2a1dc3395a33c01e4`.
- Registered version: 1171 under the existing legacy name `ima-agentic-v6-research-candidates-win-probability`. That registry name does not identify the active campaign version.

## Verification Receipts

71 targeted tests passed across throughput, request guard, resource admission/estimation, lifecycle and probit runtime policy. The actual sealed server canary passed with zero paid calls, unchanged scientific source binding and zero cancelled fits. The live receipt is `throughput-live-proof.json` in the sealed release above; its locally captured SHA-256 is `f90a2b039e6b9f19c29fa277ae099268f1e9910ec710d226a50e685be3e03bbd`.

Scope addition: this validation-only Markdown file records the requested deployment readback without changing the registered canonical plan or protected code.

Deferred, not claimed fixed: broadening measured workload compatibility, redesigning aging/backfill fairness, or eliminating the four-cold-workload guard. None was required to demonstrate the independent fast/slow concurrency repaired here.
