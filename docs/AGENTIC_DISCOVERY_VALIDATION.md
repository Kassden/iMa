# V5 Validation Evidence

## Verified 2026-10-01

- Full local regression suite: 279 tests passed (71.2 seconds). Focused final
  scheduling/executor/controller tests: 44 passed, including durable fairness.
- Coverage includes future-data/current-result exclusion, delayed observations,
  missing identities, invalid specifications, cache corruption, temporal selection,
  all five lane contracts, package replay, duplicate accounting, crash/recovery,
  planner errors/deadlines, budgets independent of concurrency and spend boundaries.
- Local tracking canary: 20 completed attempts, zero pending uploads/tells,
  versioned models, true source/matrix inputs and trace readback.
- Dedicated server canary C: 20 completed, 20 registered model versions, all five
  model kinds, zero failed/pending work; package probability replay maximum error
  `9.8879238130678e-17`. Race probability sums equal one to numerical precision.
- Dedicated server canary E (revision `9397432`): 20 completed, 20 model versions,
  all five models, zero pending uploads/tells; package replay maximum error
  `9.71445146547012e-17`. Selected run `7513a5c2ecbc4932bbc01290b1fc505b`
  has two MLflow dataset inputs. The experiment contained 30 traces at readback.
- Canary E's planner chose budgets including 8 and 6, demonstrating neither a
  mandatory ceiling nor a budget tied to the two-worker canary limit. Decisions and
  fresh evidence are preserved in its campaign. Genuine OpenRouter responses are
  journaled; no fixture planner or bootstrap claimed as live agentic work.
- Final pinned revision `0915b619f6a40dcc390c3044107831293f826660` canary:
  20 completed, 20 linked model versions, all five lanes, no failed or pending
  uploads/tells. Registered URI
  `models:/ima-agentic-v5-discovery-candidates-win-probability/56` loaded successfully
  and produced exactly the package's probabilities (maximum error zero); saved CSV
  replay error was `9.71445146547012e-17`. Re-running the completed campaign exited
  successfully without dispatching more trials.
- Chromium inspected the actual MLflow traces table: planner USD previews and
  meaningful cycle/campaign bests are visible. Screenshot `.tmp/v5-mlflow-traces.png`.
  The UI's 4.4 MB JavaScript asset took 44 seconds over the observed network, exceeding
  initial browser navigation timeouts. Browser verification used a temporary SSH
  relay with a valid original Host header; server DNS-rebinding protections were
  not disabled. Cost previews are supported; no fabricated native cost column.
- Failed canaries A/B are retained: disconnected discounted provider routing and
  missing isolated `httpx` respectively. Plain-model routing and v5-only dependencies
  corrected those proven causes without changing v4 or shared packages.

## Scientific Interpretation

The fixed-model synthetic fixture shared the same feature definition across Benter
and boosted controls. Benter mean paired loss delta was approximately `-0.0016644`,
with confidence interval including zero: inconclusive. Boosted delta was about
`+0.561845`: reject for that fixture. These are tests of scientific bookkeeping,
not historical accuracy claims or evidence of profitability. Repeated development
search creates selection exposure; final confirmation remains protected.

## Build and Scheduler Gates

The representative Featuretools benchmark retained all 271,858 historical rows,
evaluated 10,000 representative cutoffs and produced 372 columns. Wall time was
799.8 seconds, process peak RSS 2.147 GiB, matrix memory 0.0278 GiB. It used strict
cutoffs and no scored labels. The subsequent actual full build evaluated all
271,858 runner cutoffs with the same 372 columns: 857.14 seconds, process peak
2.147 GiB, matrix memory 0.7535 GiB. Reports are preserved under the production
campaign's `ops/build-benchmark/{representative,full,report}.json`.

The isolated pool benchmark completed all four tasks in 0.4947 seconds and
replenished short work ahead of its long task (completion order: .05, .1, .08, .4
seconds). Ray Tune 2.49.0 completed the same task set in 7.7024 seconds including
initialization. Its first attempts exposed missing optional fsspec and an overlong
Unix socket path; the separate benchmark environment and short temporary directory
resolved those issues. Production dependencies were not changed. Report:
`ops/scheduler-benchmark-c/comparison.json`. The persistent spawn pool was selected
for the existing single-host/single-writer architecture. No predictive-throughput
claim follows from a sleep-task queue benchmark.

## Production Handoff Gate

The user explicitly authorized execution and v5 rollout on 2026-10-01, superseding
the earlier planning-only instruction. V4 received a graceful STOP request. It
must reach zero running ledger attempts before disabling its supervisor. V5 must
then demonstrate completed real-history trials, a subsequent evidence-aware
OpenRouter decision, tracking/model readback and protected shared-service health.
Canary success alone does not satisfy this gate.

## Verified Production Handoff

At 2026-10-01 16:00:21 Asia/Shanghai, v4 finished with 1,005 completed and zero
running attempts, emitted `mode=stopped` and exited successfully. Its supervisor
was disabled. The guarded server-side handoff then enabled and started v5 at
revision `0915b619f6a40dcc390c3044107831293f826660`. Receipt:
`agentic_v5_discovery/ops/handoff.json`. No trial was killed; no shared app restarted.

Production reached 16 in-flight attempts under its CPU/RAM reservations. A 30-second
host CPU sample averaged 14.9%; service RSS sum was about 19.7 GiB with 94.2 GiB
available host RAM. Shared cache builders explain initial idle waiting workers;
in-flight count is not a claim that all 16 CPUs are busy simultaneously.

The first generated-feature boosted attempt completed with fundamental race log
loss `2.19732089830168`, MLflow run `c48cbd2f707c4dd8bce80f00f2502c03`, registry
version 57, two dataset inputs and 72 generated candidates. Registered-URI replay
matched exactly; CSV probability replay maximum error was `9.974659986866641e-17`.
The first LambdaRank attempt completed at negative NDCG@3 `-0.7527063491756949`,
run `2655d247edcd44b68455c7b6ee642ac6`, registry version 4. These early scores do
not improve on the compatible v4 champions; software validation is not accuracy
improvement. Compatible v4 references are retained read-only (boosted loss 2.15972,
Benter loss 2.18169, ranking negative NDCG -0.75626).

The controller replenished completed slots while other attempts continued. Its
next frozen evidence included two evaluated hypothesis events and the new feature
manifest/selection reports. The genuine OpenRouter follow-up chose 12 trials,
approved 12, and proposed revised discovery recipes, a control, and the missing odds
lane. The earlier successful decision chose 48 and approved 42. Neither budget is
a concurrency multiple or a mandatory 260. Evidence ID:
`1d75a2569d5ab332d8942a91`; `ops/live-proof.json` records API/ledger readbacks.

The initial production discovery proposal was rejected by the pre-existing
conservative narrative term guard (benign wording containing "results"). Its
billed response and failure were preserved, and the next agent decision corrected
the request and launched valid work. This is a documented false-positive limitation,
not evidence of target leakage. No silent bootstrap or pinned-source hotpatch was
used. Current planner/tracking errors were empty, pending tells were zero, and the
supervisor had zero restarts at verification.

### Extended Live Observation

Cycle 3 exhausted its 12,000-token proposal output allowance (9,938 reasoning
tokens), leaving truncated JSON; its bounded repair request then failed with a
connection ReadError. Both the billed response and the decision failure remain
recorded. Training continued independently. Cycle 4 subsequently recovered
without a restart, chose and approved 24 trials, and cleared the current planner
error. An isolated medium-reasoning/24,000-output-token probe also encountered a
ReadError, so it did not justify changing the live configuration.

At 16:51 Asia/Shanghai, production had three completed attempts and 16 in flight,
with zero pending tracking records or Optuna tells. The recorded-odds attempt
completed at log-final-odds MAE `0.47479374169465055`, MLflow run
`29173f24fcea4b05acb848b5e2358823`, registry version 4. This does not improve on
the compatible v4 reference `0.422614132517678`.

The first Benter historical-feature cache was still building, not yet a completed
primary-path validation. Most Benter workers waited on its shared cache lock.
A read-only timing probe over 256 identities in the 271,858-row source measured
4.105 seconds for the repeated full-column identity queries. Extrapolating those
queries to 12,012 horses and 16 sequence passes gives approximately 3,082 seconds
for the scans alone, excluding Featuretools and other work. This is a timing
estimate, not a full-build benchmark or proof of completion. Pre-indexing query
rows per identity is a follow-up performance opportunity; the pinned live source
was not hotpatched. Zero failed trials and active CPU do not by themselves prove
that a Benter model has finished.

The primary cache subsequently completed with 225 candidates in 3,920.567 seconds
(65.34 minutes), allowing its waiting workers to enter training. Sixteen concurrent
full-history jobs then sustained soft-limit pressure: service memory reached
84.67 GiB under the original 80/88 GiB soft/hard limits. Hard-limit hits and OOM
kills were both zero. The dedicated v5 service limits were raised, without restart,
to 88/96 GiB and persisted in its own user unit; the source unit passed
`systemd-analyze --user verify` with `XDG_RUNTIME_DIR=/run/user/1001`. At subsequent
observation service memory was 89.74 GiB, hard-limit hits/OOM kills were still zero,
the PID remained 756004 and restart count remained zero. This is an operational
resource-limit adjustment, not a model-code hotpatch. The 80 GiB scheduler budget
still measures declared reservations rather than actual worker RSS; per-model
measured memory admission remains a limitation, not a claimed adaptive capability.

All protected services remained active: `cortex-web`, `cortex-worker`, and
`solar-simulator`. `imaopt` has `Linger=yes`; v5 is enabled and does not require the
Mac or SSH session. Unlimited total trials remain subject to the explicit $5 planner
spend pause, operator stop and failure safeguards documented in the runbook.
