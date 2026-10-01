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
cutoffs and no scored labels. Full-history output is a separate gate and must not
be inferred from this sample.

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

Actual production identity, full-build/scheduler outcomes and live advancement
evidence are appended after verification; this document does not claim those
pending gates have passed.
