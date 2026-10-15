# Continuous Feature Exploration Validation

## Deployed Outcome

The own-user `ima-feature-exploration.service` started on 2026-10-15 at
00:11:52 UTC. Scientific code remains pinned to the existing qualified main
campaign release. The new queue/worker release is
`bb5dde3cbd8510d43a869172e0742ee58e297fb5`.

The queue initially contained 33 hypotheses: 32 deterministic definitions and
one already accepted planner proposal containing three definitions. It refills
with unseen definitions below its low watermark; 32 is a refill size, not a
lifetime trial limit. Accepted planner definitions receive priority. Adaptive
bundles of promising screened definitions are also eligible for paired tests.

At the first expanded live readback, five paired screens had finished, another
was running, and 27 remained queued. No candidate failures, rejected builds or
pending tracking deliveries were recorded in that snapshot. The cached control
was trained once. One existing immutable planner-proposed dataset was reused;
four further datasets were built for the deterministic hypotheses.

## Scientific Readback

All screens used the parent dataset's protected three temporal folds, 1,502
development races, fixed `baseline-v1` Benter conditional logit, L2=1, maximum
1,000 iterations, seed 17, no market blend, and training-fold-only fitted
preprocessing. No confirmation data was used. Fixed control log loss:
**2.225278678658062**.

| Feature Hypothesis | Screen Log Loss | Paired Race Delta |
| --- | ---: | ---: |
| Body/carried-weight ratio | 2.225716170 | +0.000437090 |
| Relative distance change | 2.224735644 | -0.000543812 |
| Absolute distance shift | 2.225173729 | -0.000105161 |
| Recent speed deviation | 2.225690571 | +0.000413065 |
| Planner's speed-consistency/recency/distance bundle | 2.223456018 | -0.001816799 |

Negative delta is better. The paired race-weighted delta is not necessarily the
same as subtracting the fold-averaged displayed objectives. The best initial
bundle's race-bootstrap 95% interval was [-0.003822196, +0.000359853], crossing
zero. These are adaptively selected, model-specific development screens, not
confirmed gains over the richer main champion. The main planner must evaluate
reuse under its own matched controls and later confirmation protocol.

The original parent data and cached control predictions were not modified.
Registry outcomes explicitly identify the fixed screening model and warn that
these are not rich-model/champion improvements.

## MLflow And Planner Feedback

- Experiment `ima-feature-builds`: immutable dataset-build runs.
- Experiment `ima-feature-exploration`: one running controller, one finished
  cached control, and separate finished candidate-screen runs.
- Screens link their feature-build and baseline run IDs and log objective,
  paired delta, race count and bootstrap limits. Transport failures remain
  retryable tracking work, not scientific failures.
- Example screen: `0d51cf1f98694e89b5bf7c70c63c30b5`; control:
  `b0e6133dc7d94489a95f99c0bf0b59a3`; its build:
  `2c247a1dba2a46ffb3a46067a554a166`.
- The installed active controller's actual `_capabilities` consumer was loaded
  in an isolated verifier, with its captured execution-loop binding checked.
  It read the new datasets and paired outcomes from the shared registry.
- This consumer readback does not itself prove a new paid planner decision.
  The existing planner's next normal review reads this dynamic registry. At
  the initial readback it was awaiting a decision after an existing duplicate
  proposal rejection; there were no active model fits. Feature work continued
  independently. No extra planner calls were made by this worker.

## Resource And Preservation Proof

The own-user main service retains a **92 GB decimal** hard limit and its
previous 24-core CPU quota. The feature service has **8 GB decimal**, one fit
worker and a two-core CPU quota. Combined hard model/feature memory limits are
100 GB; this does not allocate or modify other users' services.

Before reducing the main limit, the allocator waited for safe committed/current
memory, acknowledged a brief cgroup freeze, rechecked memory, applied the
partition, checked all recorded process IDs and memory events, and thawed in a
`finally` path. Main PID `2695343` stayed unchanged, all recorded model cgroup
processes survived the transition, and both services were running/thawed after
activation. No fits were cancelled. The main fit observed earlier in this task
had finished naturally before activation.

At the first full live proof, main memory was about 7.47 GB and feature memory
about 2.97 GB. Feature peak observed earlier was about 4.61 GB; instantaneous
usage varies while datasets are prepared. Both cgroups had zero `oom` and
`oom_kill` counters. The feature service restores the previous main limit if it
stops; it does not restart the model controller.

## Tests And Review

The focused scientific/resource/worker regression suite passed **78 tests and
12 subtests**. Synthetic fixtures intentionally lack observed `prize` values in
some training folds; scikit-learn's resulting warnings are expected and are not
claims about live dataset quality.

Adversarial coverage includes tainted/unknown units, duplicate ASTs, restart and
single-owner locking, failed candidate isolation, corrupt cached predictions,
expired control retries with fresh durable execution deadlines, tracking outage
and build-outbox recovery, disk pause, unsafe memory partition and thaw on an
injected allocator exception. Real local MLflow runs were read back, not merely
mocked. The read-only reviewer was collected and natively closed after its
three findings were fixed and tested.

GitHub's broader legacy job is **not green**: run `38658576377` reported one
scheduler test failure and 18 import/setup errors. Its environment does not
install pytest, although earlier committed probit/lifecycle tests already import
it. The scheduler failure was independently reproduced from a clean archive of
the pre-feature parent `89a71fd` (`test_large_fit_does_not_serialize_independent_fits`).
These are separate preexisting CI/release issues; they are not represented as
passing or repaired by this feature-worker rollout. The new scientific tests
were executed with the existing local pytest environment, and the server uses
its already-qualified pinned scientific dependencies.

## Receipts And Limits

Server release directory:
`/home/imaopt/research-v2/live-releases/ima-feature-exploration-bb5dde3cbd8510d43a869172e0742ee58e297fb5`.
It contains the sealed manifest, canary, activation and `feature-live-proof.json`
readbacks. Durable queue/status/trials are under
`/home/imaopt/research-v2/feature-exploration/bb5dde3cbd8510d43a869172e0742ee58e297fb5`.
The paired outcomes and reusable datasets are in the main campaign's existing
shared registry. Local operational receipt: `.tmp/feature-exploration-live-proof.json`.

This worker creates typed definitions from eligible existing pre-race inputs;
it does not invent new raw data sources. Its initial cheap screen is win
probability under one fixed model, not automatic validation for every target or
pipeline. Main research retains richer models, target selection, trial budgets
and promotion authority. Vocabulary or disk exhaustion is visible in status;
neither condition is silently called research progress.
