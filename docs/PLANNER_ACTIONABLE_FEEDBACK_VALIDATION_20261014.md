# Planner Feedback Qualification

## Observed Failure
D241 was rejected after two paid responses. Its second proposal declared pipeline_composition only but changed the outer Benter l2 and max_iter; those settings really override the primary graph estimator. This was a valid rejection, not a token/cost telemetry failure.

## Repair
- Guard version 4 explains outer-primary precedence and that fixed_parameters means no search, not unchanged model settings.
- Fixed-control errors contain proposal ID, parent attempt ID, declared axes, changed fields and up to eight leaf-value previews, prioritizing model changes.
- A model-axis hint distinguishes family changes from hyperparameter changes. Repair choices are explicit: restore the undeclared fields or honestly declare a joint test.
- The existing correction message receives these structured errors without its generic 2000-character truncation. Detail is capped at 12000 encoded bytes, with omissions marked.
- Original provider JSON, recipes, admission gates and two-call ceiling are unchanged; no automatic scientific-intent editing.

## Verification
- Final targeted suites: 179 tests and 43 subtests passed; six existing all-missing synthetic prize warnings.
- Additional planner/memo/context/discipline/expansion consumer suite: 96 tests and 57 subtests passed; 12 existing Optuna constant_liar experimental warnings. Combined nonoverlapping targeted coverage: 275 tests and 100 subtests.
- Preserved D241 second response replay gives exact l2 0.001048550555018385 versus 0.041030125528818434 and max_iter 4847 versus 3525.
- Both legitimate corrections pass semantic preflight while retaining all three programs and the original 10-trial budget.
- Actual guarded choose replay accepts restore and joint corrections, rejects repeated-invalid JSON after exactly two calls, and preserves token/cost sums and original input data.
- git diff --check passed. Preexisting dirty/untracked work remains outside this repair.

## Deployment
- Scientific/guard release: 172c06cd21b7b2af3cea51f0a84915571ac6ce01, pushed on fix/planner-actionable-feedback, not merged.
- Sealed own-user stage: /home/imaopt/research-v2/live-releases/ima-probit-feature-172c06cd21b7b2af3cea51f0a84915571ac6ce01.
- Server self-test passed; zero paid calls, zero fits cancelled; 4-hour initial probit policy preserved.
- Dataset/build provenance copied unchanged from the predecessor and explicitly recorded as inherited, not a new feature build.
- Own single-use ima-planner-feedback-handover-20261014.service completed the safe drain and is now inactive, disabled, Result=success.
- At 2026-10-13T15:38:00Z: current fit remained PID 2475466, fold 3/3, 748 completed, 39 failed. No other user/service changed.
- Activated at 2026-10-13T16:40:23.567489Z, controller PID 2499284, zero fits cancelled. The preserved fit completed, increasing completed count to 749 with failed count unchanged at 39.
- D242 was accepted. Both fixed proposals retain the parent outer model exactly: l2=0.001048550555018385, max_iter=4847; changed_axes=pipeline_composition. The planner explicitly acknowledges D241's undeclared model changes in review_reason.
- Planner chose a 6-trial budget, allocated 2 one-trial composition experiments and held 4. These are a three-way Benter/boosted/probit pool and a Benter/probit comparison, not a forced ceiling-size batch.
- Guard receipt D242 records version=4, terminal_watermark=788. Reported D242 input tokens=115545, output tokens=16071, total=131616, cost USD0.017736015. Campaign total USD11.08696395, no unresolved spend.
- MLflow backend get_trace confirms tr-de06c7bea1aa5b9a01ef79f587d3dd67 state=OK, native tokenUsage and total_cost=0.017736015, and source pointing at the reviewed sealed release. Outbox delivery=1, last_error=null.
- At 2026-10-13T16:46:15Z actual training resumed: attempt-d65874104ab3d8e64ed95a1eb0a58559a710e888f955cf5874c38f7b286811cf, fold 1/3, probit component prepared with quadrature_order=48. The primary family label is Benter because this is a composed graph, not a standalone Benter-only trial.

## Independent Review
- Reviewer found missing memory-parent recipe indexing, malformed optional-section iteration and ambiguous parent-ID substring matching. All three were repaired and covered by new adversarial tests.
- Second read-only review of 172c06c found no remaining blocker in this path. The first staged release was superseded before activation; only the handover helper was stopped, not training.
- Whole-worktree scope check reports 4068 changed/generated paths including preexisting dirty work and downloaded .tmp evidence; it is not a clean scope pass. Commits contain only the explicitly planned five owned paths.

## Limitations
Better error feedback does not guarantee provider obedience or better log loss. Future acceptance rate must be observed. Generic schema errors retain their existing reporting. Scientific guard violations remain errors rather than silently accepted experiments.
One live accepted decision demonstrates recovery, not a measured long-term rejection-rate reduction. The best objective has not improved because of this guard repair; new experiments are still running. The accepted trace's preview still has health/blocker=unknown (state is OK); broader telemetry nomenclature is outside this repair.
