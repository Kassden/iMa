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
- Preserved D241 second response replay gives exact l2 0.001048550555018385 versus 0.041030125528818434 and max_iter 4847 versus 3525.
- Both legitimate corrections pass semantic preflight while retaining all three programs and the original 10-trial budget.
- Actual guarded choose replay accepts restore and joint corrections, rejects repeated-invalid JSON after exactly two calls, and preserves token/cost sums and original input data.
- git diff --check passed. Preexisting dirty/untracked work remains outside this repair.

## Deployment
- Scientific/guard release: 172c06cd21b7b2af3cea51f0a84915571ac6ce01, pushed on fix/planner-actionable-feedback, not merged.
- Sealed own-user stage: /home/imaopt/research-v2/live-releases/ima-probit-feature-172c06cd21b7b2af3cea51f0a84915571ac6ce01.
- Server self-test passed; zero paid calls, zero fits cancelled; 4-hour initial probit policy preserved.
- Dataset/build provenance copied unchanged from the predecessor and explicitly recorded as inherited, not a new feature build.
- Own single-use ima-planner-feedback-handover-20261014.service is enabled and running its safe drain; it disables itself after activation.
- At 2026-10-13T15:38:00Z: current fit remained PID 2475466, fold 3/3, 748 completed, 39 failed. No other user/service changed.
- Activation and next live planner decision are still pending. This is staged and qualified, not yet proven live-effective.

## Independent Review
- Reviewer found missing memory-parent recipe indexing, malformed optional-section iteration and ambiguous parent-ID substring matching. All three were repaired and covered by new adversarial tests.
- Second read-only review of 172c06c found no remaining blocker in this path. The first staged release was superseded before activation; only the handover helper was stopped, not training.
- Whole-worktree scope check reports 4068 changed/generated paths including preexisting dirty work and downloaded .tmp evidence; it is not a clean scope pass. Commits contain only the explicitly planned five owned paths.

## Limitations
Better error feedback does not guarantee provider obedience or better log loss. Future acceptance rate must be observed. Generic schema errors retain their existing reporting. Scientific guard violations remain errors rather than silently accepted experiments.
