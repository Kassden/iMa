# Planner Preflight Recovery Validation

## Incident and Repair

D239 stopped planning at 304,084 > 300,000 context bytes before any provider call. Missing pre-dispatch events incorrectly made that local failure an unknown charge. A separate D238 graph experiment selected adjusted-speed residual histories despite the absence of a nested residual-fit contract.

The repair uses the existing guard, typed recipe validation and [HTTPX tracing](https://www.python-httpx.org/advanced/extensions/). It does not raise the hard limits or change models, evaluation datasets, bankrolls or wagering rules.

- Adaptive projection targets 90% of the configured context limit. Older optional narratives are reduced and exact duplicate family recipes become explicit references to full contract-champion recipes. Parent IDs, availability/taint metadata and scientific contracts remain intact; full historical evidence remains stored.
- An adversarial shared-object test found that duplicate champion dictionaries could alias each other. The final fix replaces the family row with a distinct copy, leaving the canonical recipe untouched.
- The actual choose/post boundary emits a matched, nonce-bound `request_started` / `request.preflight_failed` pair only if the underlying HTTP function has never been entered. Request-size checks happen before incrementing the dispatch counter.
- The existing classifier recognizes only that complete guard-produced pair as locally unsent. Missing/truncated/contradictory records remain unknown, and any HTTP send/response evidence takes precedence. An error after the first paid call cannot waive that call's cost.
- Semantic preflight rejects adjusted-speed residual graphs inside the existing bounded correction loop. The planner sees the constraint in its request contract; valid non-graph alternatives remain available.
- Historical D239 recovery is separately bound to independently inspected decision/evidence hashes. Original failure/provider records are not rewritten. Future local failures use actual boundary events rather than per-decision exceptions.

## Tests and Qualification

Final targeted suite: **218 tests and 120 subtests passed**, covering guard/context, transport classification, billing consumers, worker deadlines, warm starts, feature lifecycle and real feature consumers. Six expected warnings come from an all-missing synthetic prize column.

Adversarial tests cover oversize mandatory safety, growing optional history, idempotent projection, aliased champions, unchanged taint/parent identities, zero-post local failure, post-entry timeout, sent siblings, truncated/forged event identities and residual graph rejection. The reviewer withdrew a stale ordering finding after confirming that the byte check precedes dispatch accounting.

Exact D239 server replay: **277,415 bytes**, including receipts. Omissions: three older hypothesis records, one older failure record and three duplicate family recipes. The hard context limit remains 300,000 bytes; request limit remains 380,000 bytes.

Sealed predecessor guard/allocation parity, dataset identity, existing feature build, probit floor and billing readbacks passed. The final scientific release is `c41464404086292818f7d11e74e3e9d67697f408`, staged under the own-user live-release directory. Its manifest SHA256 is `455883accc71e15ccd529e616ea584f4952b303aa0b3e636fe6217e264dc0a38`.

## Live Recovery

The core repair release `88b5f0b09e70f4c2433e3d52a44837511f05bb6f` activated at 2026-10-13 12:45 UTC. D240 was accepted with four trials after two bounded physical calls. Reported planner cost: USD 0.027691189; known and total accumulated spending: USD 11.060601548, no unresolved decision IDs.

A valid Gaussian-probit component began fitting within the selected pipeline. Its optimizer telemetry showed 35 iterations and training objective improvement from 2.2614348 to 2.1966080; it subsequently reached fold 2. These are training diagnostics, not new development-validation scores or a champion improvement.

The final alias-safety release is waiting for that productive fit to finish. The own-user one-shot `ima-preflight-handover-20261013.service` owns the graceful wait, survives SSH disconnection, and disables itself after successful activation. Only the foreground handover helper was terminated while moving the wait into systemd; the training PID and fit were preserved. No model fits were cancelled. The original campaign is intentionally draining during this short safe handover.

Final release `c414644` activated at 2026-10-13 13:15:33 UTC. The one-shot handover unit finished successfully and disabled itself. No fits were cancelled. Fresh post-handover readback at 13:23:48 UTC shows training, 748 completed / 39 failed / one running, zero pending tells/tracking and no unresolved charges. Known and total spending match at USD 11.069227935.

Preserved Gaussian trial `attempt-3dccd98bb0a31cdfcee67962e4bb5f6f5111f86f128d2c99ce270c3c00555adb` completed successfully in 1475.199247 seconds (24.6 minutes), with three probit fit profiles and development fundamental race log loss 2.161712945474768. Its runtime receipt proves initial allowance 14,400 seconds and hard cap 21,600 seconds. A 20-minute deadline would have interrupted this trial. This score does not beat the approximately 2.14942 champion.

Trial `attempt-c70ea85e4d6471ed3dbff563cadefccaa4b76b582c6ee20904707e81e197b075` is running after the handover, proving resumed execution rather than merely an active service.

Latest subsequent planner decision D241 was rejected for a legitimate undeclared model change, not context size or unknown billing. Its composition-only fixed control changed outer Benter regularization from 0.001048550555018385 to 0.041030125528818434. The executor explicitly overlays outer model parameters onto the primary graph estimator, so this is a real training change, not harmless metadata. The guard must reject that matched-control claim rather than normalize it away. Reported D241 cost is USD 0.008626387; accepted D240 work continues. Planner proposal quality remains imperfect; this repair does not guarantee every proposed hypothesis is valid.

## Boundaries and Limits

MemoryMax remains 100 GB and CPU quota remains 24 CPU-seconds/second. Other users, repositories and services were untouched. Dirty workspace files were preserved and only owned commits were pushed to `fix/planner-context-predispatch`; main was not merged. The global scope checker reports thousands of preexisting/generated paths, so it cannot claim a clean whole-worktree pass.

Projection cannot discard genuinely mandatory safety metadata to meet a cap. If mandatory data itself eventually outgrows the hard limit, it still fails closed with a truthful unsent receipt; further pagination/selection would need an explicit contract. No claim is made that unbounded history fits forever, that provider availability is guaranteed, or that this repair improves prediction accuracy.
