# Planner Acceptance And Cost Repair Validation

## Proven Causes

1. D200-D203 finished and were charged by OpenRouter, but campaign admission rejected
   their empty per-proposal evidence IDs. Their top-level request identities were correct.
   These four failures cost USD 0.22552578. This was local admission failure, not a
   provider refusing all requests.
2. Live D205 exposed a second false rejection: a matched parent's explicit registered
   dataset ID and the proposal's default current dataset were compared as different.
   The configured and registered data bytes and protocol parameters are identical.
3. D207 made a genuine undeclared transform change. This semantic error was checked
   only after the bounded planner repair loop returned. Move provable declaration
   errors into that existing correction loop, rather than silently permitting them.
4. One previously admitted learned-feature graph failed on `catboost_classifier`, which
   its fold-local executor does not support. That exact combination is now preflighted
   and explained to the planner; standalone CatBoost is not disabled.

## Repair

- Bind only omitted/empty proposal evidence IDs after checking exact decision/evidence
  identity. Explicit stale IDs and unknown parents remain invalid.
- Canonicalize the verified default dataset only in audit comparison copies. Actual
  execution recipes, provider responses, trial parameters and scientific release stay intact.
- Preserve full evidence and receipts; summarize bulky historical searches, remove exhausted
  capacity entries and redundant runtime paths. Essential context is not silently truncated.
- Apply hard pre-dispatch limits: 300,000 UTF-8 JSON bytes of context and 380,000 bytes
  for the full outbound payload, including any schema-repair candidate.
- Persist guarded request identities. Two identical semantic rejections at an unchanged
  result watermark block further paid calls until new results or a qualified policy change.
- Keep native USD/token accounting, reconciliation and historical ERROR traces truthful.
- Preserve the operator's compiled free-allocation prompt. Sealed-source reconstruction
  must match installed server bytecode/constants before adding semantic preflight.

## Verification

- 139 tests and 131 subtests pass across guard, context, orchestrator, billing,
  proposal audit, experiment discipline and transport.
- A deployed-style isolated-globals test proves a too-large request never calls the transport.
- Recorded D203 replay supplies four missing envelopes while preserving raw response.
- Own-server predecessor canaries remain valid: duplicate pending programs rejected,
  provisional champions retained, no scientific source edit and zero paid canary calls.
- Recorded D205 now passes full discipline validation with no warnings. Genuine other-dataset
  controls remain rejected in regression tests.
- Undeclared transforms feed the existing two-call correction loop. The regression test
  requires one candidate correction, counts both charges, and preserves the raw response.
- Deployment gates caught two incompatible introspection attempts before activation;
  failed staging logs are retained and no paid calls occurred in those self-tests.
- Immutable final operations release: `32582e5a7a5eeb6823ce6ed81f4801a8097c8da4`.
- Original scientific release: `d30052d8df1aa8b99f47aefd0c60a5d51458ae91`, unchanged.
- Original operations wrapper SHA256: `c663043d71021870a42e4f077c5ce2a318f4c118ae31fe153462f3afe05e49c9`, unchanged.
- Broad scope checker reports pre-existing dirty files and generated artifacts. Commits
  contain only the named new guard, its tests and additive adapter; unrelated work preserved.

## Live Qualification

The first guard accepted D204 and D206 and completed three new fits (678 to 681).
D205's new comparison defect was preserved as an ERROR, repaired and replay-tested;
it was not relabeled as successful. The active fit finished naturally during the second
graceful drain. Both STOP flags were archived, not deleted; no training fit cancelled.

| Decision | Status | Native USD | Native Total Tokens | MLflow Trace |
| --- | --- | ---: | ---: | --- |
| D204 | Accepted | 0.025021381 | 132934 | tr-56864fd9186ec0177ded00f3654f0a0d |
| D205 | Failed, false dataset comparison; subsequently fixed | 0.022413332 | 131349 | tr-3af6ff62bbca096a8d3c71dca849aeea |
| D206 | Accepted, includes schema repair | 0.035026257 | 260222 | tr-00a0751dd673e552cc65b29063b7e157 |
| D208 | Accepted | 0.025349815 | 137034 | tr-4c7a7d16b19a7f638d2180e8d70f4c4e |
| D209 | Accepted under final guard v3 | 0.017085955 | 125467 | tr-30b8697f18a71e0f9864cb63c459e30a |
| D210 | Accepted under final guard v3 | 0.017108373 | 125107 | tr-5bf3daf251f568d23bd90d63e2271517 |

Those costs are actual native reports, not a guarantee of per-call savings. Input context
is bounded; cost also depends on output/reasoning, caching, provider rates and repair calls.
Final-version live qualification passed on Oct11: consecutive D209/D210 accepted,
both native MLflow traces OK with exact matching USD/token metadata, and a LightGBM
ranker completed after the final rollout. Total completed increased from 678 to 683;
35 failed fits remain honestly recorded, including the one unsupported graph discovered
during qualification. Benter training and further preparation/dispatch continue.

Final own-user service PID was 2094701. The billing reconciliation timer is active and
its last execution succeeded. Startup and live guard file hashes read back correctly.
Operational proof: final release `self-test.log`, `guard-manifest.json`,
`qualified.json`, `unit-before.conf`, archived STOP and `live-readback.jsonl`.
The preserved failed qualification attempts are not active releases. Their validation
failures consumed zero paid calls; the final release uses strict installed-code parity.

Limits: this restores reliable admission and bounded correction, not a guarantee that
every future scientific experiment will succeed or improve scores. Explicit invalid
proposals still fail safely; repeated same-class semantic failures without new results
stop further paid requests. Historical ERROR traces are not relabeled as OK.

## Operating Boundary

Only imaopt's own campaign and user unit changed. Resource caps and chronological/leakage
validation remain active. All historical records and billing-only recovery are retained.
The campaign stays V7; its legacy MLflow experiment name is not a new campaign version.
