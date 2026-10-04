# Research Expansion Validation

Evidence date: 2026-10-04 UTC. Scope: CI/CD/OPS on
`feat/agentic-research-expansion`; no staging, commits or branch changes. Remote
dependency setup, committed source transfer, Linux fixture canary and guarded
official build were authorized and performed. Dedicated fixture/build transient
units were started; no protected V5/service mutation. Other agents own
implementation/config and coordinate cutover. This report
does not certify the whole plan or V6 launch.

## Evidence Status

### Luna High Session And Legacy Research Handoff

User explicitly requested Luna 6 High as orchestrator for a few real trainings.
Actual subagent `01a106f3-d5e3-71e2-bba4-37773eb3149f` ran on `gpt-6-luna`,
reasoning `high`; accepted evidence-bound decisions were relayed into the V6
external inbox. Exact runtime `376964fab8ba18f4e493824a3fa7702f9e4848e2`.
Local full regression passed 752 tests / 230 subtests; the subsequently added
immutable-config test passed in the focused three-test/four-subtest suite,
also passing on the exact Linux release. Both GitHub research checks passed.

Campaign `agentic_v6_luna_session_20261004`: five completed, zero failures,
zero pending tells/models/traces; mode complete, own unit inactive/exit 0,
no OOM and terminal cgroup peak 7.63 decimal GB under its 10 GB cap.
Independent production MLflow experiment 7 readback confirms five FINISHED
runs, five linked READY registered versions 6-10, and 31 OK backend traces:
two external decision traces plus 29 execution snapshots.
Receipt: `v6-cd-owned/376964fab8ba18f4e493824a3fa7702f9e4848e2/luna-session-20261004-terminal-receipt.json`.
SHA256 `dea86d489ad0d4a58e5d177091dac93c9f237ee97b45b9ccfb99d9ad9dbb701d`.

Best Luna loss 2.176621 narrowly misses matched C1 best 2.176278. Its boosted
pilot 2.181632 improves matched C1 boosted 2.188142 but not the overall Benter
reference or calibrated market 2.000519. No independent confirmation or
market-beating result. No OpenRouter request was sent for session decisions;
session billing is unavailable, not a fabricated API USD zero.

User authorized V5 stop: graceful STOP followed by own-unit cancellation after
snapshotting eight long-running attempts. Service inactive/PID 0; source/data/
config/unit-definition hashes unchanged. Preserve 469 completed, two failed and
eight interrupted ledger records separately. Snapshot `ops-parent/v5-stop-20261004`,
ledger backup SHA256 `e9ea47f520bc4db03a03bfce675a0fa465e652a2f59ab00015f491218b5ec93d`.

Detailed audited handoff `docs/V4_V5_RESEARCH_HANDOFF_TO_V6.md`, SHA256
`bcdd074861678857760fac3056bef6c65f1f888d1600c6cc6dfc72bdb1fd8b5e`,
was copied to the server with matching readback and explicitly read by Luna.
Its receipt and next hypotheses appear in `.tmp/luna-v6-session/feedback.md`
and the campaign's `research-handoffs/LUNA_FEEDBACK.md`. Five trials were the
entire allocation; no continuing autonomous session-agent service is claimed.
The original OpenRouter qualification incidents stay frozen, not retried.

### C1 Integration And Official Controls

Exact runtime: `c1eb224fffd302e47f450b90c233b44a97c9193f`. Full suite:
750 passed, 226 subtests, 185.62 seconds. CI runs 37199169308 and 37199172431
both passed. Immutable release package archive SHA256:
`27c3df14053da07568e70f1000ed439a00b71d894f231d3804f9a2ad09d47fc7`.

Exact Linux private fixture receipt at
`v6-cd-owned/c1eb224fffd302e47f450b90c233b44a97c9193f/fixture-terminal-receipt.json`
shows ten completed physical FINISHED runs, ten linked READY versions, twenty
OK traces, zero pending tells/models and successful paper-study feedback.
All protected hashes stayed unchanged. Paid calls zero; production backend not
used. This fixture does not certify official or paid planning.

Parent-owned official controls at
`campaigns/agentic_v6_official_controls_c1_parent` finished 11:59:17 UTC:
status complete, five completed fits, zero pending tells/models/traces and no
tracking errors, own unit exit 0. Accepted dataset V4 and production MLflow
experiment 7 were used. Nonpaid fixture planning is not autonomous OpenRouter
planning. Independent terminal readback confirmed all five FINISHED runs and
matching READY versions 1-5, 65 backend traces OK and zero pending deliveries.
Receipt `.tmp/v6-data-owned/c1-official-controls-terminal-receipt.json` SHA256:
`6cad316df4fbf6e53513728adad2844cab1c3c4329806091b5ef7d425440aea7`.
Actual scored population checked for all five fits: 18,075 runners and
1,502 races, identical protocol/population hashes. No OOM events.

Logical fit budget remained 8 decimal GB. Physical cache allowance reached
10 decimal GB after fresh reservation-aware admission; available 28.85 GB was
above required 22.53 GB including possible V5 growth and 2 GiB reserve. Same
invocation, CPU allowance and estimates; no protected service mutation. First
fit private peak was 4.14 GB versus estimate 5.65 GB. Continuous V6 and new paid
planner qualification are not accepted; no whole-plan acceptance is claimed.

The newly authorized C1 private planner qualification failed at 12:08:37 UTC.
Its exact runner passed seven offline tests (SHA256
`d3ca6effc557ed66eb7369f13c5ad0f1561aa56bced7be195c6bc93b7d4afab0`).
TCP/TLS succeeded; body transmission lasted approximately 87.32 seconds,
followed by ReadError before headers at 91.300 seconds. No generation ID,
response cost or physical completion receipt; native USD remains null/unknown.
Unit exited 1, no retry, no fits, zero pending private traces and all ten
protection hashes unchanged. V5 remains active. A private trace status OK
only acknowledges snapshot delivery; planner_status is failed and acceptance
is false. Paid admission is frozen; continuous V6 remains blocked.
Receipt: `canaries/acceptance-runner-owned-c1eb224f/operator-evidence/paid-execution/terminal-receipt.json`.

### Current Accepted Data, Frozen Paid Work

Parent/Sagan independently PASSED fresh V4
`dataset-19aaa959e948d6e145ce62c8160413087b46ba3c2578e0de15c452054d9a83db`.
Builder source remains `6245f3ca5a3e37116a13cc7420839bb6dc05f55b`; the persisted
row-hash metadata is corrected. Parent reports feature Parquet bytes identical
to V3 (recorded feature SHA256
`2321218048ffb1f2b9d9ea9a357ec8a753775420f9808385dc323741ad1795a1`).
Accepted V4 manifest SHA256:
`bbe54689bf9e13d1b1afd06122baf1caa9c67e61ddb007be106ea33eb771ce65`.
Observer 52555 reported registry verified at 10:06:44.882104 UTC and then
completed normally. Parent/Sagan acceptance is independent evidence supplied by
parent, not a second exhaustive audit by this OPS role. V3 is preserved unchanged
and remains unaccepted; the corrected V4 acceptance does not rewrite V3 history.

The explicitly authorized isolated 52a paid transport/proposal invocation FAILED.
It is not full-release acceptance and did not train or launch production. Tested
runner SHA256 `25df6c8f2dcb9d8774b480eac8405eb54e0ff554f2c536d50e8317d96eb53fed`
matched; full production private SDK PYTHONPATH was used, not partial overlays.
Prepared/approved bundle SHA256:
`def76f1ac902f3e56afcce68f724af13ed3cc192f17a4fa6837f560e1d5a8acd`.
Paid unit `ima-v6-paid-52a-v4-paid.service`, invocation
`02cef1414d7e4fc192ed0dab0519a655`, ran 10:14:43-10:16:57 UTC; exit 1,
CPU 9.786325s, MemoryPeak 1,572,839,424 bytes. Fresh two-sample headroom and
pre-exec gates proved actual two CPUs/8GB decimal/270s/Restart=no. Error was
`OpenRouter bounded request failed: TimeoutError:` at planner deadline 120s,
not a unit hard-timeout/OOM. One invocation, no persisted HTTP completion
receipts or generation ID; cost is UNKNOWN, not zero. No retry performed.

Private SQLite trace `tr-be5d73c9b6a5caf6ba39cbcdd8883efb` is OK, zero outbox
pending/errors, but no native USD/cost attributes. No runs/model versions,
fits/dataset builds or production MLflow writes. All ten protection hashes,
original V5 PID/invocation and STOP absence matched. Independent terminal receipt
at `canaries/acceptance-runner-owned-52a3c3e/official-v4-once-ops/independent-terminal-receipt.json`,
SHA256 `981c26818a3cf7bfa761d104d72bf20d6e918a1c25591223fe6598831519c08a`.

Non-charged GET diagnostics 10:22:02-15 UTC: same HTTPX 0.28.1/private origin,
all eight upper/lowercase proxy-presence flags false. Models/default IPv6
200/6.0652s, forced IPv4 200/2.6455s, forced IPv6 200/2.0070s; TCP/TLS
successful for each. Approved model exists. Both DNS families use tailscale0,
table 52, online exit 100.88.109.45. Own MLflow health GET 200. Key/credits GET
200, minimal counters only, no secrets. No pre-call baseline or saved generation
ID allows precise billing attribution with concurrent V5 usage. Failed POST
phase was not instrumented; generation delay is a hypothesis, not proven cause.
No IPv4-forcing/host-route change justified or made. Diagnostic receipt SHA256:
`5d452f7976344a4dc06786d0e3e53f778370244543a7a76bb3eb534cb4e2d54a`.

Further paid calls, training and cutover remain FROZEN. Parent reports helper
`7876d830bd762a8e5beba3b7b20e0848f7889567` committed/pushed/tested/CI green;
Parent subsequently reports local canary v2 PASS: ten trials, actual paper-study
WIN/PLACE/QIN/TRI outputs, later planner feedback received, no outboxes/paper
pending. Paper engine committed `1fabe3f9bfdd6d69aab860f1fa9b33198968329f`.
Controller agent released its source; parent tests and final commit remain
pending. This is parent-supplied local fixture evidence, not a Linux rerun,
paid acceptance or official-data training proof by this role.
Nietzsche owns the controls unit; no unit mutation is assigned to this OPS role.
Next final SHA needs exact immutable transfer, refreshed builder hash binding and
fresh integrated Linux fixture `--paper-study --max-trials 10` on private MLflow:
ten FINISHED trial runs/linked READY versions, all traces OK, zero pending/errors/
tells/paper studies, WIN/PLACE/QIN/TRI study outputs and later planner feedback.
Future paid actual-data/new-graph and
actual training require new explicit scope; none is proven by the old 52a attempt.
All this role's required observer/exec sessions completed. No remote action or
paid request occurred during this documentation reconciliation.

### Historical V4 Launch And 52a Source

V4 request was transferred and started from exact committed builder
`6245f3ca5a3e37116a13cc7420839bb6dc05f55b`, 09:38:58 UTC. Own unit
`ima-v6-dataset-build-6245f3ca-v4.service`, invocation
`dafcc8e3278f4e2eb21766146765528e`, PID 1030588. Pre-execution readback proved
actual 7200s hard timeout, two-CPU quota, MemoryMax 8,000,000,000 and zero swap;
two conservative fresh headroom samples passed. Request wall budget 5400s.
At historical observer readback 10:03:12 UTC it was building, no request error,
CPU 1453.074507s, peak 1,981,923,328 bytes. That sample was not final peak or
acceptance; completion and independent acceptance are now recorded above.
Observer session 52555 was successfully polled in this role's namespace; parent
cannot access that local session and independently monitors the durable unit.

Then-designated final 52a transfer also passed independent PAX/tree/blob/mode and
extracted readback; exact SHA `52a3c3eecb7cf3342d0b690b059e55a8f67e1845`.
231 committed files, 2,467,840 archive bytes; archive SHA256
`f1aa0c7985774cc781c3d90473c226a1f677654d1b013cce308edd87d0adbdba`;
package receipt SHA256
`d8e46711f0e06cb5814687c7a12e979e97e77f1be10c678991bf856102b12a7e`.
Receipt at `v6-staging/52a3c3eecb7cf3342d0b690b059e55a8f67e1845/package-receipt.json`.
All 11 builder module hashes match 624/52a; 130/133 runtime files match old e23
fixture, with registry/orchestrator/expansion differences retained explicitly.
Full fixture equivalence is NOT claimed. Exact 52a CI runs 37192679625 and
37192676301 were read back successful. V3 remains unchanged and unaccepted;
parent/Sagan proved eight signed NaNs/four columns explain its hash discrepancy.

### Future Bounded Actual Training Preparation Only

Runbook now specifies separate immutable bounded/continuous own-ops configs,
production MLflow port 5000, audited V4 manifest paths, future assigned final SHA and real
campaign. No training, paid call, remote config installation or V5 STOP executed.
V4 identity is now accepted; config bytes/hashes await the final integration SHA.
The attempted 52a private paid gate failed. Parent must assign a fresh scoped
acceptance after final fixture and separately authorize actual training.

Actual local isolated contract verification used AST-extracted classes/loader/
identity function from exact committed 52a, not dirty working files: nine checks
passed (both configs valid, invalid 260/5 budget and two-program queue rejected,
same identity accepted, data/protocol/revision/environment drift rejected).
Prepared Python recipe compiled. Identity test isolated the portfolio-version
lookup; this is not a full runtime/MLflow/full-data canary. Source SHA256:
optimizer `c9ec8d711f519066511f2df6feb879135bf41a17ececc66467bf1c01e9e8fcd0`;
identity owner `c2a199650fafa6773ee65ec67ffaecd9bb10a0bdaff4bcacd00c1fe1778e511d`.
Operational settings do not enter campaign identity. Package dependencies and
production backend/model require separate equality receipts because environment
identity is Python/platform-only and backend/model are not identity fields.

### Historical Prepared V4 Request

Explicit preparation assignment created `.tmp/v6-dataset-request-v4.json`, new ID
`v6-initial-strict-fullhistory-latest-v4`, with rationale for the parent-committed
ordered-row hash representation repair. Structured JSON comparison proves ONLY
`request_id` and `rationale` differ from v3. Population/full history/strict policy,
protocol, five targets, raw manifest identity/watermark and 5400s budget are
identical. Actual local DatasetRequest schema validation passed; normalized
fingerprint `a58032954cb8153444959fcf985af15ff773bb5cd1c71c2b0f3149fc190b0f7a`;
artifact SHA256 `28d4d52e16fa97e287609702db4bf939490c6b282f513e76995743e2da051d30`.
Prepared runbook recipe retains verified-before-exec 7200s timeout, 8GB decimal
and two CPU quota/native threads 1, exact committed packaging and independent
persisted-Parquet row-hash acceptance. Kierkegaard owns the repair; no new builder
SHA/execution assignment had been received at that preparation checkpoint;
subsequent 624 build/transfer and limited paid attempt are recorded above.
The pending immutable V3 candidate, v2/v3 state and protected units were not
changed. The later signed-NaN cause proof and fresh V4 repair supersede the
then-unresolved disposition without modifying those historical artifacts.

### Historical Terminal Receipt: Fixture Passed, V3 Held

Historical accepted fixture source is exact committed SHA
`2a87784a52be31c2e16272fe2791b5e7b58ed9b7`. Immutable release:
`/home/imaopt/research-v2/live-releases/ima-v6-2a87784a52be31c2e16272fe2791b5e7b58ed9b7`.
Actual archive: 231 committed files, 2,457,600 bytes; committed verifier and
target PAX/tree/blob/mode/post-extraction readback passed. Archive SHA256
`c65d27e570555c603a16f0174354726222af7b0ffd34b113df534b8604bf92b9`;
package receipt SHA256
`000138dba154b25b4a5ef043390b5675e41d56ac2f1165aeebc76fd9c691dcd7`.
Parent reported corrected CI run `37191680881` GREEN. No staging/commit by
this role; dirty protected workspace files were never archive inputs.

The executed fixture source remains
`e23c2e629ce131c9b1d9e7affe7a3f9df65ab257`, not relabeled as 2a. Actual
extracted SHA256 and committed mode/blob equality passed for all 133 runtime
files under ima/scripts/deploy/config in the two immutable releases. Runtime
tree SHA256 `7d266bef1c429468ea817732fd5ce41b5a68f1b584e37deca773294fc8498f15`.
Parent explicitly accepted this binding in place of a redundant fixture because
the successor changed only the preparation concurrency test.

Fixture root `/home/imaopt/research-v2/canaries/agentic_v6_e23c2e62_fixture5`;
unit `ima-v6-fixture-e23c2e62.service`, invocation
`a76cfb7d4d954ac89fb724ada5fd758e`, started 09:24:07 UTC after fresh two-sample
headroom passed (28,673,970,176 / 28,656,803,840 available vs
24,536,166,400 / 24,514,252,800 required). Before execution, verified
MemoryMax 8,000,000,000, CPU quota two, swap 0, timeout 900s and expected
MemoryHigh page rounding. Unit/resource proof is `evidence/unit-before-exec.json`.
Status complete at 09:25:04.508802 UTC; five completed attempts, all uploaded/told,
zero failed/errors/pending tells/tracking/trace delivery. Backend read-only SQLite
acceptance: FIVE FINISHED physical runs, FIVE READY versions linked to exactly
those runs, SIX traces all OK, and zero undelivered trace outbox entries. All
runs record e23 and unchanged environment hash
`2827ddd042f17d0f82dfb5473fddd8fff1ebd512694e24110b2b1290d6f539c4`.
Journal CPU 100.858s / wall 60.405s / reported peak 935.1M; status observed exact
memory peak 980,627,456 bytes. Backend/status, not CLI exit alone, establishes
fixture acceptance. Synthetic two-family work is not official-data training,
paid planning, production MLflow authorization or advanced graph acceptance.

Durable terminal receipt:
`/home/imaopt/research-v2/v6-staging/2a87784a52be31c2e16272fe2791b5e7b58ed9b7/fixture-runtime-binding-receipt.json`,
SHA256 `73cb7d3d5d1985d0754b086c1fc12d83576ea51466357a8dd73b81977ed21396`.
It retains run/version/trace IDs, runtime file hashes, original source revisions,
environment identity and all ten matching protected SHA256 values. V5 STOP absent;
no protected V5 unit/configuration change or paid/live launch performed.

V3 registry published `dataset-62744fab4a46cbeada2488be5d15a2112d3d364d8148331ccd48f6dda736a4e8`
at 09:18:54.853585 UTC, with 171,782 development rows / 13,924 races from
178,718 source rows / 14,488 races. Registry report: 564 excluded races / 6,936
rows, 6,258 confirmation rows quarantined, three folds, zero unaccounted source
keys and no confirmation labels published. Recorded build time 1611.606s;
unit journal CPU 1614.815s / wall 1648.860s (includes resource wait gate), reported
peak 3.9G. Unit is inactive; no operator cancellation occurred. Original build
SHA remains d45; its 11 builder-identity file hashes match e23, which matches 2a
runtime. Published directory is under
`campaigns/agentic_v6_research/datasets/datasets/<dataset-id>`.

**Historical V3 disposition:** publication and registry `verified`
state are NOT final independent data acceptance. Parent/Sagan audit reports
declared ordered-row hash `f19ba...` differs from persisted-row hash `f6ddb...`
(abbreviated values supplied by parent, not full hash receipts). File checksum,
dtypes and provenance checks reportedly passed. Parent/Sagan later proved eight
signed NaNs/four columns explain the discrepancy; corrected fresh V4 passed.
Do not silently replace hashes or promote the retained V3. Another build/transfer
still requires an explicit parent SHA/request. Unsupported data requirements
remain as disclosed by the builder;
no optional preparation benchmark was run. The older checkpoints below preserve
their timing and must not override the current accepted-V4/frozen-paid disposition.

### Historical e23 Transfer And Queue

Parent authorized exact final code SHA
`e23c2e629ce131c9b1d9e7affe7a3f9df65ab257`, reporting full 668 tests passing
in 158.656s, exit 0; those are parent-supplied suite results, not this role's
rerun. The authorized next execution is a fresh five-trial/two-worker Linux
fixture with private SQLite and 8GB/two-CPU resource-before-exec gates. No paid
planner, production tracking authorization or V5 cutover has been assigned.

Actual committed archive: 231 files, 2,457,600 bytes, archive SHA256
`0642cae49685b77200b5e32ccf56e3f0d32b2a5cbf1d7070172cf39f8a0d5eea`;
receipt SHA256 `fdf7f6736b66820138cc66a6cade93729cf64cc6716b4399f7e2b4cd8127f21e`.
Committed verifier/source readiness and target PAX/tree/blob/mode/post-extraction
readback passed. Read-only release:
`/home/imaopt/research-v2/live-releases/ima-v6-e23c2e629ce131c9b1d9e7affe7a3f9df65ab257`;
receipts under `v6-staging/e23c2e629ce131c9b1d9e7affe7a3f9df65ab257`.
Local artifacts:
`/var/folders/wq/tn0d0r417cb3kghxltyqhjt80000gn/T/ima-v6-final-release-92ghxp27`.
All 11 dataset-identity builder file SHA256 values match the interim d45 release;
exact values are retained in `builder-hash-comparison.json`. This permits explicit
builder-identity reuse after valid dataset readback, not provenance relabeling.

Parallel fixture NOT started: headroom samples at 09:08:45/50 UTC failed the
conservative simultaneous-unit guard. Available 30,998,769,664 / 30,979,428,352
bytes vs required 33,991,254,016 / 33,990,488,064. Requirement includes new canary
8GB + host reserve 8GB + possible V5 growth to its unchanged 107,374,182,400 cap
+ remaining active build allowance to 8GB (current 1,153,503,232). Host pressure
avg10 0 does not override failed capacity. `parallel-canary-headroom.json` records
both samples. At this checkpoint fixture queued until build completion; its later
fresh guard/execution/backend pass is recorded above. Do not lower caps/reserve,
stop V5 or claim fixture tracking acceptance from this transfer receipt.

Mandatory backend acceptance: exactly five completed/zero failed attempts,
five FINISHED runs, five READY model versions linked to those physical runs,
nonempty traces all OK, no result/tracking errors, no pending outbox/tells.
Exit zero, READY versions or an empty outbox alone are insufficient. At this
historical checkpoint fixture was QUEUED; terminal acceptance is recorded above.
Paid/live gates remain separately held.
Parent subsequently reported e23 hosted CI run `37190999593` failed ONLY
`test_concurrent_miss_only_builds_once`: sleep-based `any(wait)` assertion was
flaky, while actual build count 1/cache checks and model telemetry passed. Parent
is replacing the test timing assumption with an explicit spawn barrier, not
changing production runtime. Two actual-archive checks were skipped in CI.
The final runtime e23 remains the assigned fixture target; corrected test-only
release/green CI are later gates. Earlier local 668/focused Linux passes do not
override the failed clean-checkout suite. No production cutover; queued fixture
is evidence gathering only.

Targeted dependency reuse receipt `dependency-reuse-receipt.json` verifies four
accepted frozen receipt/version-manifest hashes unchanged, without rescanning
46,947 files. Actual imports: NumPy 2.5.3, pandas 2.3.3, PyArrow 25.0.1,
Pydantic 2.13.5, MLflow 3.16.1, scikit-learn 1.9.1, LightGBM 4.7.0,
JupyterLab 4.6.2, Graphviz 0.21, Plotly 7.1.0, all from private dependency paths;
ima/scripts from the exact final release. Ten protected hashes still match.
Optional XGBoost is absent and is not a declared requirement here; it was not
installed or falsely included in the successful import receipt.

### Historical Interim v3 Build Start

DATA-BUILD ONLY assignment: exact committed SHA
`d45dc21649d3931d615282565b5517f388160acc`, incorporating parent builder repair
`df07382` and future-event-empty fast path `d45dc21`. Parent supplied 65 passing
event/speed/data tests; this role did not rerun that suite. This interim SHA is
NOT the final production controller/tracking release and authorizes no MLflow
fixture, paid planner or live V5/V6 change.

Actual archive from Git commit only: 230 files, 2,375,680 bytes. Committed verifier
source integrity, required paths, PAX/tree/blob/mode and target extracted readback
passed. Archive SHA256
`816366ccf7f4956b7f626203cc9852369fa77696bc9251c60f2cdf6174506ede`;
package receipt SHA256
`3f5556dab034c51aa68323b9773f056d6d8472fe231e6f62816837f0e1f4d88f`.
Read-only release:
`/home/imaopt/research-v2/live-releases/ima-v6-d45dc21649d3931d615282565b5517f388160acc`.
Staging/evidence:
`/home/imaopt/research-v2/v6-staging/d45dc21649d3931d615282565b5517f388160acc`.
Local generated artifacts:
`/var/folders/wq/tn0d0r417cb3kghxltyqhjt80000gn/T/ima-v6-build-v3-_op0je9u`.
No dirty/uncommitted source included; no mutable release alias changed.

Later parent-reported gates belong to OTHER revisions: hosted Linux suite/package
verification green on `d1925c9`; immutable MLflow code-copy repair committed as
`024b0bd`, with 18 tests passing on Linux. Global trace SDK/provider race repair
was then parent-owned/in progress and later included in e23. These reports alone
do not certify this interim
data-build SHA for production. Final release must pass its own full suite and a
fresh five-trial private SQLite Linux fixture with five FINISHED runs, zero
tracking errors and no pending tells/tracking. That canary has NOT been run on
the interim assignment. Production tracking authorization and actual paid
decision/graph acceptance are later separate gates, after valid data.

Separately assigned request `inputs/dataset-request-v3.json`, SHA256
`4e7bfe3ce72ddc489185a896a60d32faa57fdb4dcacfc87452196f87b7cabdd3`;
normalized fingerprint
`83bf59f02f89fa142f2288c5fb5a18877c026ae3f0f7c53258e34869b68ddff9`.
Same full-history strict population/protocol/500-race quarantine; request budget
5400s. Source is frozen `inputs/official-snapshot`, all 17 copy hashes reverified,
manifest SHA256 unchanged. V2 request bytes and staging are preserved; all ten
protection hashes match, V5 active original PID/invocation and no STOP.

Two headroom samples passed: MemAvailable 32,431,144,960 / 32,519,340,032 bytes
against required 27,918,581,760 / 27,941,584,896; full pressure avg10 0.
Dedicated unit `ima-v6-dataset-build-d45dc216-v3.service`, invocation
`d8bbfe34b0e146f9a5815ba9711d9ae8`, PID 1023674, started behind an explicit
pre-execution gate. Before builder execution, systemd readback verified
RuntimeMaxUSec `2h` (7200s), MemoryMax 8,000,000,000, MemoryHigh 6,000,000,000,
swap 0 and CPU quota two. Kernel `memory.high` 5,999,996,928 is the verified
4096-byte page rounding; memory.max 8,000,000,000, swap.max 0, cpu.max
`200000 100000`. Initial exact high-limit equality assertion paused the gate;
page-rounding verification then opened it without a unit restart.
Proof `build-unit-before-exec.json` and `resource-gate-approved.json` records
builder-not-executed at readback. `build-command.json` records exact wrapper and
builder argv; durable output is in the own unit journal (no SSH-bound pipe).

Gate opened 08:52:00.657842 UTC; registry requested 08:52:01.586336 and building
08:52:01.591512 UTC. First builder process readback was CPU-active, peak
398,610,432 bytes / CPU 11.203s. Final dataset verification, raw manifest object
and byte provenance, units, source accounting, target eligibility and immutable
payload readback were PENDING at this start checkpoint. Publication later completed,
but independent row-hash audit now holds promotion as recorded above. No optional
benchmark was run.

### Authorized Committed Linux Checkpoint

Parent authorized source/build/fixture execution for exact commit
`5b63862360f5380b6f247bf1c4da0a0854528aa6`, reporting 629 full tests in
121.491s and latest 39 executor/data tests passing. Those parent counts are
supplied release-gate evidence; earlier in-progress test failures below remain
historical, not a claim they still fail in this commit.

Actual committed Git archive: 230 files, 2,365,440 bytes; PAX/Git tree/blob/mode
and mandatory paths passed using the verifier FROM that commit. Archive SHA256
`24148e22009c1ac43017aac0afed9be357647655c7c31d8f4b33367d8f24c5d9`;
receipt SHA256 `da00069b96e6746e162415c20462481337ae9f7fbaaee9d5a2fc19239516f990`.
Transferred over Tailscale and independently verified/extracted/read back to
`/home/imaopt/research-v2/live-releases/ima-v6-5b63862360f5380b6f247bf1c4da0a0854528aa6`.
Committed files/directories are read-only; exact `REVISION` and copied receipt
are generated metadata. No dirty protected user files were archive inputs.
Staging/receipts: `/home/imaopt/research-v2/v6-staging/5b63862360f5380b6f247bf1c4da0a0854528aa6`.
Local command/log/artifact root:
`/private/var/folders/wq/tn0d0r417cb3kghxltyqhjt80000gn/T/ima-v6-release-zba7k2qy`.

Linux fixture unit `ima-v6-fixture-5b638623.service`, invocation
`2a0e5c563e7a4dbbb4bd54b2ed652d6e`, exited 0 in 58.246s, CPU 91.589s.
Actual limits read back: MemoryMax 8,000,000,000, MemoryHigh 6,000,000,000,
swap 0, cpu.max `200000 100000` (two CPU quota). Observed peak 1,059,840,000
bytes; memory high/max/OOM events 0. Root:
`/home/imaopt/research-v2/canaries/agentic_v6_5b638623_fixture5`.
`evidence/unit-readback.json` and `evidence/canary-validation.json` exist.
Private SQLite only; production MLflow/paid planner not used. Ledger/status:
complete, exactly five completed, zero failed, null result errors, zero pending
tells/tracking; all five attempts have uploaded/told timestamps. Five READY
model versions link to five physical runs; five traces are OK. Revision lineage
matches the exact commit, environment hash
`2827ddd042f17d0f82dfb5473fddd8fff1ebd512694e24110b2b1290d6f539c4`.

**Tracking gate is NOT clean:** all five private MLflow runs are FAILED.
CLI output records `NonRecordingSpan` missing `context` and five permission
errors cleaning temporary `model/code/ima` directories. Read-only source directory
modes copied into MLflow temporary code trees are the likely permission cause;
registration succeeded before the exception, then idempotent retries reused
registrations and drained the outbox without repairing failed run statuses.
Do not conflate READY models/empty outbox/exit 0 with clean tracking. Exact
errors remain in `linux-canary.log` locally; frozen release/dependencies and
failed statuses were not changed to conceal this. Parent owns tracking/runtime
fix and a new committed canary if needed. No production campaign launch.

Authorized dataset request copied separately (not code archive) to
`campaigns/agentic_v6_research/inputs/dataset-request-v2.json`; SHA256
`1fc4d6e6090320927d94cd7366deb835d9e8d2c2a3b0cd8778a08953e6ab8f91`.
Fixed snapshot manifest SHA256
`fcbb6a3f9b83ae06add78dcb674649c8ebbb202e86c32157d4efed1616efaee7`
matches the request; raw inventory 178,718 rows/14,488 races, parser v14.
Separately authorized frozen snapshot relocation completed at 08:39:53 UTC:
`campaigns/agentic_v6_research/inputs/official-snapshot`, 17 regular files,
776,217,339 bytes, every source/copy SHA256 equal and directories/files read-only.
Manifest bytes/hash remain identical. No symlinks, mutable alias, hardlinks,
dataset rebuild or recorded original-source-path/identity rewrite. Disk available
before copy was 749,632,700,416 bytes. Receipt beside the copy:
`inputs/official-snapshot-copy-receipt.json`, SHA256
`4dc59678a6375da3feda29d9bf44fbbb9504cc3d64f8d66d05a93ca7e90010cd`.
The parent launcher must validate relocated manifest/file hashes against recorded
original provenance; path relocation alone is not identity equivalence.
Request is strict/full-history, latest three whole-meeting folds with 9000
minimum train/500 calibration/500 score and 500 final-confirmation races.
This deliberately differs from pinned V5; no original-contract replay claim.

Dataset build `ima-v6-dataset-build-5b638623.service`, invocation
`20a35342a6bb443bbc41dd3adef6ce9a`, began 08:23:18 UTC after a two-sample
headroom pass. MemoryMax/CPU quota match the fixture; RuntimeMax readback 30min.
Registry: `/home/imaopt/research-v2/campaigns/agentic_v6_research/datasets`.
At approximately 08:36:39 UTC it was CPU-active at 13m20s elapsed, CPU 800.645s,
current memory 1,295,482,880 and peak 1,660,284,928 bytes. Live RuntimeMax remains
30min at that readback. Parent subsequently specified a 3600s hard limit;
an attempted dedicated-unit `set-property --runtime RuntimeMaxSec=3600` was
rejected. A runtime `systemctl edit` accepted the drop-in but live readback still
reports 30min with no loaded DropInPaths (transient unit). Therefore 3600s is NOT
proven effective. No stop/restart was used to change the running build. The 1800s
acceptance budget is distinct from the desired hard timeout. Request history
contains requested/building; source/confirmation setup and dataset staging exist.
The process is CPU-active on one Python thread. No intermediate substage timing
is emitted, so a precise active Python function is not proven by this readback.
Final build was CANCELLED by explicit assignment at 08:41:16 UTC after parent
confirmed P1 raw-manifest variable shadowing (`raw_manifest.json` would become
the string `trainer_id`) and P2 event distance unit `1` instead of `m`. These are
invalid-builder findings, not timeout/OOM findings. No dataset was published or
accepted and target/source-accounting acceptance was not reached. No benchmark
was run because no valid candidate exists. Unit checkpoint `build-unit-readback.json`;
exact command/log files are `build-command.json`/`official-build.log` locally.
Durable logs/commands were copied to staging through Tailscale. Cancellation
receipt `build-cancellation-receipt.json`, SHA256
`49f14b88b6224ec7c6663d0b537b5db45b37f2ff928b55ec9af86d4d795b54f3`;
journal `build-cancelled-journal.log`, SHA256
`dd1d712647912b7401b868f1b9351a6f92d93ffb198de6b9de41a317e1e0a183`.
Journal final wall 1078.622s / CPU 1078.315s / peak 1,660,284,928 bytes. Stop and
inactive/PID0 readbacks occurred within the same UTC second. Systemd's post-GC
`Result=success` is not build acceptance. The SSH wait had disconnected with 255
while the same process continued; final cancellation was a separate explicit stop.
Original request remains `building`, byte-preserved SHA256
`566f6f6b0bd04c0ccab42d2f6cf76f8cf988a3a71df4f0f59253208ddbd149d1`;
its separate cancellation receipt is authoritative for operator disposition.
Staging directory and request were neither deleted nor silently retried.
All ten protection hashes match after cancellation; V5 active PID 792999 and
invocation `0e74db4db0fa4f05bf8a308e200b022c` unchanged, STOP remains absent.
Parent subsequently reported GitHub CI failure on this SHA: exactly two
`tests/test_data.py` tests assumed gitignored
`data/processed/historical/runners.csv.gz` exists in a clean checkout. Parent
reported the portable-fixture fix committed as `9a15365`; this role has not
modified those tests or certified that successor CI. New committed release
and green CI are required before cutover. This cancelled attempt retains its
original code/dependency/request identity. Builder defects now require a repaired
committed release and fresh request v3; earlier unchanged-builder reuse reasoning
is superseded. Never accept or reuse the invalid old staging candidate.
Parent also confirmed initial paid-planner capabilities omit the raw-manifest
identity needed for a DatasetRequest. Controller capability/regression fix is
parent-owned and requires a new committed release; builder files are reportedly
unchanged. Do not run paid/live planning on `5b638623...`. This assigned build
uses the explicit hash-backed request and does not exercise that planner gap.
Optional official preparation benchmark is authorized only after main-build
readback, under a fresh 8 GB/two-CPU unit and fresh headroom. Parquet/CSV identities
must be recorded separately and all candidate rows preserved.

| Check | Actual result | Interpretation |
|---|---|---|
| Canonical megaskill plan check | Passed | Plan structure only |
| Parent-provided pre-change suite | 334 passed in 67.307s | Supplied baseline; not rerun at that revision by this role |
| Existing CLI `python -m scripts.optimize run --help` | Passed locally | `run --campaign PATH --config PATH` exists |
| Local `pip check` | Passed | Existing local environment only |
| Existing locks/extras pip resolution | Local `pip install --dry-run` passed, exit 0 | No installation; not a fresh Linux environment proof |
| Corrected targeted regression command | 29 passed in 0.160s | `tests.test_optimizer tests.test_research_resources tests.test_research_discovery_state` |
| Initial targeted command | Failed to import `tests.test_research_scheduler` | That planned test module did not exist; corrected command above |
| New launcher `bash -n` | Passed | Syntax, not deployed execution |
| Full working-tree suite | 334 passed in 67.147s, exit 0 | Collected while concurrent implementation proceeded; not all later files were included |
| Earlier successor/telemetry tests | 60 ran in 0.259s; eight failures, five errors; exit 1 | Historical in-progress snapshot, superseded below |
| Latest successor/telemetry tests | 86 ran in 24.235s; three failures, zero errors; exit 1 | `2026-10-04T06:48Z`; real-CLI canary failure resolved, remaining IDs below |
| Workflow YAML and guardrails | Passed | Read-only token, existing test command, no secret/deployment references |
| Launcher failure cases | Passed | Unknown args/extra args rejected with 64; missing environment rejected with 1 before credential access |
| Embedded launcher preflight | One valid fixture passed; five adversarial fixtures rejected | Temporary local root substitution; revision, predecessor policy/campaign, unknown key, missing/relative input; no ledger created |
| Linux unit validation | `systemd-analyze verify --man=no` passed, exit 0 | Systemd 252 package installed in disposable Debian container; actual unit verified; container removed automatically |
| GitHub Actions | Workflow authored; not remotely run | No credentials, paid planning or deployment in workflow |
| Parent final config schema | Passed via `_load_config` and `CampaignConfig.validate()` | Working tree only; no campaign executed |
| Parent canary CLI `--help` | Passed | Local fixture execution/tracking now verified below; official/live runtime gates remain pending |
| Effective remote V5 `pip check` | Exit 1: missing JupyterLab, Graphviz, Plotly | Preserve V5; repair only private V6 dependency closure |
| Remote offline locked pip dry-run | Exit 1: no Pydantic 2.13.4 wheel in existing stores | Private locked environment not ready from existing wheels alone |
| Authorized private copy-mode setup | Offline install and `pip check` exit 0 | 120 original versions preserved; 189 effective distributions, 25 private module origins |
| Parent fixture canary | Parent CLI exit 0; readback mode complete, 10 completed, three decisions, pending tells 0, tracking 10 | Synthetic/offline; not paid or advanced dataset/graph acceptance |
| Parent fixture champion replay | Two families, prediction/objective delta 0 | Local artifact readback; unknown release/environment lineage, not pinned V5 replay |
| Parent local MLflow fixture canary | Read-only SQLite: five FINISHED runs, five READY linked model versions, four OK traces; status complete, five completed, tells/tracking 0 | Local fixture tracking only, not remote MLflow/paid/native USD UI proof |
| V6 live canary, replay, UI and sustained feedback | Not run by this role | Parent coordinates cutover; launch gates remain open |

## CD Preparation Refresh

Historical preparation evidence below predates the authorized committed Linux
checkpoint above. The initial assignment was read-only remotely; later isolated
setup/source upload/fixture/build assignments were executed. Paid launch and V5
mutation remain held. The earlier in-progress test failures below are historical.
Own-user venv/lock/extra inventory, actual OpenRouter Tailscale routes, frozen
snapshot transfer sizes and the prepared V5 STOP procedure are in the runbook.
No protected STOP/config/route/service action was performed. Dedicated successor
fixture/build transient units were subsequently started under explicit assignment.

Local package verification exercised a scoped `git archive --format=tar` from
committed SHA `6ef7481dc93f406e77174d510b36d99155ef38b4`, 1,515,520 bytes.
Its committed source integrity passed; deployment readiness correctly failed for
14 missing required paths, including the then-not-yet-created successor config and
uncommitted controller/service/docs/tests. Temporary test archives were removed;
this is a verification receipt, not a prepared V6 release artifact. No source was
copied from dirty/untracked workspace files into the archive.

Archive adversarial checks passed: unknown commit, non-full SHA, modified content,
duplicate file, symlink, traversal path, omitted file and wrong PAX commit metadata
were rejected. Full required-path success subsequently passed for the 230-file
committed package in the checkpoint above.
The verifier outputs package SHA256/size only after complete source/readiness
checks; it does not certify model/data/recovery gates or extract/deploy anything.
CI runs that committed-SHA package check after the regression suite, without
uploading/deploying a package. Hosted CI subsequently ran and failed as recorded
in the committed checkpoint; this is not a current unrun claim.

Runbook schema example was accepted through the real `scripts.optimize.main`
config loader/constructor with `run_campaign` mocked to validate only. No campaign
was created, no planner called and no `config/agentic_research_expansion.json`
workspace file written by this role. Parent has since supplied the final config
and canary. Their schema/help checks passed; approved model is
`deepseek/deepseek-v4.1-flash`, cap $5, campaign
`/home/imaopt/research-v2/campaigns/agentic_v6_research`, ceiling 26 workers,
24 CPU threads, 100,000,000,000 bytes / 93.13225746154785 GiB. A ceiling is not
an actual admission count. Loader normalizes the 260-trial decision alias.
Parquet readers exist in profile and executor; actual preparation/training on
the frozen Parquet input remains a separate gate. The package verifier now
also requires the committed canary script. Committed package readiness later
passed; repaired production runtime/tracking acceptance remains pending.
Launcher was corrected for the observed own-user Python executable symlink;
existing V5/acquisition runtimes were not changed. Bash syntax, CI YAML guardrails
and the canonical plan checker passed locally.

Interpreter path fixtures passed for a regular in-root executable and own-runtime
symlink; unapproved executable target, out-of-root entry point and escaping dataset
symlink were rejected. No fixture campaign ledger was created. Final own-user
readback at `2026-10-04T06:22:07.472087Z` matched all ten protected hashes,
V5 `training` / cycle 123 / 434 completed / two failed / eight running / zero
pending tells and tracking; V5 STOP absent, V6 unit absent, MLflow health 200.

CI/CD/OPS owned artifacts are ready for parent integration. Whole-plan verdict is
not done: successor tests failed at the later snapshot, hosted CI has not run and
V6 launch/readback is parent-owned. The megaskill whole-plan final gate with
`--verdict blocked` passed against the shared execution state; it is not an
acceptance pass. No `.mega` records were written by this role because they are
outside its allowed edit surface.

Later command: `.venv/bin/python -m unittest tests.test_research_expansion
tests.test_research_telemetry`. Observed failures include blank review/unallocated
reason validation, Benter-only graph lane classification, legacy recipe-hash
null-field parity, incomplete-identity champion grouping and common-population
comparison of different feature datasets. Errors include missing `search` and
`v6_proposal` fixtures on `PackageIndexAlignmentTests`. This is an in-progress
working-tree observation, not a verdict on the eventual implementation. Those
files are outside this role's ownership; no edits were made to them. Resolve and
rerun before launch; the earlier 334-test pass does not supersede these failures.

## Historical Parent Handoff: Exact Failures

Latest focused rerun at `2026-10-04T06:48Z`, after concurrent implementation
changes: `.venv/bin/python -m unittest tests.test_research_expansion
tests.test_research_telemetry`, 86 tests in 24.235s, three failures / zero errors,
exit 1. Previous 06:35 run collected 82 tests and four failures in 30.589s.
The `_prepared_fold(... pins=...)` real-CLI canary failure no longer fails;
the original three blockers remain:

| Exact current test ID | Observed assertion | Test line |
|---|---|---|
| `tests.test_research_expansion.RuntimeFixtureTests.test_fixture_planner_respects_one_trial_operator_ceiling` | `item.trial_budget` is 2, expected <= 1 | 705 |
| `tests.test_research_expansion.RuntimeFixtureTests.test_missing_preparation_manifest_schedules_preparation_before_fit` | Expected `_prepare_program` once; called zero times | 694 |
| `tests.test_research_expansion.RuntimeFixtureTests.test_restart_resumes_reserved_attempt_before_asking_replacement_work` | Original `attempt-91b8fc59ecb2aa8fe45286bbefb2f1f7d44f887345fc277d215cc76abdbc9aee` absent; replacement `attempt-ce941d8000fe25fcb29ae6e81d46ae283a058eaa0bd8c53a4d1ef4b4ea7c8f1b` completed | 652 |

No claim is made that the whole suite or launch gates pass. This role did not
edit implementation/test files. The latest table is the active parent handoff;
historical details below are preserved to distinguish fixes from missing proof.

### Earlier Snapshot

Read-only test capture started `2026-10-04T06:10:24.344472Z`:
`tests.test_research_expansion` and `tests.test_research_telemetry`, 60 tests in
0.179s, nine failures / one error, exit 1. A unittest runner collected exact
failure IDs and tracebacks; fixtures execute in their local temporary directories.
The earlier eight-failure/five-error snapshot is historical. Missing test-fixture
attributes no longer failed in this refresh. Owners must rerun after their changes.

Every test prefix below is `tests.test_research_expansion.` except the two marked
`tests.test_research_telemetry.`. Lines describe this working-tree snapshot and
may move during simultaneous edits.

| Exact test suffix / module | Result and observed assertion | Test line |
|---|---|---|
| `DecisionApplicationTests.test_invalid_fixed_control_extension_does_not_partially_apply_new_program` | Failure: unexpected program `980ce35dd584cd53` remains after rejected extension | 391 |
| `MeetingBoundaryTests.test_variable_meeting_sizes_do_not_score_the_same_race_twice` | Failure: eight scored entries / six unique race IDs | 513 |
| `PackageIndexAlignmentTests.test_invalid_probabilities_cannot_pass_vacuously_on_nonoverlapping_index` | Failure: expected ValueError not raised | 541 |
| `PlannerDecisionTests.test_no_op_requires_nonblank_review_reason (reason='   ')` | Failure: expected ValueError not raised | 80 |
| `PlannerDecisionTests.test_positive_remainder_requires_reason (reason='   ')` | Failure: expected ValueError not raised | 121 |
| `PortfolioCompatibilityTests.test_benter_only_graph_preserves_benter_lane` | Failure: actual `experimental`, expected `benter` | 280 |
| `PortfolioCompatibilityTests.test_legacy_recipe_hash_keeps_pre_expansion_null_field_semantics` | Failure: actual `3e56d8c782e0f515`, expected `ce66143e3edf472c` | 303 |
| `tests.test_research_telemetry.ChampionTests.test_missing_protocol_identity_does_not_group_unrelated_attempts` | Failure: one grouped champion, expected two | 170 |
| `tests.test_research_telemetry.ComparisonKeyTests.test_different_feature_datasets_can_compare_on_same_score_population` | Failure: comparison keys differ for `dataset-1` and `new-features-same-runners` on the same evaluation population | 93 |
| `PackageIndexAlignmentTests.test_valid_race_probabilities_do_not_depend_on_dataframe_index` | Error: ValueError `Packaged probabilities must sum to one by race`, raised at `ima/research_model_package.py:148` for valid `[0.2, 0.8, 0.3, 0.7]` | 537 |

At this historical handoff deployment remained held; assigned source transfer and
fixture/build operations later occurred. Paid production launch remains held.
This role has no callable parent-agent message channel;
this shared owned report is the concrete parent handoff. No implementation/test
files were modified to resolve these failures.

## Protected V5 Baseline

Parent-supplied observation at `2026-10-04T05:48:00Z`: cycle 123,
432 complete, two failed, eight running, tracking pending zero.
`memory.current=90095411200`, `memory.peak=103083323392`,
`MemoryMax=107374182400` bytes. These are historical samples, not V6 results.

Read-only refresh at `2026-10-04T05:52:37.745783Z`: status `training`,
cycle 123, ledger 434 completed / two failed / eight running,
zero pending tells, zero pending tracking and no tracking errors. A separate
SQLite connection opened with `mode=ro` confirmed the attempt status counts.
Actual scheduler snapshot: eight reserved CPU threads and 32 GiB reserved RAM,
eight workers, 24-thread CPU budget and 80 GiB RAM budget. Source and scheduler
agree: one native thread / 4 GiB reservation per job. Reservation is not RSS.

| Protected identity | Value |
|---|---|
| Own account | `imaopt`, uid 1001 |
| Host | `100.95.24.121` over Tailscale |
| Campaign | `/home/imaopt/research-v2/campaigns/agentic_v5_discovery` |
| Release alias | `/home/imaopt/research-v2/live-releases/ima-v5-current` |
| Alias target | `/home/imaopt/research-v2/live-releases/ima-v5-0915b61` |
| Revision | `0915b619f6a40dcc390c3044107831293f826660` |
| Dataset SHA256 | `8f33d989cb258ecb45f16b49363e5a1b791b95a8ca9a40b8e22db7a8a64feb6c` |
| Protocol identity hash | `121ee03b31fd583072df255581aa21e530c4ff55ebfd82a88382f81519363ff8` |
| Environment identity | `5ee4ca2ef9503545b5cb5c15` |
| Portfolio | `feature-discovery-v5-fundamental` |
| Metric contract | `protected-development-v3-fundamental-v1` |
| Target contract | `research-targets-v3` |
| Service PID / restarts at refresh | 792999 / 1 |
| Effective high / max bytes | 103079215104 / 107374182400 |
| MLflow | `http://100.95.24.121:5000`, `/health` 200 OK, `/version` 200 `3.16.1` |
| MLflow experiment search | API 200, experiment `6`, `ima-agentic-v5-discovery` |

Later read-only cgroup sample: current 95,895,584,768 bytes, peak
103,083,323,392. Current-instance `memory.events`: high 787, max 0, oom 0,
oom_kill 0. Memory-pressure avg10/60/300 zero, cumulative some/full stalls
9,327,190 / 9,327,020 microseconds. Do not erase historical OOM incidents or infer
safe V6 capacity from these counters. Disk sample: 756,200,480,768 bytes available
on the own deployment filesystem. Dynamic counts, PID and memory can advance;
immutable identity and unit definitions are the protection comparisons.

## Protection Digests

SHA256 below refers to exact file bytes, unlike the canonical protocol identity
hash above. Read these files before and after any parent cutover; preserve them.

| File under own account | SHA256 |
|---|---|
| `research-v2/campaigns/agentic_v5_discovery/campaign-identity.json` | `1cde1d418a741dcf2c6c2d4fdea6ae14d681f8e1b4d2b03b7d4e024363df7e2e` |
| `research-v2/campaigns/agentic_v5_discovery/inputs/protocol.json` | `ba9e8bd670c3d18ee90d9b3ce006bc0254017c018b5248a8ff3483c9276bb4ea` |
| `research-v2/live-releases/ima-v5-current/deploy/systemd/ima-discovery-v5-supervisor` | `7a7d6f88a6d99da19b1e96097a0e9d57067d53edec96219b0e61cc4ad6d43f0e` |
| `.config/systemd/user/ima-discovery-v5-supervisor.service` | `ae05a36667cdafd7c4de2da5254981a6d36017c8cec3e049348d8189828bbf4b` |
| `.config/systemd/user/ima-hkjc-acquisition.service` | `4c62251ec90b498746b8e7f183b9e3886493b3e17ced06e9f465612604fa869d` |
| `.config/systemd/user/ima-hkjc-snapshot.service` | `ac86ca879d5ed6957f2a96901d1e7062fc7b0f027d63b8076b6b366090f60c6c` |
| `.config/systemd/user/ima-hkjc-coverage.service` | `1bb7904d2a4a4ae582099ba5d7003245eaa87efe7f776336fe7daa5bbfe832d5` |

Acquisition was active/running, coverage was activating and snapshot was inactive
at the initial refresh. Timer transitions are expected; compare definitions and
record any intentional own-user change. No other users' repositories or services
were inspected. Cortex/Solar health is therefore unverified by this role, and no
claim of a before/after end-to-end check of those systems is made.

Final role readback at `2026-10-04T06:04:02.217566Z`: all seven file digests above
matched, release alias target unchanged, V5 still `training`, cycle 123,
434 completed / two failed / eight running, zero pending tells/tracking. This
proves protected own-user files stayed unchanged during this role's audit; it
does not certify a later parent cutover.

Additional read-only protection baselines, captured after that comparison:

| Own V5 file | SHA256 |
|---|---|
| `campaigns/agentic_v5_discovery/ops/openrouter-config.json` | `4ca079b6c12faa6944d4fffe32e64b98a5722c5adf5769bd1f01cd8d730f353c` |
| `live-releases/ima-v5-current/ima/research_v5.py` | `52d872b85167358c376ce2bed8325ef563aa676927098f683bfb5a2fb8b3f1aa` |
| `live-releases/ima-v5-current/scripts/optimize.py` | `422263ddef8d159d142c2baf1c075cf6f56781e51a92dac1a620914bba248919` |

Paths in this additional table are relative to `research-v2`. Config bytes were
hashed without displaying its contents or credentials. Use these extra digests
for the parent's next before/after cutover readback.

## Dependencies And Deployment

### Historical Immutable Transfer Preparation And Tests

At this earlier mechanics-test stage the parent release SHA was not assigned.
Source upload/read-only extraction subsequently completed as recorded above.
No dirty source archive, V5 STOP or protected service mutation was performed. Supervisor success
receipt now includes the exact `committed_tree` mode/blob map derived from
`git ls-tree`. Runbook packaging uses the verifier FROM that committed SHA,
not an uncommitted workspace launcher; SHA/receipt arguments are validated and
the future receiver independently verifies hashes, PAX, blobs, executable modes,
file set and post-extraction content before read-only release metadata.
Target read-only `git --version` returned 2.53.0; no target Git repository is
needed. Transfer shell `bash -n`, supervisor syntax and diff whitespace pass.

Local mechanics test uses committed baseline
`6ef7481dc93f406e77174d510b36d99155ef38b4`: 1,515,520-byte archive, 178 committed
files. The actual supervisor reports source integrity passed but required V6
readiness failed (15 missing paths). First baseline attempt including newer
uncommitted doc pathspecs failed Git archive with exit 128; test scope was then
limited to existing COMMITTED entries, never filled from the workspace.
Receiver tests use an explicitly synthetic readiness flag only to exercise
transfer/extraction, with a canonical temporary root substituted for the remote
root. This does NOT certify a V6-ready archive or permit transfer.

Final mechanics result: one valid local extraction/read-only metadata fixture
passed; 11 rejected fixtures: bad receipt SHA, archive SHA, existing destination,
failed readiness, PAX revision, blob content, executable mode, symlink, traversal,
omitted file and duplicate file. Rejections occurred before destination creation
(existing destination sentinel preserved). Fixture artifact/test-results path:
`/private/var/folders/wq/tn0d0r417cb3kghxltyqhjt80000gn/T/ima-v6-transfer-test-bkjr36wu`.
Initial harness argument-count/macOS `/var` symlink-path errors were corrected
without weakening the receiver. Positive complete-V6 committed package and
actual Tailscale upload/extraction subsequently passed under the exact assignment
recorded above; these baseline mechanics tests retain their original identities.

### Guarded Build: Read-Only Feasibility

At `2026-10-04T07:23:58.863161+00:00`, own user manager was running, systemd
259.5, CPU/memory/pids delegated. Host physical memory 130,391,482,368 bytes,
MemAvailable 29,046,710,272 (includes reclaimable cache), MemFree 5,423,894,528;
28 CPUs/load averages 9.64/9.84/9.45. V5 current 98,719,813,632, peak
103,083,323,392, max 107,374,182,400. V5 remains active PID 792999/invocation
`0e74db4db0fa4f05bf8a308e200b022c`. Own manager ancestors' memory/cpu maxima
are unlimited, so no observed ancestor ceiling prevents a sibling build unit.
Historical manager-ancestor OOM/kill counters 16/7 are nonzero; host full memory
pressure avg10/60 was 0.06/0.08. No transient unit was created as a probe.

Runbook proposed limits are MemoryHigh 6,000,000,000 / MemoryMax 8,000,000,000
decimal bytes, swap 0, CPUQuota 200%, native threads 1, runtime 3600 seconds.
Admission reserves another 8,000,000,000 host bytes plus possible V5 growth
to its own max. At the initial sample the required available total was
24,654,368,768, leaving 4,392,341,504 above that allowance. Thus provisionally
feasible, NOT proof a full build fits 8 GB or is safe later.

Exact documented guard was subsequently executed read-only over SSH: two samples
five seconds apart, exit 0. MemAvailable 30,869,491,712 / 30,692,278,272;
required 24,742,957,056 / 24,740,732,928; V5 current 98,631,225,344 /
98,633,449,472; host full-pressure avg10 0/0. Both were admissible at sampling.
Require a fresh two-sample pass immediately before any assigned build.
Local `scripts.build_research_dataset --help` passed; exact request/registry/
source-snapshot/raw-manifest flags are in the runbook. Committed source,
approved request/output and Linux import/readiness proof remain prerequisites.
Actual transient-unit creation/limits enforcement and official build exit,
peak, immutable output/coverage/protocol validation remain UNTESTED. Neither
V5 mutation nor campaign launch is authorized by this feasibility result.

### Authorized Private Setup Receipt

Completed `2026-10-04T06:49:26.590593+00:00`, only under
`/home/imaopt/research-v2/v6-dependencies`. No sudo, V5/acquisition modification,
direct server download, source upload, STOP or service action. Actual changes:
fresh `venv/` generated with own Python 3.12.12 `--without-pip`; 27,197 base
files and 3,990 overlay files copied without hardlinks/old scripts; generated
`preserve.constraints`; locally downloaded `wheels/` (95 files, 59,588,314 bytes)
transferred with `scp` over Tailscale; private pip installed only JupyterLab
4.6.2/Graphviz 0.21/Plotly 7.1.0 plus missing closure. Target no-index dry-run,
install and pip check all exited 0. All original 120 effective versions match;
189 effective distributions means 69 additions, not upgrades. Pydantic remains
2.13.5/core 2.46.5: approved copy mode, explicitly not research-lock parity.

Copy integrity verified per-file SHA256 before repair: base ordered path/content
hash `cfae0f9b9772cb39732624ea113a4f15ddb55844ca3888218c29cbc491f49837`,
overlay `915f228d12b79d8a9e769e305a04c7a6381da6e51c5d867ba4cdc8c0c1099c1f`.
All 25 tested research/features/repair/native module origins and all effective
metadata paths are private; prefix is private and sys.path has no V5/acquisition
package directories. Executable resolves to the unchanged own managed Python.
Setuptools/Woodwork emitted its existing deprecated `pkg_resources` warning;
no upgrade was made. Ten protected hashes match; V5 active, PID 792999,
invocation `0e74db4db0fa4f05bf8a308e200b022c`, STOP absent.

Exact private content inventory: `evidence/environment-files.json`, 46,947
file/link entries, 1,466,981,855 regular bytes across venv/overlay/wheels and
constraint file. Generated cache and evidence receipts are separately scoped
inside this root; nothing was written to another remote path.

| Evidence under private `evidence/` | SHA256 |
|---|---|
| `setup-receipt.json` | `3a108b4002b5c8f17dce43912f15e09ac41dae2bd6e78b8bda76185c3fb90ed6` |
| `environment-files.json` | `2dd0482463a56e33c84dc11fe2174469a5cc2451efaa7c0ecd4bad0d9311ab4c` |
| `resolved-effective.json` | `11d9d339cc1903a37763eb0aed7d52ea422507db284464d0f2b31a7ada045092` |
| `resolved-effective.lock` | `87794eb50cc8e4d39c37c505632177889be470f7ed8a26ae3b46a47a151cfc19` |
| `import-origins.json` | `0c7c4570931fcc7c3b77ce2f2aa7d7dedd5cf6a273ccef7c1d557babe654d811` |
| `wheel-manifest.json` | `d91a28c1fa3c8ed768c2366f6097639f4b16662940eacdcf00e4ff8dddc4c33f` |

`copy-integrity.json`, baseline/post protection JSON, offline dry-run/install
reports/logs, `pip-check.txt`, `pip-freeze.txt` and baseline effective JSON also
exist; their checksums are indexed by `setup-receipt.json`. This is a ready
private dependency base, not full acquisition/browser extras, code deployment,
replay on official races or a running V6 service. Preserve this resolved identity
for later committed-release canary gates; do not silently apply partial locks.
Local receipt readback at the runbook's artifact root verified all 15 indexed
checksums and the setup receipt SHA256. Bash syntax, CI YAML/read-only permissions
and decimal service limits passed after these updates. No exec sessions from
dependency setup remain running.
After compaction, read-only remote refresh again returned `No broken requirements
found.` with exit 0. Existing `setup-receipt.json` SHA256 remains
`3a108b4002b5c8f17dce43912f15e09ac41dae2bd6e78b8bda76185c3fb90ed6`;
`pip-check.txt` receipt SHA256 remains
`9261363b733079a641c2e4cc9bc46ffa1d8336945a87f807b6cf68847dbc9b09`.
Refresh created no remote files or source/service mutation.

### Parent Fixture Readback: Not Launch Acceptance

Parent reported new canary CLI exit 0 and two workers. This role read
`.tmp/v6-canary-integration-3/campaign/status.json`: update
`2026-10-04T06:44:59.212117+00:00`, mode complete, three cycles/decisions,
10 completed, zero pending tells, 10 pending tracking (expected offline).
Program completed counts total eight Benter/two boosted. No advanced generated
feature, paid planner, official dataset, uploader/UI or live recovery proof is
implied. Parent separately reported 32 legacy-controller tests passing after the
cache helper fix; that exact 32-test command was not independently rerun here.
Readback of the updated canary code confirms complete mode/exact count,
zero failed attempts/pending tells, and zero pending tracking when MLflow is
configured are now enforced. This does not imply offline entries were uploaded.

Subsequent local tracking canary `.tmp/v6-canary-tracking-1` uses its own
`mlflow.db`, not the remote pinned MLflow service. This role read status JSON
and both SQLite databases with URI `mode=ro`: status updated
`2026-10-04T06:52:09.433022+00:00`, complete, five completed, three cycles,
pending tells 0, pending tracking 0. MLflow database has five FINISHED runs,
five READY model versions (all five join to their run IDs), four OK traces.
Status SHA256 `10cc8eb64735da19f4045b5de61138fe588b4ce1ee93f4ebceb1f25105f24fc6`.
This verifies the observed local tracking/registration/trace endpoints and empty
pending outbox, not paid planner costs, native USD UI, publication-safe official
datasets, advanced graph/features, committed Linux execution or live recovery.
V5 remains outside these fixture paths. No live launch is authorized yet;
official dataset/controller gates remain held.

`.tmp/v6-replay-smoke/replay-report.json` readback has two records with Benter
and boosted package manifests, each 120 scored runners, prediction parity true,
prediction/objective deltas 0; parent reports rtol 1e-10. Report SHA256
`6f89b522a5c30edb8fb556373f2b1bc361542af876989be9a8db05caefc4527a`.
Source is `.tmp/v6-canary-integration-2/campaign`, not the pinned live V5.
Replay lineage code/environment is `unknown` and automatic promotion false;
this is fixture parity only, not committed-release or V5 champion replay proof.

### Earlier Read-Only Inventory

Read-only refresh `2026-10-04T06:26:24.957593Z` confirms Python 3.12.12/Linux
x86_64. Effective 120-distribution version/metadata-path manifest SHA256:
`97dd8c60d6edda3180a9ae0a586c6d98a1f6c837f8f0ae2f5570da9608bf5bd9`.
Resolved base purelib under release `2868bec` has 27,197 files / 1,150,969,295
bytes, no links/hooks; V5 overlay has 3,990 files / 28,858,761 bytes, no links,
only the standard setuptools `.pth` shim. The advertised `f54c604` venv shares
that base; release names are not environment isolation.

Actual effective remote `pip check` exit 1 exact findings:

```text
ima-racing 0.1.0 requires jupyterlab, which is not installed.
catboost 1.2.10 requires graphviz, which is not installed.
catboost 1.2.10 requires plotly, which is not installed.
```

Read-only offline `pip install --dry-run --no-cache-dir --no-index` with both
current research/overlay locks and existing research/V5 wheel stores exited 1:
Pydantic 2.13.4 was not available. Its required core is 2.46.4, versus effective
V5 2.13.5/core 2.46.5. Missing acquisition extras are Scrapy/pypdf in the effective
research environment; pypdf was not found in the audited acquisition wheel store.
Playwright is a separate UI gate, not headless runtime. The runbook records exact
future wheel preparation, Tailscale copy and isolated no-index install commands.
Those strict-lock/acquisition commands have NOT run. Subsequent user assignment
instead authorized a fresh private copy of the audited effective environment,
preserving all 120 versions; prefix and metadata-origin checks passed. Its only
permitted repairs are JupyterLab/Graphviz/Plotly closure, downloaded locally and
transferred through Tailscale. At this historical inventory stage no source was
uploaded; subsequent exact-source transfer and fixture/build units are recorded
in the authorized checkpoint above, alongside the completed dependency receipt.
Local repair candidates JupyterLab 4.6.2, Graphviz 0.21, Plotly 7.1.0 are observed
local versions, not target compatibility receipts. Require a complete Linux
wheel closure, own manifest/hash/import-origin readback and successful private
`pip check` before acceptance. Never repair this drift in live V5.

At `2026-10-04T06:26:26Z` own env/routing refresh still showed public models HTTP
200; OpenRouter addresses 104.18.2.115 and 104.18.3.115 via `tailscale0` table 52,
source 100.95.24.121 uid 1001. Exact own env source is
`/home/imaopt/.config/imaopt/openrouter.env`, mode 0600 uid 1001, key present
without value/length output. No authenticated paid call was made. Parent gates
were prerequisites at this historical readback. Setup/source/fixture/build were
later assigned and performed; paid production launch remains explicitly held.

V5 interpreter: `/home/imaopt/research-v2/releases/f54c604/.venv/bin/python`,
Python 3.12.12. Overlay: `research-v2/v5-dependencies/site-packages`.

| Dependency | Remote installed version |
|---|---|
| NumPy / pandas / SciPy | 2.5.3 / 2.3.3 / 1.18.1 |
| scikit-learn / Joblib | 1.9.1 / 1.6.0 |
| MLflow / Optuna / psutil | 3.16.1 / 4.5.0 / 7.2.2 |
| Pydantic | 2.13.5 |
| CatBoost / LightGBM | 1.2.10 / 4.7.0 |
| Featuretools / feature-engine / Woodwork | 1.31.0 / 1.9.4 / 0.31.0 |
| httpx | 0.28.1 |

Pinned V5 `requirements-research.lock` SHA256:
`2c961dae392342503f3ab7053e0f27e21160d7beafe9dac5b38667332fcec310`;
overlay lock SHA256:
`1de397463e2dd4d3a827a20ac613bc97501394195bd57f8ad980920b60e34126`;
`pyproject.toml` SHA256:
`98dd1654d763e75daa70a336d3496bc751b3a57b2b25e36e5b105c7732401523`.
Existing research lock pins Pydantic 2.13.4; live V5 reports 2.13.5. Record this
drift; do not fix it inside V5. Existing lock files are partial and do not lock
all transitive dependencies. CI installs their explicit pins plus existing extras
and prints resolved versions. A V6 release still needs its own resolved manifest,
`pip check`, environment identity and replay evidence. No blanket upgrades or new
distributed scheduler are needed for this role.

## Dataset And Protocol Comparability

Read-only official pointer:
`/home/imaopt/acquisition/hkjc-20261002/snapshots/latest_verified.json`.
At refresh it pointed to `snapshot-20261004T050721Z-122062e8`, verified at
`2026-10-04T05:25:04.142934Z`, parser `official-corpus-v14`:
14,488 races / 178,718 runners; 173,884 previous baseline runner keys retained.
Input digest `84f6b3eb8c2315eda49f2d41d4acb0df6e127f4d3eeef3bc4fe603d5beb9ecb6`
identifies acquisition inputs at job start; immutable dataset identity is its
manifest and lineage. Latest is a pointer, never a campaign identity.
Independent readback reports 14,471 feature races and 138 attrition rows;
raw retention does not imply every raw race enters training. Its verdict is
sampled roster readback, not exhaustive website coverage or a temporal/leakage audit.

V5 protocol is `min_train_races=17000`, `calibration_races=1000`,
`score_races=1000`, `max_folds=3`. Even the raw 14,488-race count cannot provide
the minimum training block; the first complete scoring fold needs 19,000 races.
No silent minimum reduction, replay claim or cross-contract score promotion is
valid. Freeze a separate V6 protocol against eligible whole-race counts and date
boundaries; report exclusions, coverage and common race intersection. Preserve
original V5 champion replay on its original dataset/protocol/environment. Label
retraining or scoring controls on the V6/common population as new comparisons.

## Remaining Launch Gates

| Requirement | Required concrete proof before acceptance | This role's result |
|---|---|---|
| Events and speed sources | Publication/coverage-aware joins; race/workout/trial units; whole-race quality audit | Not independently verified |
| Dataset lifecycle | Request -> validated immutable artifact; failed build cannot promote; baseline retention | Fresh V4 independently accepted by parent/Sagan; V2 cancelled and V3 unaccepted artifacts retained |
| Features and selection | Novel formula reuse; >32 eligible selected; trained-only selectors; leakage rejection | Unverified |
| Graphs and performance models | Chronological OOF, controls, scale/identification, valid joint outcomes and packaged replay | Unverified |
| Controller and budget | Planning during busy fits; continuous refill; idempotent restart and retirement | Legacy scheduler observed; V6 unverified |
| MLflow | Paid plan and linked snapshot; native USD field API/UI; one physical call charged once; current comparable champions | Historical fixture backend passed; paid 52a private trace OK but USD/cost unknown, acceptance failed; no production write |
| Paper EV and Kelly | Typed controller-selected action plus quote/settlement/portfolio/no-lookahead proof | Parent local ten-trial paper-study v2 passed; final controller commit and fresh Linux --paper-study fixture pending; no bets submitted |
| Performance | Equal concurrency cold/warm repeated full-history benchmark, cgroup peak/PSS/USS/I/O/pagefaults | V5 resource baseline only |
| Recovery | Builder/controller/worker/uploader restart drills; no duplicate spend/allocation/registration | Not run |
| Deployment | Exact release/data/protocol/environment identities; multiple real decisions and completed graph; protection readback | Parent-owned, pending |

For each gate record command, revision, UTC timestamps, exit code, artifact path
and readback. Fixture success cannot certify historical coverage, native USD UI,
full-history throughput or live recovery. Missing historical publication evidence
and pre-race exotic quotes remain source limitations when observed; unavailable
controller paths or failed tests remain implementation gaps. Keep those separate.

### Parent Integration Checkpoint 2026-10-04

The committed implementation passed 629 local tests in 121.491 seconds after the
final dead-heat quarantine guard. The preceding full pass was 628 tests; earlier
failed runs exposed real restart, publication, identity, JSON null and taint
propagation faults and are not release evidence. Focused actual executor/data
integration also passed 39 tests. Registered feature definitions build verified
successor datasets and replay from packages; these remain provenance fixtures,
not proof that the complete official corpus has been built or trained.

An actual read of the latest official runners file found 31 multiwinner races
and zero missing-winner races. The current conditional-logit contract rejects
multiple winners, so the registry now explicitly excludes complete dead-heat
races and retains original evidence upstream. All publication, quality,
population and confirmation exclusions must be read back after the full build.
The registry protects confirmation keys before filtering, binds numeric and
categorical eligibility to dataset identity, and records unsupported delayed
outcome publication rather than inventing historical availability.

Paid API receipts are authoritative for cost, including rejected/repair calls;
unknown cost freezes further paid planning. Queue-aware native-thread allocation
avoids reserving four cores for every fit when a large trial queue is ready.
Actual host/cgroup admission and progressive capacity still govern concurrency.
No V5 stop, V6 launch, real betting, or other-user modification has occurred at
this checkpoint. Remote Linux execution, actual official dataset readback,
champion replay and paid planner/trainer/MLflow feedback remain rollout gates.

### C1 Nonbilling Transport Boundary 2026-10-04

Runtime remains `c1eb224fffd302e47f450b90c233b44a97c9193f`; later docs commits
do not repin it. The second isolated paid proposal failed with ReadError before
response headers. No raw provider response or generation ID exists. Normalized
and native trace cost are null, not zero; both unknown-spend incidents remain
frozen, with no paid retry or continuous launch.

Completed unauthenticated diagnostics used only synthetic JSON at
`https://openrouter.ai/api/v1/ima-nonexistent-transport-diagnostic`, never the
model endpoint. Each request had an absolute 60-second bound. Models GET returned
200 in 4.285s. The 256-byte POST returned 404 in 11.211s; body-send duration was
0.0023s. A 250,000-byte POST body took 31.050s to send, then failed with ReadError
before headers at 34.058s. A matched 202,674-byte synthetic body took 35.286s to
send, then failed before headers at 37.289s.

Offline reconstruction from frozen C1 evidence/code using installed HTTPX JSON
serialization gives Content-Length 202,674 and body SHA256
`44b777fef7a3aaca51511c852eb1726105fec33737538f90e89c27d2290f0e5a`.
The evidence file is 207,352 bytes. This is a reconstructed HTTP body, not a
packet capture or total TCP/TLS wire-byte count; the receipt's bare Request
header-byte subtotal is not the actual client's complete headers.

Both resolved IPv4 and IPv6 OpenRouter addresses route through tailscale0/table
52. Interface MTU is 1280; all eight HTTPX proxy environment flags were false.
All ten protected hashes remained unchanged. No routing, MTU, firewall, security,
V5, or other service changes were performed. This reproduces a size-associated
non-model upload failure; it does not prove an MTU fault, isolate the exit/edge,
establish model-endpoint health, or reconcile unknown paid spend.

Receipt: `/home/imaopt/research-v2/canaries/acceptance-runner-owned-c1eb224f/operator-evidence/noncharged-transport/terminal-receipt.json`
SHA256: `2020a0295c5cf7f04f7d0f0a284f58c927714c57935563a201d2712f55ba1cda`.
Diagnostic unit exited 0; all owned exec sessions completed. Live/paid gates
remain blocked pending independent cost reconciliation and transport resolution.

### OpenRouter Lane-Ready Successor 2026-10-05

This checkpoint supersedes the transport/startup blocker above for a new
campaign; it does not reconcile or overwrite the older unknown-charge calls.
The scientific release is pinned to
`8d6110204591fee492388b84252a2877b3df967e`. The campaign is
`/home/imaopt/research-v2/campaigns/agentic_v6_openrouter_ready_8d61102`, owned by
`imaopt`, under `ima-v6-openrouter-ready-8d61102.service`. The preceding
`ima-v6-openrouter-7da5b0d.service` is stopped, with its two completed models,
paid receipts and explicit operator-recovery history preserved. No scientific
release was hotpatched, and V5 remains stopped.

Root cause: after two Benter completions, the 80/20 dispatcher required an
experimental trial, but both preparation slots held expensive Benter programs.
The immutable successor prioritizes uncovered lanes and inexpensive references
before costly discovery. Existing lane dispatch, budgets and resource admission
are unchanged. Regressions cover startup, restart, exhausted readiness and fair
ordering after lane coverage. Local full suite: 763 passed. Remote Linux focused
suite: 43 passed, one optional skip. Both GitHub CI runs succeeded. The deployed
archive verified all 245 committed files; protected hashes and scientific
dependency versions were unchanged.

The exact first request body was 90,621 bytes, SHA256
`44a3f992627cd5a2a4a301a8fe09634c3676dc8102c76fbc36fc76bce07e5bb7`.
Three bounded unauthenticated synthetic uploads of that size succeeded before
one paid qualification. No research contents or credentials were sent to the
diagnostic endpoint. Transport receipt SHA256:
`ad12a420e99d1e69553ee5b59b165713dffab0b742ef5508faeaaaaf37cd3d0b`.

The full 26,931-byte handoff memo is included in planner evidence, not merely a
path or summary. Its SHA256 is
`bcdd074861678857760fac3056bef6c65f1f888d1600c6cc6dfc72bdb1fd8b5e`.
Both accepted decisions acknowledged this exact version and used its prior
research in their hypotheses. Acknowledgment verifies delivery/version, not
human-like comprehension. Previous V6 results are also read-only references.

- D000001: one physical call, six accepted programs, 24 chosen/allocated trials,
  reported USD 0.0182244, backend trace
  `tr-7b2013003f01d70294a69ba21d4cba2c`.
- D000002: automatic controller decision, six accepted programs, 15
  chosen/allocated trials, reported USD 0.02557164, backend trace
  `tr-57e5c0861076a8a8d17c14e09a46b93e`.
- Both backend traces are OK with native USD matching durable API receipts;
  cumulative known spend at this checkpoint is USD 0.04379604.
- The service has completed trials in both lanes. Benter run
  `d647149b31c140f498c67417ff4f57e0` has fundamental log loss
  2.174423729207687, registered version 16. Boosted experimental run
  `e3d5b94e8fa44b789cbeb3ca7a01a263` is registered version 17.

The Benter score is below the matched strict control 2.176278241358009; this is
a development-score improvement, not statistical significance or market alpha.
Do not compare it directly with legacy V4/V5 scores from different populations.
The accepted dataset remains `dataset-19aaa959...a83db` with the frozen strict
protocol and protected confirmation population.

The actual kernel envelope is 24 CPU threads, 90/100 decimal GB high/max and
zero campaign swap, with no tighter ancestor cap. Fit capacity progressed
2 -> 4 -> 8 after measured folds/headroom; 26 is a ceiling, not an actual
concurrency claim. No OOM events or tracking errors were observed. Model uploads
are asynchronous: transient pending uploads are not failed training, and each
registered model requires independent backend readback. The owned service is
enabled for its user's boot target, without modifying other users or services.

Planner model is `deepseek/deepseek-v4.1-flash`, medium reasoning, 32,768 output
tokens, 300-second deadline, no provider pin or OpenAI flex setting. The agent
chooses budgets up to 260; the campaign has no total trial/time limit. The USD 5
prior-spend admission threshold is not a guaranteed provider billing cap.
Missing historical event availability, independent time-target identification,
and absent actionable historical exotic quotes remain data limitations; this
rollout is not proof that every research-plan acceptance item is complete.
