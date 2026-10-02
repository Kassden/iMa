# Official HKJC Acquisition Validation

## Boundaries
- Branch: `feat/hkjc-official-acquisition`. Additive acquisition work only; no implicit optimizer dataset switch.
- Remote ownership: `imaopt`, `/home/imaopt/acquisition/hkjc-20261002`. Isolated Python/Scrapy venv, transferred wheels and releases over Tailscale.
- Only own acquisition, coverage and snapshot user units were installed/reloaded. V5 supervisor remains PID 792999; other users/Cortex/Solar untouched.
- Source: public `racing.hkjc.com` allowlisted racing routes. Offsite/account URLs and malformed queries are rejected. Access denial stops acquisition; no bypass or IP rotation.

## Verified Results, 2026-10-02
- 63 focused local tests pass, covering identity, field completeness, date/venue/race-number mismatches, current-result header recovery, pagination, result/trial-selector navigation, null census venues, trial context, profile metadata/conflicts, sectionals, workout corroboration/equivalence, legacy layouts, exact baseline retention, snapshot-refresh failure/exclusive-owner gates and rich-feature regressions.
- Release-r15 acquisition tests: 27 pass in the isolated server venv. A real normal service stop wrote closed.json before exit, and the persisted queue resumed on the next immutable release. Coverage-r2 passes 29 tests; snapshot-r4 passes 45 tests in its separate actual server environment.
- Full 14,091-cache filename/header audit: 14,091 matching race numbers, zero mismatches. An earlier spot-check mistakenly used `R1*`, selecting R10; that was an inspection error, not observed cache corruption.
- `snapshot-007`: full raw replay with displayed meeting/date/venue/race-number checks; 14,095 races, 173,870 runners, 9,040 horses, 356 explicit nonfinishers/disqualifications. All 14,091 original cached race candidates pass full-field validation; four additional 2001 pilot races are included.
- Features: 14,083 races / 173,768 rows. The 102-row difference is reported; feature-market validity is separate from successful raw acquisition.
- Date range: 2001-04-14 to 2025-12-27, with only four pilot races before the original 2008 cache boundary. This does not establish complete historical coverage. The directory name `hkjc-2005-2025` does not prove 2005 coverage.
- Snapshot files, source blobs and captured code hashes are retained. Previous snapshots are not overwritten. Snapshot-007 independently passes artifact/code hashes, normalized probabilities, whole-field retention, 12 stratified meetings and 20 raw-roster horse identities.
- Snapshot-007 contains 47,741 normalized supplementary source-event facts, none used in historical training without publication evidence. Source facts are not necessarily unique physical events.
- Dated horse ratings recovered: 169,785 of 173,870 source runners, versus 115,149 in snapshot-006. The 54,636-row gain follows legacy title/brand parsing fixes, not imputed ratings.
- Country of origin: 173,819 attributed runners, zero observed country conflicts. Mutable age/sex/trainer/current rating stay in captured profile snapshots, not retroactive historical predictors.
- Venue and distance: zero missing values in both source runners and feature rows.
- Snapshot-008 independently passes the same hashes, probability, full-field and stratified raw-roster readback: 14,096 races, 173,884 source runners, 9,041 horses, 14,084 feature races / 173,782 feature rows; date maximum is now 2026-10-01. It has zero excluded race/page candidates and 67,886 supplementary source facts. Historical publication remains unproved; these supplementary events are not used in the historical feature matrix.
- Snapshot-008 corroborates 1,118 daily workout identities and identifies 1,118 repeated source observations across formats, retaining all source evidence. There are still 26,339 unresolved supplementary identities; no name-only joins are accepted.

## Source Discoveries and Recovery
- Daily workout HTML is a JavaScript shell. Public paginated JSON supplies the actual rows, with 50 records per nonterminal page and verified advancing `next` links.
- Complete captured pagination chains: September 30 (1,307 rows), October 1 (1,555), October 2 (1,314): 4,176 rows. These are complete as captured, not a promise that the source cannot update later.
- By 06:24 UTC all 21 server daily chains were complete as captured, totaling 28,903 rows. These are source snapshots, not immutable daily publication guarantees.
- JSON daily workouts are name-only. A real replay found 268 records corroborated against separately captured official horse workouts using exact name/date/type/track/workout/gear. This is a replay count before dataset deduplication, not 268 new horses.
- One actual sectional page parses all 14 runners with full IDs, positions, margins, main splits and optional 200-metre sub-splits. Sample: SIGHT DREAMER, HK_2023_J542, 13.72 / 21.71 / 23.45 / 23.11 seconds on 2025-07-13 ST race 1.
- Snapshot-005 readback passed: captured code/artifact hashes, normalized probabilities, whole-race feature retention, 12 stratified meetings across both venues and 20 identities independently matched to raw result rosters.
- Cached forms for 922 horses yielded 3,589 actual official result hrefs before 2008-04-02. Four 2001 pilot races recover 55 runners after adapting the official 11-column table layout; missing running positions remain missing.
- All 31 previously quarantined cached races pass current full-field validation: 28 involved TNP; three involved numeric placings with missing numeric attributes. Missing attributes are preserved, not invented. Snapshot-006 and snapshot-007 readbacks passed.
- Current-results landing page raw replay now recovers 2026-10-01 ST race 1, 1,200 metres and 14 runners from displayed headers despite missing URL context. Conflicting explicit parameters and ambiguous headers remain rejected. Snapshot-008 certifies this recovery with parser v11.
- Identical full-ID/date/name/type/track/workout/gear observations receive a shared equivalence key while retaining their source records. Name-only observations never get a horse-based key; equivalence is not a guarantee of distinct physical-event counts.
- Barrier date formats are route-specific. A previously attempted URL returned a different day's data and remains quarantined by displayed-date verification. A new actual public Search click resolves 09/09/2025 to `/archive/btresult?Date=2025/09/09`, and independently parses 31 events dated 2025-09-09. That verified archive route is used for the 178 discovered selector dates; do not assert year-first dates are universally invalid.
- Parser v12 recovers trial batch context from source headers: the 31-event pilot has SHA TIN ALL WEATHER TRACK, distances 1,200/1,050 metres, going WET SLOW, winner times and sectionals. Each runner's own time is a separate field; source batch text remains retained. Unknown batch context is not filled from a previous batch. Older snapshots remain unchanged.
- Parser v13 preserves recognized profile labels including sire, dam, dam's sire and import type; conflicting labels are quarantined. These are capture-time profile attributes, not backdated training features or inferred related-horse IDs.
- Exact meeting-navigation audit identifies missing race numbers relative to observed same-date/same-venue official links. It does not invent a complete historical fixture denominator.
- Official result-date JSON currently returns 164 candidate dates with null venues. The rendered selector confirms blank venue values, while individual race entries contain explicit Sha Tin/Happy Valley and distance. Candidate dates are not an exhaustive local-meeting census.
- Discovery amplification: a real public Search click for 23/09/2026 navigated to a date-only results URL. Independent page parsing recovered HV race 1, 1,650 metres and 12 runners; observed sibling links cover races 2-9, sectionals and linked horse histories. The renderer exported 162 nonfuture date leads with control evidence; unknown/nonlocal dates are not accepted as local meetings.
- Date-only meeting leads are prioritized at 75,000, verified same-meeting race/sectional links at 65,000, archive result roots at 50,000, and verified workout pagination at 200,000. These are scheduling priorities, not coverage/quality scores.

## Operational Readback
- Acquisition release-r15 is versioned separately from earlier releases; JOBDIR is resumed after graceful SIGINT, without concurrent owners of the same queue. New captures identify a hash-addressed collector manifest containing code, seed inputs, Python/Scrapy versions and segment limits.
- Normal stop at 06:17 UTC wrote closed.json with 1,028 captured pages before the parent exited. Release-r13 resumed and reached 1,032 pages by 06:18 UTC. V5 remains PID 792999. These services are imaopt user units, not system-wide units.
- Verified daily seeds and pagination have high priority above the persisted general frontier; archive result seeds have priority above broad horse traversal. Pending JSON chains can continue immediately after new date seeds are captured.
- Polite acquisition: domain concurrency 2, delay 2 seconds, AutoThrottle; CPU quota one core, memory maximum 2 GiB. Acquisition observed near 120 MiB, not the training campaign's large memory allocation.
- Read-only gap audit runs every five minutes. It hash-checks and reparses cached raw with the current adapter; it does not rewrite corpus documents or training data.
- Remote report: `/home/imaopt/acquisition/hkjc-20261002/coverage.json`.
- Local reports: `data/historical/hkjc-official-corpus-20261002/coverage-*.json`.
- Autonomous dataset refresh: own `ima-hkjc-snapshot.service/.timer`, separate immutable snapshot releases and `snapshot-venv`, 2-core CPU quota / 8 GiB memory cap. Dependencies copied from the existing own-user environment into the new environment; missing Scrapy dependencies installed offline from previously Tailscale-transferred wheels. Live research and acquisition environments were not changed. The server versions are separately manifested (numpy 2.5.3 / pandas 2.3.3 / pyarrow 25.0.1), not represented as identical to the Mac environment.
- First full server snapshot-r1 build passed independent readback at 07:12 UTC: 14,140 races / 174,430 runners, 27 minutes 24 seconds, 4.3 GiB peak RAM. It nevertheless omitted 41 locally verified runners from three older pilot races and the local sectional pilot: higher row count was not a superset proof. A real baseline check rejects that snapshot with the exact 41 missing race-and-horse keys.
- Five original local pilot capture roots are imported separately, preserving raw bytes/documents/collector manifests without modifying the crawler JOBDIR. The next snapshot invocation includes those roots and requires exact retention of all 173,884 local snapshot-008 runner keys before updating the verified receipt.
- Snapshot-r4 canary passed actual independent readback at 07:34 UTC: 120 races / 1,496 runners, all four 2001 pilot races (55 runners), 14 sectional records and 79,343 supplementary facts including 9,034 trial records. Zero supplementary facts enter historical training without publication evidence. This is a canary, not a full historical superset or V5 promotion.
- The in-flight snapshot-r3 full build is left unchanged. Updated snapshot-r4 unit configuration affects only a subsequent invocation. Full baseline-gated publication remains pending until its real readback succeeds.
- Dataset refresh fingerprints raw/normalized inputs, normalization code and dependency versions; unchanged inputs are skipped only after verifying the previous snapshot. Input digest is a job-start rebuild trigger, not an assertion that acquisition stopped during a build. Each immutable dataset's actual identity comes from its own manifest and raw lineage. A failed build/readback cannot update `snapshots/latest_verified.json`; no automatic V5 promotion exists.

## Not Complete Yet
- Exhaustive official meeting/race denominator, particularly before 2008 and during 2026.
- Historical availability/publication proof for workouts, veterinary events, trials and movements. First capture is conservative future availability, not permission to backdate features.
- Full retired-horse workout/veterinary recovery: some legacy requests redirect to profile pages, which does not establish event coverage.
- Complete sectional/incident/trial archive coverage and independently stratified sample readback.
- All public-source Benter feature refinements and complete mutable-attribute history. The paper does not publish every proprietary factor.
- Acquisition is running and measured coverage improves; this is not yet an ULTRA COMPLETE dataset or a completed promotion gate.
