# HKJC Official Corpus and Benter Data Coverage

## GOAL
- Collect an extensive, resumable HKJC-only racing corpus; recover retired identities through official links; normalize it into versioned point-in-time datasets outside the agentic optimizer.
- User authorized implementation and iteration on 2026-10-02. Keep current V5 data/code/services pinned; publishing an acquisition dataset does not authorize switching a campaign to it.
- Benter's paper describes feature families and selected examples, not every proprietary definition. Report faithful implementations, proxies and missing sources separately; never certify exact full reproduction.
- User strengthened the goal to ULTRA COMPLETE on 2026-10-02: this is an evidence-driven recovery program, not one bulk download. Revise adapters/frontiers and the plan whenever a measured gap exposes a better official route.

## Ultra-Complete Recovery Loop
1. Establish the expected universe from official calendars, result indexes, horse form links and trial date selectors; report unknown denominators instead of inventing completeness percentages.
2. Maintain a gap ledger keyed by family, season, venue and cohort, including expected/fetched/parsed/identity-resolved/temporally-usable counts and reason codes.
3. Prioritize recoverable high-impact gaps: missing race starters, missing race-dated ratings, retired identity routes, real workout/trial histories, sectionals and incident context.
4. For each gap inspect raw HTML and rendered public controls/network requests. Add only verified official endpoints; register fixtures, pagination/date/identity checks and provenance before expanding collection.
5. Reparse cached pages with the corrected adapter before fetching again. Preserve old snapshots and collector revisions; compare coverage deltas and source disagreements.
6. Promote into a new immutable dataset only after full-field, identity, point-in-time, contamination and reproducibility gates pass. Never swap the active optimizer's dataset implicitly.
7. Continue while a recoverable gap or new official frontier remains. Exhausted/unavailable/denied sources remain explicitly unresolved; neither a running daemon nor 99.8% of the existing cache proves complete historical source coverage.
- Public JSON workout pagination is a distinct source stream, not another HTML navigation link. Complete the current dated stream before broad traversal; verify advancing pagination and report incomplete streams after a bounded segment.
- Name-only daily workout records are not full horse identities. Confirm against independently captured official horse-page workout records or dated official registration evidence; reject fuzzy/name-only joins.
- Record available_at conservatively from first public capture where historical publication is unknown. This can support later races, but must not be backdated into earlier races.
- Closure requires an explicit list of unresolved paper features, source intervals and archive experiments, plus measured recovery evidence. No arbitrary claim that all proprietary Benter factors are reproducible.

## Acceptance Criteria
- [ ] Crawl only public HKJC racing information and permitted official archives, not the entire corporate website, media files, accounts or betting endpoints.
- [ ] Reuse existing cached official pages; every accepted fact links to captured URL, response hash, parser version and fetch time.
- [ ] Raw snapshots and durable queues survive graceful stop/resume; failed/blocked/unavailable/empty/unparsed pages have distinct statuses.
- [ ] Resolve active and retired horse identities through stable official IDs, alias records, form/race links, same-sire controls and archive routes, not name similarity alone.
- [ ] Results, trackwork, barrier trials, veterinary events, movements, sectionals and incidents each have measured availability and temporal coverage.
- [ ] No third-party data enters official-only normalized tables, provenance or feature matrices by fallback or derived aggregate.
- [ ] Event dates, availability evidence and current profile snapshots are distinct. Historical features never use future outcomes/current mutable attributes without historical support.
- [ ] New snapshot has a source manifest, coverage matrix, row/race/horse counts, conflicts, exclusions and explicit gaps. No constant-zero family is presented as genuine observed absence.
- [ ] Complete pilot and bulk coverage across declared seasons and both venues, using documented recoverable gaps rather than assuming every calendar day has racing.
- [ ] Existing prediction/research contracts remain unchanged; running V5 and other users/services untouched.

## Research
- Local primary paper: research/1994-benter.pdf, especially pp.184-185 for feature categories and pp.187-189 for separate-sample combination and incremental information.
- [Scrapy JOBDIR](https://docs.scrapy.org/en/latest/topics/jobs.html): persistent scheduler/dedup/state; graceful stop required, same Scrapy version on resume, trusted exclusive job directory.
- [Scrapy AutoThrottle](https://docs.scrapy.org/en/latest/topics/autothrottle.html): reuse adaptive latency-based pacing plus hard domain concurrency/delay limits.
- [HKJC trackwork](https://racing.hkjc.com/en-us/local/information/trackworksearch), [barrier archives](https://racing.hkjc.com/en-us/local/information/archive/btresult?Date=09%2F09%2F2025), [retired horse example](https://racing.hkjc.com/en-us/local/information/otherhorse?horseid=HK_2022_H033).
- Verified public result-date JSON feed: `/racing/information/json/DateList/LocalResults.aspx?lang=en-us`. On 2026-10-02 it exposed 164 date candidates, all with null venues. These are leads, not an exhaustive meeting denominator; do not guess venues or equate null with no racing.
- Cached official forms expose pre-cache history for 922 horses, earliest 2001-04-14. Extract actual result anchors from their raw profile pages into a provenance-backed archive frontier; fetched result content must still verify the requested date and race number. This is recovery evidence, not proof that every earlier meeting is already available.
- [HKJC disclaimer](https://www.hkjc.com/en-us/disclaimer): official does not mean error-free. Preserve corrections/conflicts and avoid overstating accuracy/completeness.
- Discovery observed: direct horse lookup redirects to otherhorse for retired H033; that page retains form and race links and a Same Sire selector. Explore actual DOM controls instead of assuming parents are navigable hyperlinks.
- racing.hkjc.com/robots.txt returned HTTP404 in an initial direct probe. Recheck per run and obey actual restrictions; this is not a legal permission determination. Stop on login gates, CAPTCHAs or explicit denial; do not rotate IPs or bypass them.

## Root-Cause Baseline
- Trigger scope: missing model input coverage and uncertain archive/identity acquisition.
- Proven C1: build_v4_dataset.py invokes load_full_rich_history without trackwork/barrier/veterinary inputs. The inherited V5 snapshot has constant-zero availability and entirely absent trial speed/workout recency.
- Proven C2: build_historical.py imports third-party runners and enrichments; load_full_rich_history includes legacy third-party runs. Merely tagging its output official would misrepresent provenance.
- Proven C3: backfill_horse_profiles.py calls fetch_profile, not fetch_all, and exports only profiles and form; horse_pages.py has enrichment parsers but no full historical acquisition/export path.
- Proven C4: identity discovery uses one horse-link regex on cached race HTML; it is not relationship/control-aware traversal. BulkArchiveCollector equates parser failure with no meeting on race1, hiding possible parser/archive errors.
- Proven C5: daily trackwork HTML is a JavaScript shell; actual rows are in the public paginated TrackworkOneDayRecords JSON feed. Empty placeholder tables cannot certify no workouts.
- Proven C6: numeric-only result parsing omitted starters with PU/UR/FE/DNF/DISQ and misclassified withdrawal suffixes. Whole-field validation and explicit statuses recovered these races without inventing finishing positions.
- Proven C7 (validation gap, not demonstrated corruption): initial replay verified date but trusted requested race number/venue. A mistaken R1 glob selected R10 during a spot-check; the subsequent full 14,091-file filename/header audit found 14,091 matches and zero mismatches. Require exact displayed meeting header and race number anyway, and fully replay raw before promoting a snapshot built with the stronger validation. Keep observed data errors separate from hypothetical risks.
- Proven C8: the 2001 archive has 11-column result tables, without Running Position, whereas later tables have 12. Normalize the known layout using a structured DOM adapter, preserving missing positions rather than dropping all runners.
- Proven C9: 28 remaining cached races contain TNP runners. HKJC's 2009-05-17 race-7 stewards report explicitly deems SUPER BABY a runner despite refusing to leave the stalls. Preserve TNP as an official nonwinner without fabricated finish/time; do not equate it with a refunded withdrawal. See [official report](https://racing.hkjc.com/en-us/local/information/racereportfull?Date=5%2F17%2F2009) and [official racecard abbreviation legend](https://racing.hkjc.com/racing/content/PDF/RaceCard/20260222_starter_all.pdf).
- Proven C10: cached official profile titles use old B/C cycle-prefixed brands (CA018) and `(Deregistered)`, not only `(Retired)`. The legacy parser misread these titles and the strict brand check rejected dated form ratings. Normalize only these source-observed title/alias forms for the already linked full ID; preserve displayed alias and registration status at capture. Never merge different full IDs by their shared trailing brand.
- Proven C11: stopping the continuous service interrupted its subprocess wait; the parent exited while the crawler only logged Closing spider and no closed.json appeared. Likely H3: systemd then terminated the remaining child before its queue/readback flush completed. Use an isolated child session, main-only initial signal (KillMode=mixed), explicit SIGINT forwarding and parent wait. Test real stop/resume and require closed.json plus scheduler readback; a shutdown-start log is not proof of graceful completion.
- Likely H1: inactive horse pages need otherhorse, archive, season and related-entity navigation. User-described routes are hypotheses to verify and preserve as fixtures.
- Possible H2: older workout/veterinary pages have retention limits or changed layouts, and backfilled current pages cannot prove historical availability.
- Missing evidence: oldest accessible official seasons per source, complete live DOM route inventory, archival event publication timing and authoritative retired identity coverage.
- Minimum evidence: raw page hashes/headers, URL path/query, requested versus displayed dates, selectors/form options, route outcomes, identity match method, collector/parser revision and per-family date distribution.
- Mutation boundary: new corpus paths/branch only. No production rebuild, legacy-file overwrite, optimizer switch, unrelated service restart or inferred correction of historical observations.
- Remediation mapping: C1/C2 -> official snapshot builder; C3 -> family collectors/export; C4/H1 -> graph/control discovery and explicit parse statuses; H2 -> archive experiments and honest temporal exclusions.
- Generic hardening: retry/disk guards are resilience work, not evidence that missing archives were recovered.
- Not-done: a successful HTTP200 with navigation-only content, all-zero missing families, guessed identities, contaminated derived features, or a daemon running without dataset readback.

## SOTA, Standards, And Best Practices
- Reuse Scrapy, Parsel/lxml structured selectors, existing result/profile parsers where verified, pandas validated joins and existing rich feature functions.
- No hand-written crawler queue/retry/thread runtime when Scrapy provides it. No arbitrary internet crawl, browser stealth, login scraping or API credential use.
- Browser discovery is only for public JS/form controls or verifying rendered content; use exposed official links/forms, not bypassed access controls.
- Begin domain concurrency2, minimum delay2 seconds, AutoThrottle target0.5/max60 seconds; all page families share the domain slot. Honor Retry-After and pause on repeated429/403.
- Unit tests use captured fixtures. Bulk runs use cached reparse whenever possible and versioned normalized outputs; never regenerate previous snapshots in place.

## Scope and Data Contracts
Planned Touch Files:
- `scrapper/official_corpus.py`
- `scripts/collect_official_corpus.py`
- `scripts/discover_official_seeds.py`
- `scripts/build_official_dataset.py`
- `scripts/audit_official_coverage.py`
- `scripts/verify_official_snapshot.py`
- `tests/test_official_corpus.py`
- `tests/test_official_dataset.py`
- `tests/test_official_coverage.py`
- `tests/test_official_collector.py`
- `config/hkjc_official_seeds.json`
- `deploy/systemd/ima-hkjc-acquisition.service`
- `deploy/systemd/ima-hkjc-coverage.service`
- `deploy/systemd/ima-hkjc-coverage.timer`
- `ima/rich_features.py`
- `pyproject.toml`
- `docs/HKJC_OFFICIAL_CORPUS_PLAN.md`
- `docs/HKJC_OFFICIAL_CORPUS_VALIDATION.md`

- Initial horizon: official available seasons covering at least 2005-2026; probe earlier boundary separately and extend only when official records are accessible. Record latest completed meeting, not imagined future results.
- Corpus layers: immutable raw response blobs -> page inventory/link graph -> normalized official entities/events -> validated source snapshot -> feature-ready dataset.
- Start with racing.hkjc.com, HTTPS only and explicit local-racing route allowlist. Add another HKJC host/path only after audited official evidence and documented scope expansion. Reject offsite redirects/links, credentials in URLs and unbounded query variants.
- Every response: requested_url, final_url, fetched_at_utc, HTTPstatus, content type, body hash, byte length, referring URL, family, capture path and parse status. Keep access/error evidence separately from facts.
- Every fact: source_url/body_hash/parser_version, official_source, event_date, available_at/evidence method, horse_page_id and family-specific stable event key. Fetch time is not historical publication time.
- Identity: preserve full HKJC identifier; brand codes and names are aliases scoped by dates/context. Never collapse horses from trailing four-character brand alone. Unverified links/names are discovery leads, not accepted identity facts.
- Sire/dam/related links are discovery edges, not substitutes for a horse's own history. Same-sire dropdown values may reveal IDs not present in hrefs. Follow resulting horse -> form -> old race -> other runners transitively with deduplication.
- Page statuses: fetched_parsed, verified_empty, fetched_unparsed, missing_archive, denied, retryable_error, unsupported. A zero count means verified empty only with positive coverage evidence for that period.
- Existing non-HKJC files are retained but quarantined from this corpus. They may generate human-review leads, never contribute model values or be relabeled official.
- Resolve source conflicts by preserving both captures and audited rule, not averaging. Mutable profile age/rating/trainer/gear at capture cannot be applied indiscriminately to old races.
- Preserve official profile snapshots separately. Country of origin is an attributed immutable predictor, with conflicting reports quarantined. Age, sex/gelding status, current trainer/rating/gear and cumulative profile model_features remain capture-time snapshots, not retroactive predictors.
- Output tables: official-races, official-runners, identities/aliases, profile-snapshots, horse-form, trackwork, barrier-trials, veterinary-events, movements, sectionals, race-incidents and coverage. Some sources may remain unsupported; never create fabricated records.

## Exploration Matrix: Try These Routes and Record Outcomes
| Experiment | Samples / method | Required evidence | If it fails |
|---|---|---|---|
| Active vs retired profile | active, retired H033, older retired, renamed horses | final URL, ID, form count, title | otherhorse route, linked race, explicit unresolved |
| Related controls | same-sire option values, parent/offspring links where actually present | DOM/control action and recovered IDs | no assumed link from plain text |
| Horse -> old race -> runners | form race links in multiple seasons/venues | requested/displayed race match, complete runner IDs | archive/date route and parser inspection |
| Legacy vs modern routes | LocalResults.aspx and actual modern redirect | final route/query and semantic parity | record migration map, not URL guessing at scale |
| Season/pagination | profile season controls, archives, dropdowns | oldest/latest event, options and row counts | bounded official form actions or browser pilot |
| Barrier trial archives | recent, one-year old, 5/10/20-year samples | date, batch, venue/distance, runners, pass/fail/comments | classify archive boundary; do not infer race-level speed directly |
| Workouts | horse/date search and racecard work pages | record dates/types, retrieval window and ID match | temporal coverage gap, not zero workouts |
| Veterinary / movements | active and retired pages, race-dated summaries | dated event, cleared date, source retention | unavailable historical interval |
| Sectionals / incidents | result-linked sectionals and reports | race/date match, runner position/comment linkage | raw quarantine and parser fixture |
| Race census | official fixture/calendar/results indexes | meetings/races expected vs acquired | compare horse-form recovered dates and audit anomalies |
- Minimum pilot: 20 stratified horses and 12 meetings spanning eras/venues, plus each available family, including a route failure and genuine empty result. It is a coverage pilot, not an unbiased statistical sample.
- Escalation rule: after a parser failure inspect raw page and public browser render, test corrected selector on multiple fixtures, then reparse cache before further network calls.

## Benter Feature Coverage Contract
- Current condition: recent form/rest/age and actual dated workouts/trials. Missing capture is not rest/no exercise.
- Past performance: position, beaten lengths and times normalized using declared distance/course/going/time context. Keep cancellations/withdrawals/dead heats as explicit labels/status, not winner0 inventions.
- Past adjustments: strength of competition, historical weight/jockey/post advantage and trouble comments. Begin documented proxies, then controlled residual models on earlier data. Exact unpublished definitions remain unclaimed.
- Today's context: race-dated weights/draw/jockey/field and conditions, never today's profile substituted into an old card.
- Preferences: distance/surface/going/track conditioned history and an explicitly tested residual distance-preference implementation; retain uncertainty/support for sparse horses.
- Coverage report for each family: relevant universe/denominator, source capture %, parse %, identity join %, historical availability %, nonmissing/unique/variance, usable races and earliest/latest dates by season/venue/cohort.
- Publication unknown: facts may remain in raw/normalized research inventory, but exclude them from leakage-safe historical training unless a defensible source-specific publication rule exists.

## Dependency and Tooling Preflight
- Install or repair commands: `uv pip install --python .venv/bin/python 'scrapy>=2.13,<3'`; if public browser exploration requires it, `uv pip install --python .venv/bin/python playwright` then `.venv/bin/python -m playwright install chromium`. Remote installs use transferred wheels in a separate venv.
- Scrapy optional acquisition extra; install project-locally, record exact version and keep it pinned per JOBDIR. Existing parsers and unittest are retained.
- Required commands: `.venv/bin/python -m unittest ...`, collector --help, dataset builder --help, fixture tests and Playwright route inspection when needed.
- Browser setup is required work if a public control needs rendering; install local Playwright/Chromium if absent rather than calling JS discovery complete without it.
- Remote work: dedicated imaopt acquisition release/venv/output; transfer wheels/artifacts through Tailscale or approved tunnel, no modifications to the shared training environment. Own-user limited-CPU/RAM background service only.
- Blockers: denied public access, unreachable host, disk shortage, unavailable archives/publication evidence, incompatible dependencies or conflicting identities. Preserve evidence and report precise incompleteness.

## Deterministic Real-User Test
- Operator commands: collect --seed-file official-seeds.json --output NEW_CORPUS --limit 30; inspect report; normalize cached pages; build NEW_SNAPSHOT; resume same JOBDIR after graceful stop.
- Add exact CLI contracts and tests before use. Fixtures: redirected retired horse, same-sire control, race link, wrong-date page, genuine empty/blocked response, trial batch, workout table and veterinary table.
- Assertions: no external requests or values; ID recovered through link/control; full provenance; explicit missingness; no duplicate events after resume; source hash replay deterministic; old V5 checksum unchanged.

## Fulfillment and Readback Proof
- Expected result: actual captured and normalized official event records and immutable dataset manifest, not merely scraper code or queued URLs.
- Inspect corpus inventory/summary, raw hashes, normalized sample events, source-only checks, coverage by period and feature matrix rows. Recompute selected samples independently from raw fixtures.
- Requirements: demonstrate nonconstant measured workout/trial/veterinary inputs where available, report all unavailable intervals, and account for row/race attrition versus expected official universe.
- Not-done: successful process with empty parse outputs, all website links captured but no facts, hidden third-party contamination or guessed historical availability.

## Armageddon Mode
- Cases: offsite and login redirects, blocked response, Retry-After, malformed date, HTML layout shift, enormous query fanout, duplicate/cyclic related links, reused names/brands, renamed horse, scratched/dead heat runner, mutable age, future clearance and missing archive.
- Kill/restart during capture/export, corrupt blob, disk low and network loss; preserve partial state and replay cache. Clean pause/resume is supported; abrupt-kill recovery must be explicitly tested, not assumed from JOBDIR.
- Must-fix: nonofficial facts admitted, wrong horse/race join, future leakage, denied-access bypass, overwriting pinned data, request flood or mislabeled zero availability.

## Generality Guardrail
- Existing mechanism: Scrapy persistent scheduling and existing official result/profile parsers; recurrence is high across all acquisition families. General mechanism decision: one reusable official corpus owner rather than unrelated per-source crawlers.
- One official acquisition owner, family adapters, provenance store and identity graph; route both active and retired sources through them. Reuse existing feature functions without mixing legacy loaders.
- Source acquisition remains outside training. Future orchestrator requests may invoke a bounded acquisition job, but no training trial browses live webpages.

## Ordered State and Dashboard
- Plan-scoped ledgers: .mega/hkjc-official-corpus/state.jsonl and evidence.jsonl. Record source probes, parser fixtures, tests, samples, remote launch and corpus readback.
- Dashboard: .mega/dashboards/hkjc-official-corpus.html, generated from this canonical plan and its scoped ledgers. Do not import previous plan completion states.
- Status labels: planned/active/blocked/done; a bulk run in progress does not mark dataset coverage complete.

## Regression Guardrails
- Branch strategy: dedicated isolation branch feat/hkjc-official-acquisition; preserve unrelated dirty V3/graph/plan files.
- Damage radius: large, because wrong identity/provenance can silently corrupt research; additive corpus code does not change live training.
- Protected: current V5 pinned files/campaign, legacy dataset builders and third-party archives, public scraping boundaries, other server users/Cortex/Solar services and existing prediction contracts.
- Edit surface: new scrapper/official_corpus.py, scripts/collect_official_corpus.py, scripts/build_official_dataset.py, scripts/audit_official_coverage.py, tests/test_official_corpus.py, tests/test_official_dataset.py, tests/test_official_coverage.py, config/hkjc_official_seeds.json, fixtures and this plan; pyproject.toml optional extra. Changes to existing parser/feature adapters require declared expansion and regression tests.
- Proof plan: deterministic source/identity tests -> live bounded pilot -> cache replay -> official-only snapshot -> resumable bulk -> iterative coverage readback.

## Phase 1: Discovery and Access
### Subphase 1.1: Census, routing and exploratory evidence
- Commit: docs(data): document HKJC source route and coverage census.
- Planned Touch Files: this plan, docs/HKJC_OFFICIAL_CORPUS_VALIDATION.md, scripts/discover_official_seeds.py, official route fixtures.
- Tests: fetch public robots/route samples, inspect DOM/controls, compare existing raw corpus counts and sources; record commands and source hashes.
- Success Criteria: verified allowlist, initial active/retired/sample horizon and explicit unresolved routes.
- Checklist:
  - [ ] Audit source policies, endpoint patterns and actual old/new routes.
  - [ ] Execute exploration matrix and save representative fixtures without unrelated navigation data.
  - [ ] Inventory official cached results and horse IDs; quarantine nonofficial source artifacts.

## Phase 2: Acquisition Foundation
### Subphase 2.1: Resumable official-only collection
- Commit: feat(data): add scoped HKJC corpus collector with provenance.
- Planned Touch Files: scrapper/official_corpus.py, scripts/collect_official_corpus.py, pyproject.toml, tests/test_official_corpus.py.
- Tests: `.venv/bin/python -m unittest tests.test_official_corpus`; deterministic CLI help/pilot/resume tests.
- Success Criteria: bounded polite crawl, atomic raw captures, explicit statuses and offline replay, no external fetching.
- Checklist:
  - [ ] Scrapy JOBDIR, dedup, AutoThrottle, hard domain limits, Retry-After and graceful shutdown.
  - [ ] Allowlisted seeds/links/forms, same-origin redirects and query canonicalization preserving actual event identity.
  - [ ] Hash raw responses, record link/control provenance and report empty/unparsed/blocked separately.
  - [ ] Import cached official HTML with capture timestamps retained; current reparse time is not original fetch time.

## Phase 3: Entity and Archive Recovery
### Subphase 3.1: Recover identifiers and expand historical frontier
- Commit: feat(data): discover retired horses and archive event links.
- Planned Touch Files: scrapper/official_corpus.py, scripts/discover_official_seeds.py, tests/test_official_corpus.py, source fixtures and validation report.
- Tests: active/retired/renamed/same-sire/race-neighbor fixtures; browser public-control replay where required.
- Success Criteria: recovered inactive ID links to its own historical form and full old race; cycles deduplicate and ambiguous aliases quarantine.
- Checklist:
  - [ ] Explore same-sire options and genuine pedigree relationships, then race-event neighbor discovery.
  - [ ] Preserve full IDs and date-scoped aliases; never merge on horse name or stripped brand alone.
  - [ ] Extract official date/season/venue indexes; bounded boundary probes instead of brute-forcing all dates/races.
  - [ ] Resolve failures through cache/browser evidence, not external scraped copies or access bypass.

## Phase 4: Normalization and Coverage
### Subphase 4.1: Parse event families and validate completeness
- Commit: feat(data): normalize official event families and coverage states.
- Planned Touch Files: scrapper/official_corpus.py, scripts/build_official_dataset.py, scripts/audit_official_coverage.py, tests/test_official_corpus.py, tests/test_official_dataset.py, tests/test_official_coverage.py and fixtures.
- Tests: structured table semantics, locale/date/unit parsing, requested/displayed date equality, corrected report versions, stable hashes and record keys.
- Success Criteria: independently verified sample records per obtainable family, identity joins and measured gaps.
- Checklist:
  - [ ] Normalize race/profile/form, trial/workout/veterinary/movement and obtainable sectionals/incidents; raw-unparsed stays quarantined.
  - [ ] Event timestamps and official publication evidence, cancelled/withdrawn states, nonstandard times and historical identity handling.
  - [ ] Coverage denominators are declared official census, not successful downloads only; report newly found meetings and unresolved census ambiguity.
  - [ ] Reparse cache after every parser correction and compare multi-era fixtures; do not refetch everything.

## Phase 5: Dataset Validation
### Subphase 5.1: Build immutable official-only feature snapshot
- Commit: feat(data): build point-in-time HKJC-only research snapshots.
- Planned Touch Files: scripts/build_official_dataset.py, scripts/verify_official_snapshot.py, tests/test_official_dataset.py, docs/HKJC_OFFICIAL_CORPUS_VALIDATION.md; ima/rich_features.py only if verified adapter expansion is necessary.
- Tests: `.venv/bin/python -m unittest tests.test_official_dataset tests.test_rich_features tests.test_historical`; schema/provenance/future-label perturbation and race consistency tests.
- Success Criteria: actual versioned dataset with source manifest, leakage-safe observed coverage and no legacy fallback; model-compatible schema without silent zeros.
- Checklist:
  - [ ] Load only official normalized runners, bypass legacy mixed loader; pass all actually supported family tables explicitly.
  - [ ] Enforce availability cutoffs and historical mutable-field rules; preserve no_event vs unknown/not_collected.
  - [ ] Emit Benter coverage matrix and denominators, missing identities/conflicts/exclusions, per-season counts and nonconstant source checks.
  - [ ] Validate full race rosters, one-winner policy and time/unit consistency; feature computation hashes fixed inputs.
  - [ ] Read back raw-to-row samples; do not promote or repoint V5 to this snapshot.

## Phase 6: Operations and Iteration
### Subphase 6.1: Isolated bulk run and coverage-driven recovery
- Commit: docs(ops): document official acquisition bulk run and recovery receipts.
- Planned Touch Files: scripts/collect_official_corpus.py, docs/HKJC_OFFICIAL_CORPUS_VALIDATION.md, deploy/systemd/ima-hkjc-acquisition.service, deploy/systemd/ima-hkjc-coverage.service, deploy/systemd/ima-hkjc-coverage.timer, own imaopt acquisition launch configuration under separate path; no training unit changes.
- Tests: full relevant unittest suite, graceful resume, network/denial/disk tests, own-user live bounded canary and protected dataset hash readback.
- Success Criteria: bulk run observed retrieving/parsing records, source coverage improves, restart works and resource/rate limits remain honored.
- Checklist:
  - [ ] Transfer dependencies through Tailscale into separate venv; verify egress route before remote fetching.
  - [ ] Begin small pilot, inspect semantic records, then durable low-resource bulk job with daily/status reports.
  - [ ] Prioritize coverage holes by source/season/retirement, investigate each failure class and iterate parsers/routes.
  - [ ] Stop only for authentic access/archive/publication/identity blockers; record unsolved gaps. Large frontier completion may require hours/days and is not a same-turn promise.
  - [ ] Repeat immutable snapshot build after measured increments; retain manifests and previous snapshots. Completion needs declared horizon accounted for, not all possible HKJC pages.
  - [ ] Final summary distinguishes code/tests, pilot records, bulk progress, dataset quality and remaining Benter gaps. No claims of all proprietary features or future profitability.
