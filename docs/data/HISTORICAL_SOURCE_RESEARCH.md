# Historical Data Research

## Verified Official Surfaces

The official HKJC local-results archive accepts date, venue, and race number and returned a complete 2025 race sample without login from the current Hong Kong mobile egress. The result table includes finish order, runner identity, jockey, trainer, carried and declared weights, draw, margins, running positions, finish time, and final displayed WIN odds.

The official race-card route also returned HTTP 200, but its useful runner content requires browser rendering. Horse profile pages remain directly readable and expose form, trackwork, veterinary, and movement history.

Known meetings requested through `--meeting-index` are checkpointed as `archive_unavailable` when HKJC returns no parseable result. This is deliberately distinct from `no_meeting`, so inaccessible older records are never reported as races that did not exist.

## Important Limitation

Result-page odds have no observation timestamp. They must be labeled final/result odds and cannot be used as if they were odds available at an earlier wagering decision time. Genuine market-movement and closing-line analysis requires browser snapshots collected before post time or a licensed timestamped feed.

## Acquisition Order

1. Pilot one recent season of official results with immutable raw HTML and normalized JSON.
2. Browser-collect corresponding pre-race cards and reconcile runner identifiers.
3. Audit missing meetings, abandoned races, dead heats, scratches, dividends, and sectional availability.
4. Review HKJC terms and redistribution constraints before bulk collection.
5. Seek timestamped historical odds from licensed providers while accumulating first-party live snapshots going forward.

## Third-Party Archive Acquisition

The local immutable archive now includes eleven downloaded datasets. SHA-256 hashes, source URLs, licensing, timestamp semantics, and integration status are recorded in `source-registry.json`.

The most useful additions are:

- Mexwell: 2005-2012 canonical runner coverage, 2016-2018 timestamped WIN/PLACE snapshots, exotic dividends, sectionals, and horse snapshots.
- Swords 2008-09: runner numbers and names that repair Mexwell missingness, plus 733 steward/veterinary incident narratives.
- gdaley CC0: final WIN/PLACE odds and dividends for January-August 2005.
- hrosebaby: 648,016 trackwork events, 9,219 barrier trials, and 21,164 runner comments from 2015-2017.
- Lantanacamara: 2,367 race incident reports from 2014-2017.
- DatasetLabs sample: 11 point-in-time race-card dates and 1,000 barrier-trial rows in March-April 2025. The files are dated 2025 despite the dataset's “2024” title.
- heodi510 GitHub archive: 74,165 runner rows from 2013-09 through 2021-01; retained as an overlap/check source because no repository license is present.
- Leowongco GitHub archive: compact result parquet files labeled 2024-2026; retained in quarantine pending schema and license review.

The Bogdandoicin and p768lwy3 archives are retained but quarantined from the canonical merge because they overlap better-covered periods and need identifier and missing-value reconciliation first.

## Archive Semantics

Normalized outputs intentionally separate:

- `runners.csv.gz`: race/result facts used to build point-in-time features.
- `odds-snapshots.csv.gz`: timestamped WIN/PLACE observations; source timezone remains unverified.
- `legacy-final-odds.csv.gz`: untimestamped result-page WIN/PLACE odds, never valid as earlier pre-race snapshots.
- `dividends.csv.gz`: post-race payouts, excluded from pre-race outcome features.
- `sectionals.csv.gz`, `trackwork.csv.gz`, `barrier-trials.csv.gz`, `runner-comments.csv.gz`, `incidents.csv.gz`, and `racecards.csv.gz`: supplemental horse history and point-in-time declarations with explicit source fields.

## Internet Archive Check

The Wayback CDX API was tested for the modern HKJC `LocalResults.aspx` route with a 2005 timestamp filter and returned no captures. Wayback remains useful for targeted recovery when an exact legacy URL is known, but it is not a dependable bulk 2005-2025 source and cannot replace official or structured archives.

## Licensing Boundary

Only the gdaley archive is CC0. Several sources are share-alike, non-commercial, unknown, or inherit unresolved HKJC website terms. Local research use and model-development provenance are recorded, but redistribution and commercial deployment require a source-by-source legal review.
