# Historical Coverage

Generated from `data/processed/historical/coverage.json` on 2026-07-25.

## Canonical Runner Archive

- Date range: 2005-01-01 through 2025-12-27.
- 202,174 runner rows across 16,460 races.
- 173,456 rows come from reparsed official HKJC result pages.
- 26,633 rows come from Mexwell, 1,857 from Jeffrey Muller, and 228 from the Swords 2008-09 workbook after deterministic source precedence.
- 35 dead-heat races are preserved.
- 176 races have no winner row: 87 in 2005, 54 in 2006, 31 in 2007, and 4 in 2008. These are retained for provenance but must be excluded from supervised outcome training and result-based evaluation.

| Year | Races | Runners | No winner | Dead heat |
|---:|---:|---:|---:|---:|
| 2005 | 706 | 7,513 | 87 | 0 |
| 2006 | 704 | 7,859 | 54 | 1 |
| 2007 | 731 | 8,552 | 31 | 3 |
| 2008 | 715 | 8,925 | 4 | 3 |
| 2009 | 744 | 9,487 | 0 | 1 |
| 2010 | 777 | 9,753 | 0 | 2 |
| 2011 | 759 | 9,472 | 0 | 1 |
| 2012 | 769 | 9,604 | 0 | 1 |
| 2013 | 778 | 10,012 | 0 | 2 |
| 2014 | 757 | 9,652 | 0 | 2 |
| 2015 | 796 | 10,224 | 0 | 2 |
| 2016 | 789 | 10,102 | 0 | 1 |
| 2017 | 818 | 10,213 | 0 | 3 |
| 2018 | 800 | 9,832 | 0 | 1 |
| 2019 | 805 | 10,053 | 0 | 3 |
| 2020 | 840 | 10,507 | 0 | 2 |
| 2021 | 827 | 10,050 | 0 | 3 |
| 2022 | 822 | 9,807 | 0 | 1 |
| 2023 | 842 | 10,057 | 0 | 2 |
| 2024 | 841 | 10,144 | 0 | 0 |
| 2025 | 840 | 10,356 | 0 | 1 |

## Official Acquisition

- 1,506 official venue-dates are complete, containing 14,091 races and 173,456 reparsed runner rows.
- 4,832 calendar venue-dates were verified as having no meeting.
- 257 known meetings before the reliable official archive boundary are marked `archive_unavailable`, not `no_meeting`.
- Earliest recovered official meeting: 2008-04-02 at Happy Valley.
- Raw HTML is immutable under `data/historical/hkjc-2005-2025/raw/`; normalized JSON can be reproducibly rebuilt with `scripts.reparse_official_history`.
- The official reparse retains race class, distance, prize, going, course, lengths behind, running positions, jockey name, and trainer name for all 173,456 official runner rows. These fields were present in the immutable HKJC pages but were discarded by the earlier parser.

## Supplemental Tables

| Table | Rows | Date range | Semantics |
|---|---:|---|---|
| Timestamped WIN/PLACE odds | 6,402,332 | 2016-09-28 to 2018-06-27 | 259,652 source-naive snapshots across 1,510 races; timezone unverified |
| Dividends | 26,397 | 2005-01-01 to 2018-07-04 | Post-race payouts only |
| Sectionals | 365,235 | 2008-06-05 to 2018-06-27 | Runner section positions and times |
| Horse snapshots | 12,945 | 2016-09-27 to 2018-07-01 | Profile snapshots, not event history |
| Trackwork | 648,016 | 2015-06-02 to 2017-07-31 | Dated workout events |
| Barrier trials | 10,219 | 2015-06-02 to 2025-04-15 | Hrosebaby plus DatasetLabs sample |
| Runner comments | 21,164 | 2015-06-03 to 2017-07-16 | Post-race comments |
| Incident reports | 3,100 | 2008-09-15 to 2017-07-16 | Steward and veterinary narratives |
| Point-in-time race cards | 1,278 | 2025-03-02 to 2025-04-09 | Exact capture time unavailable |
| Legacy final WIN/PLACE odds | 9,010 | 2005-01-01 to 2005-08-28 | Final/result odds, never pre-race snapshots |

## Training Gates

- Reject races without exactly one winner unless the evaluation explicitly supports dead heats.
- Never use dividends, result-page odds, final pool totals, incident outcomes, or post-race comments as pre-race features.
- Keep `odds-snapshots.csv.gz` timestamps source-naive until the timezone is independently verified.
- Use source and field missingness in `coverage.json`; official results provide race class, distance, course, going, prize, jockey/trainer names, lengths behind, and running positions. Ratings, horse demographics, and stable jockey/trainer IDs still require point-in-time race-card or horse-history joins.
- Review `source-registry.json` before redistribution or commercial use. Several archives have unknown, share-alike, or non-commercial terms.