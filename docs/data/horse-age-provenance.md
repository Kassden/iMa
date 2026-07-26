# Horse Age Provenance

## Model timelines

The full model joins two timelines, but their age semantics are different.

| Timeline | Age observation | Identity | Calendar semantics |
|---|---|---|---|
| Legacy gdaley archive | Directly recorded on each runner row | Anonymized integer `horse_id` | Source dates are deliberately obscured |
| Canonical 2005-2025 archive | Projected from timestamped profile references | HKJC horse code and profile-cycle ID | Actual race date |

The legacy age is not inferred and has zero propagation offset. It remains useful as a race-row
feature, but it must not be described as an official-ID-linked HKJC profile observation.

## Why the legacy IDs do not match HKJC

The gdaley Kaggle metadata states that horse and jockey names were removed and that the dates in
`races.csv` were obscured and "are not the real ones". The file contains integer horse keys such as
`1253`; official HKJC result pages instead expose horse codes such as `D263` and profile IDs such as
`HK_2003_D263`.

This was verified against the official result page for Sha Tin race 1 on 2005-01-01. The page links
LEGIONNAIRE as `HK_2003_D263`. The legacy row labeled with the same date/race contains a different
field, confirming that date, race number, draw, or runner number cannot be used as a linkage key.

Source metadata:

- <https://www.kaggle.com/datasets/gdaley/hkracing>
- <https://www.kaggle.com/api/v1/datasets/view/gdaley/hkracing>
- <https://racing.hkjc.com/racing/information/English/Racing/LocalResults.aspx?RaceDate=2005/01/01&Racecourse=ST&RaceNo=1>

## HKJC profile evidence

Official historical result pages expose the true HKJC code and profile-cycle ID. Current pages for
retired horses retain country, colour/sex, import type, owner, rating, sire, dam, and full form, but
do not retain the horse's former age. Active profiles expose age at the current capture timestamp.

Wayback CDX checks for the sample retired horse returned captures only from 2024 onward, after its
age had already disappeared. Archived captures can therefore be admitted only when the captured
HTML contains an explicit age and the Wayback timestamp is stored with it.

## Accepted age reference contract

Every projected canonical age records:

- `horse_age_reference_source`
- `horse_age_reference_year`
- `horse_age_reference_value`
- `horse_age_year_offset`
- `horse_age_identity_method`

Accepted identity methods are exact profile-page ID or horse-code-to-profile-cycle matching. No
legacy anonymized ID is converted into an HKJC code without an explicit upstream linkage table.
