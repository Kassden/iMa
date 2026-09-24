# Research Provider Contract

Checked against OpenRouter documentation on 2026-09-24.

## Batch API

- Submit batches with `POST https://openrouter.ai/api/v1/batches`.
- Poll one batch with `GET https://openrouter.ai/api/v1/batches/:id`.
- Results are returned inline on the completed batch object; there is no separate
  results-download endpoint.
- Terminal statuses are `completed`, `failed`, `expired`, and `cancelled`.
- Batch requests use a top-level `endpoint`, `model`, and `requests` array.

## Service Tier

- `service_tier: "flex"` requests flex endpoints when any are available.
- Flex does not fall back to default-tier endpoints when flex capacity exists
  but is unavailable; a capacity error can surface.
- If a model has no flex endpoint, OpenRouter may route normally at standard
  rates.
- The response `service_tier` field is the evidence of the tier actually used.
