# Architecture Scorecard

- Risk tier: moderate
- Risk points: 6
- systems: 5
- file_edges: 154
- cycles: 1
- max_hub_degree: 18
- api_routes: 0
- workers: 48
- schemas: 71
- integrations: 5

## Recommendations
- Break dependency cycles before extracting services.
- Review top hubs for accidental shared-kernel or god-module behavior.
