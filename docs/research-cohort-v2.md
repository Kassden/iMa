# Research Cohort V2

The first agentic cohort now has executable model components beyond the legacy
hyperparameter catalogue:

- `RaceSoftmaxOffsetModel`: domain-specific linear offset over the public market
  probability. `w = 0` exactly recovers the market distribution.
- `ResearchRegressor`: secondary-target regressor for adjusted speed/finish-time
  probes.
- Target-specific diagnostics for win probability, ranking, placing, adjusted
  speed, and market-forecast probes.

These components remain research-only. Secondary-target wins do not promote a
race-day win-probability model unless a later predeclared multi-objective rule
allows that conversion.
