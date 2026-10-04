"""Honest fixed-payout paper EV and joint-distribution ticket adapters."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import itertools
import math
from typing import Mapping, Sequence

import numpy as np

from ima.betting_contracts import POOL_SIZE, Quote, QuoteStatus, RuleInputs, Ticket, aware
from ima.betting_settlement import Finish, ticket_wins
from ima.pools import benter_order_probability


@dataclass(frozen=True)
class JointScenarios:
    finishes: tuple[Finish, ...]
    probabilities: tuple[float, ...]
    assumptions: str
    source_id: str
    complete: bool = True
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        p = np.asarray(self.probabilities, dtype=float)
        if (len(p) != len(self.finishes) or not len(p) or not np.isfinite(p).all()
                or (p < 0).any() or not np.isclose(p.sum(), 1, rtol=0, atol=1e-10)):
            raise ValueError("Joint scenarios must have finite nonnegative mass summing to one")
        if not self.assumptions or not self.source_id or not self.complete:
            raise ValueError("Complete outcome coverage and assumption/source evidence required")


def scenarios_from_orders(orders: Sequence[Sequence[str]], probabilities: Sequence[float], *,
                          assumptions: str, source_id: str) -> JointScenarios:
    """Boundary for rank_distributions output; probability vector is never renormalized."""
    return JointScenarios(tuple(Finish(tuple((str(r),) for r in order)) for order in orders),
                          tuple(float(p) for p in probabilities), assumptions, source_id)


def scenarios_from_rank_distribution(distribution, *, source_id: str) -> JointScenarios:
    """Consume RankDistribution.order_probabilities and metadata without resimulation.

    Gaussian sampled support is empirical, not exhaustive population support;
    preserve its convergence/tail evidence for sizing admission and reporting.
    """
    metadata = dict(distribution.metadata)
    metadata["runner_ids"] = tuple(distribution.runner_ids)
    assumption = metadata.get("assumption") or (
        f"{metadata.get('method', 'unspecified')} joint orders; "
        f"dependence={metadata.get('dependence', 'unspecified')}; "
        f"unseen_outcomes={metadata.get('unseen_outcomes', 'not reported')}")
    scenarios = scenarios_from_orders(tuple(distribution.order_probabilities),
        tuple(distribution.order_probabilities.values()), assumptions=assumption, source_id=source_id)
    return JointScenarios(scenarios.finishes, scenarios.probabilities, assumption, source_id,
                          metadata=metadata)


def scenarios_from_package_distribution(package, frame, *, runner_column="horse_no", seed=42,
                                        tolerance=0.01, min_draws=8192, max_draws=131072) -> JointScenarios:
    """One complete race -> packaged distribution -> existing converged rank engine.

    Row identity, model protocol and fit cutoffs are checked before binding
    distribution rows to betting runner numbers. Independent-Gaussian baseline
    only; correlated native joint outputs use scenarios_from_rank_distribution.
    """
    import pandas as pd
    from ima.rank_distributions import converged_rank_distribution
    required = {"race_id", "horse_id", "field_size", runner_column, "date"}
    if not required.issubset(frame.columns) or frame.empty or frame.race_id.nunique() != 1:
        raise ValueError("Package betting adapter requires one complete identified race")
    ids = tuple(frame[runner_column].astype(str))
    if (len(set(ids)) != len(ids) or frame.horse_id.duplicated().any()
            or not (pd.to_numeric(frame.field_size, errors="raise") == len(frame)).all()):
        raise ValueError("Incomplete race or duplicate betting runner identities")
    distribution = package.predict_distribution(frame)
    expected = tuple(zip(frame.race_id.astype(str), frame.horse_id.astype(str)))
    if tuple(distribution.row_keys) != expected:
        raise ValueError("Packaged distribution row keys differ from race/horse identity")
    dates = pd.to_datetime(frame.date, utc=True, errors="raise")
    if dates.isna().any():
        raise ValueError("Missing package query timestamp")
    for key in ("training_cutoff", "calibration_cutoff"):
        cutoff = distribution.metadata.get(key)
        if cutoff is not None and dates.min() <= pd.to_datetime(cutoff, utc=True, errors="raise"):
            raise ValueError("Package betting query must follow frozen distribution fit cutoffs")
    rank = converged_rank_distribution(distribution, ids, seed=seed, tolerance=tolerance,
                                      min_draws=min_draws, max_draws=max_draws)
    manifest = package.manifest()
    rank.metadata["package_id"] = manifest.package_id
    rank.metadata["protocol_id"] = manifest.protocol_id
    return scenarios_from_rank_distribution(rank,
        source_id=f"package:{manifest.package_id}:protocol:{manifest.protocol_id}")


def conditional_choice_scenarios(runners: Sequence[str], strengths: Sequence[float], *,
                                  depth: int, max_orders: int = 250000) -> JointScenarios:
    """Exact legacy pools kernel fallback. No clipping/fabrication of zero strengths.

    Win strengths imply conditional choices only by explicit assumption. A zero
    remaining denominator leaves later positions unidentified and fails closed.
    """
    ids = tuple(str(r) for r in runners)
    p = np.asarray(strengths, dtype=float)
    if (len(ids) != len(p) or len(set(ids)) != len(ids) or not ids
            or not np.isfinite(p).all() or (p < 0).any() or p.max() <= 0):
        raise ValueError("Invalid runner strengths")
    if not 1 <= depth <= len(ids) or math.perm(len(ids), depth) > max_orders:
        raise ValueError("Invalid depth or exact enumeration exceeds declared resource budget")
    if np.count_nonzero(p) < depth:
        raise ValueError("Zero remaining strengths cannot identify subsequent ranks")
    p = p / p.max()
    if np.count_nonzero(p) < depth:
        raise ValueError("Strength dynamic range cannot identify subsequent ranks in float64")
    orders, mass = [], []
    for indices in itertools.permutations(range(len(ids)), depth):
        if any(p[i] == 0 for i in indices):
            continue
        orders.append(tuple(ids[i] for i in indices))
        mass.append(benter_order_probability(indices, p))
    scenarios = scenarios_from_orders(orders, mass,
        assumptions="Plackett-Luce conditional choices from strengths; joint finishes are assumed, "
                    "not identified by win marginals; no dead heats or scratches modeled",
        source_id="ima.pools.benter_order_probability")
    return JointScenarios(scenarios.finishes, scenarios.probabilities, scenarios.assumptions,
                          scenarios.source_id, metadata={"runner_ids": ids})


def required_depth(tickets: Sequence[Ticket], rules: RuleInputs) -> int:
    return max((rules.paid_places if t.pool == "PLACE" else rules.qpl_places
                if t.pool == "QPL" else POOL_SIZE[t.pool]) for t in tickets)


def ticket_probabilities(tickets: Sequence[Ticket], scenarios: JointScenarios,
                         rules: RuleInputs) -> tuple[float, ...]:
    if not tickets:
        return ()
    if len({t.race_id for t in tickets}) != 1:
        raise ValueError("One race per joint outcome distribution")
    if any(not f.abandoned and any(not set(t.runners) & set(f.scratches)
                                  and sum(map(len, f.groups)) < required_depth([t], rules)
                                  for t in tickets)
           for f in scenarios.finishes):
        raise ValueError("Joint order prefixes too short for requested pools")
    field = set().union(*(set(r for g in f.groups for r in g) | set(f.scratches)
                          for f in scenarios.finishes))
    field.update(map(str, scenarios.metadata.get("runner_ids", ())))
    if any(not set(t.runners).issubset(field)
           and any(not f.abandoned and not set(t.runners) & set(f.scratches) for f in scenarios.finishes)
           for t in tickets):
        raise ValueError("Ticket runners absent from joint distribution")
    return tuple(min(1.0, float(sum(p for f, p in zip(scenarios.finishes, scenarios.probabilities)
                                  if ticket_wins(t, f, rules)))) for t in tickets)


@dataclass(frozen=True)
class EVReport:
    ticket: Ticket
    probability: float
    fair_decimal_return: float | None
    stake: float
    currency: str
    decimal_return: float | None
    expected_profit: float | None
    mode: str
    executable_evidence: bool
    reason: str
    rule_version: str
    quote_source: str | None
    probability_metadata: dict = field(default_factory=dict)


def evaluate_ticket(ticket: Ticket, probability: float, stake, currency: str, rules: RuleInputs,
                    *, at: datetime, race_start: datetime, quote: Quote | None = None,
                    quote_mode: str = "quoted", max_age: timedelta = timedelta(minutes=5),
                    probability_metadata: dict | None = None) -> EVReport:
    rules.validate_ticket(ticket, at, currency)
    aware(race_start)
    stake = rules.validate_stake(stake)
    if not math.isfinite(float(stake)):
        raise ValueError("Stake outside numerical range")
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Probability must be in [0, 1]")
    if quote_mode not in {"quoted", "fair_price", "scenario", "ex_post"}:
        raise ValueError("Unknown quote mode")
    fair = None if probability == 0 else 1 / probability
    d, ev, executable = None, None, False
    mode, reason = "fair_price", "No usable pre-race quote; fair price only"
    if quote is not None:
        quote.validate(ticket, rules, currency)
    if quote is not None and quote_mode != "fair_price":
        eligible = quote.executable_at(at, race_start, max_age)
        if quote_mode == "quoted" and eligible:
            mode, executable = "quoted", rules.verified
            reason = "Indicative payout; negligible own impact; no second takeout"
            if not rules.verified:
                reason += "; effective rule inputs unverified, sizing disabled"
            d = quote.decimal_return
        elif quote_mode == "scenario" and quote.status == QuoteStatus.SCENARIO:
            mode, d, reason = "scenario", quote.decimal_return, "Hypothetical payout; not executable evidence"
        elif quote_mode == "ex_post" and quote.status == QuoteStatus.FINAL:
            mode, d, reason = "ex_post", quote.decimal_return, "Hindsight price; never pre-race sizing evidence"
        if d is not None:
            ev = float(stake) * (probability * d - 1)
    return EVReport(ticket, probability, fair, float(stake), currency, d, ev, mode,
                    executable, reason, rules.version, quote.source_hash if quote else None,
                    dict(probability_metadata or {}))


def joint_return_matrix(tickets: Sequence[Ticket], scenarios: JointScenarios, rules: RuleInputs,
                        decimal_returns: Sequence[float], *,
                        scenario_dividends: Sequence[Mapping[Ticket, Quote]] | None = None) -> np.ndarray:
    """Pre-race conditional return scenarios, never actual final dividends.

    Tied outcomes require a hypothetical per-winning-combination dividend map;
    the ordinary quote is not split or reused as a dead-heat payout.
    """
    ticket_probabilities(tickets, scenarios, rules)
    d = np.asarray(decimal_returns, dtype=float)
    if d.shape != (len(tickets),) or not np.isfinite(d).all() or (d < 0).any():
        raise ValueError("Invalid payout vector")
    if scenario_dividends is not None and len(scenario_dividends) != len(scenarios.finishes):
        raise ValueError("One dividend scenario map per joint outcome required")
    if scenario_dividends is None and any(any(len(g) > 1 for g in f.groups) for f in scenarios.finishes):
        raise ValueError("Dead heats require explicit per-combination hypothetical dividends")
    rows = []
    for index, finish in enumerate(scenarios.finishes):
        row = []
        for ticket, multiple in zip(tickets, d):
            if finish.abandoned or set(ticket.runners) & set(finish.scratches):
                if rules.scratch_policy != "refund_selected":
                    raise ValueError("Refund rules unverified")
                row.append(1.0)
            else:
                wins = ticket_wins(ticket, finish, rules)
                tied = any(len(g) > 1 for g in finish.groups)
                supplied = scenario_dividends[index].get(ticket) if scenario_dividends is not None else None
                if wins and supplied is not None:
                    supplied.validate(ticket, rules, rules.currency)
                    if supplied.status != QuoteStatus.SCENARIO:
                        raise ValueError("Conditional payout scenarios must not use final dividends")
                    row.append(supplied.decimal_return)
                elif wins and tied:
                    raise ValueError("Missing per-combination dead-heat dividend scenario")
                else:
                    row.append(float(multiple) if wins else 0.0)
        rows.append(row)
    return np.asarray(rows, dtype=float)
