"""Bounded, offline paper research over frozen current-campaign predictions.

This engine has no persistence, network, model fitting or wager submission path.
The controller owns evidence authorization and durable action receipts.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .betting_contracts import POOL_SIZE, Quote, QuoteStatus, RuleInputs, Ticket, pool_name
from .betting_ev import (conditional_choice_scenarios, evaluate_ticket, joint_return_matrix,
                        required_depth, scenarios_from_rank_distribution, ticket_probabilities)
from .betting_stakes import StakePolicy, size_portfolio
from .probabilistic_adapters import win_to_joint


class KellySizingPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    kelly_fraction: float = Field(default=0.25, ge=0, le=1)
    max_race_fraction: float = Field(default=0.1, ge=0, le=1)
    max_ticket_fraction: float = Field(default=0.05, ge=0, le=1)
    minimum_cash_fraction: float = Field(default=0.5, ge=0, le=1)
    payout_multiplier: float = Field(default=1, gt=0, le=1)


class PaperResearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    request_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    evidence_id: str = Field(min_length=1, max_length=256)
    attempt_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    model_weights: tuple[float, ...] | None = None
    pools: tuple[str, ...] = ("WIN", "PLACE", "QIN", "TRI")
    probability_basis: Literal["fundamental", "blended"] = "fundamental"
    quote_mode: Literal["fair_price", "scenario"] = "fair_price"
    scenario_decimal_returns: dict[str, float] = Field(default_factory=dict)
    stake: Decimal = Field(default=Decimal("10"), gt=0, le=10000)
    currency: Literal["HKD"] = "HKD"
    bankroll: Decimal | None = Field(default=None, gt=0, le=1000000000)
    kelly_policy: KellySizingPolicy | None = None
    simulations: int = Field(default=10000, ge=1000, le=100000, strict=True)
    max_races: int = Field(default=100, ge=1, le=1000, strict=True)
    seed: int = Field(default=42, ge=0, le=2**32-1, strict=True)
    paid_places: Literal[1, 2, 3] = 3
    qpl_places: Literal[2, 3] = 3

    @field_validator("attempt_ids")
    @classmethod
    def safe_attempts(cls, values):
        import re
        if len(set(values)) != len(values) or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", v) for v in values):
            raise ValueError("Unique current attempt IDs, not paths, are required")
        return values

    @field_validator("pools")
    @classmethod
    def canonical_pools(cls, values):
        canonical = tuple(pool_name(v) for v in values)
        if not canonical or len(set(canonical)) != len(canonical):
            raise ValueError("Nonempty unique supported pools required")
        return canonical

    @field_validator("scenario_decimal_returns")
    @classmethod
    def payout_inputs(cls, values):
        canonical = {}
        for name, value in values.items():
            key = pool_name(name)
            if key in canonical or not math.isfinite(value) or value < 0:
                raise ValueError("Unique finite nonnegative hypothetical total returns required")
            canonical[key] = value
        return canonical

    @model_validator(mode="after")
    def coherent(self):
        weights = self.model_weights if self.model_weights is not None else tuple(1.0 for _ in self.attempt_ids)
        if len(weights) != len(self.attempt_ids) or any(not math.isfinite(w) or w <= 0 for w in weights):
            raise ValueError("One finite positive weight per attempt required")
        scale = max(weights)
        scaled = [w / scale for w in weights]
        if any(w == 0 for w in scaled):
            raise ValueError("Weight dynamic range cannot preserve positive normalized weights")
        object.__setattr__(self, "model_weights", tuple(w / sum(scaled) for w in scaled))
        if self.quote_mode == "fair_price" and self.scenario_decimal_returns:
            raise ValueError("Fair-price mode cannot carry payouts")
        if self.quote_mode == "scenario" and set(self.scenario_decimal_returns) != set(self.pools):
            raise ValueError("Scenario mode requires explicit payouts for exactly the requested pools")
        if self.kelly_policy is not None and self.bankroll is None:
            raise ValueError("Kelly policy requires a bankroll")
        if self.stake % Decimal("10"):
            raise ValueError("Paper stake must be a whole assumed HKD10 unit")
        if self.simulations * self.max_races > 1000000:
            raise ValueError("Paper action exceeds one-million joint-outcome work budget")
        return self


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _artifact(result, name, campaign, attempt):
    path = Path(result["artifacts"][name])
    directory = campaign / "trials" / attempt
    expected = directory.resolve()
    if directory.is_symlink() or path.name != {"predictions": "predictions.csv", "protocol": "protocol.json"}[name]:
        raise ValueError("Paper inputs must be canonical actual attempt artifacts")
    if not path.is_absolute() or path.resolve().parent != expected or not expected.is_relative_to(campaign):
        raise ValueError("Paper inputs must be actual current attempt output artifacts")
    if not path.is_file() or path.is_symlink():
        raise ValueError("Paper artifact must be an existing regular file")
    return path


def _load_paper_inputs(request: PaperResearchRequest | dict, *, campaign_dir: Path,
                       terminal_results: Sequence[dict[str, Any]]):
    request = PaperResearchRequest.model_validate(request)
    campaign = Path(campaign_dir).resolve()
    indexed = {}
    for row in terminal_results:
        attempt = row["attempt_id"]
        if attempt in indexed:
            raise ValueError("Duplicate current terminal attempt")
        indexed[attempt] = row
    frames, identities, provenance = [], [], []
    key_columns = ["fold_id", "race_id", "horse_no", "date"]
    column = "model_probability" if request.probability_basis == "fundamental" else "selected_probability"
    for attempt in request.attempt_ids:
        row = indexed.get(attempt)
        if row is None or row.get("status") != "completed":
            raise ValueError("Paper actions require current completed attempts")
        result = row.get("result", row)
        if result.get("attempt_id") != attempt or result.get("status") != "completed" or result.get("target_kind") != "win_probability":
            raise ValueError("Paper action requires completed win-probability predictions")
        lineage = result["lineage"]
        names = ("dataset_hash", "protocol_id", "protocol_hash", "evaluation_population_id",
                 "evaluation_population_hash", "availability_policy")
        if any(not lineage.get(name) for name in names):
            raise ValueError("Missing frozen comparison/population identity")
        identity = {name: lineage[name] for name in names}
        identity.update({name: lineage.get(name) for name in
                         ("dataset_id", "declared_evaluation_population_id", "comparison_contract_id")})
        identities.append(identity)
        if identity != identities[0]:
            raise ValueError("Cannot mix different dataset/protocol/comparison populations")
        path = _artifact(result, "predictions", campaign, attempt)
        protocol = _artifact(result, "protocol", campaign, attempt)
        if hashlib.sha256(protocol.read_bytes()).hexdigest() != lineage["protocol_hash"]:
            raise ValueError("Frozen protocol artifact hash mismatch")
        if path.stat().st_size > 128 * 1024**2:
            raise ValueError("Scored prediction artifact exceeds 128MiB read budget")
        prediction_bytes = path.read_bytes()
        frame = pd.read_csv(io.BytesIO(prediction_bytes), dtype={"fold_id": str, "race_id": str, "horse_no": str})
        required = {*key_columns, "target_win", column}
        if not required.issubset(frame) or frame.empty or frame[list(required)].isna().any().any():
            raise ValueError("Missing scored runner keys, outcomes or probabilities")
        if frame.duplicated(["fold_id", "race_id", "horse_no"]).any():
            raise ValueError("Duplicate scored runner keys")
        frame = frame.sort_values(key_columns, kind="stable").reset_index(drop=True)
        population = frame[key_columns + ["target_win"]]
        population_hash = hashlib.sha256(population.to_json(orient="records", date_format="iso").encode()).hexdigest()
        if population_hash != lineage["evaluation_population_hash"] or lineage["evaluation_population_id"] != "evaluated-" + population_hash:
            raise ValueError("Scored evaluation population hash mismatch")
        if frames and not population.equals(frames[0][key_columns + ["target_win"]]):
            raise ValueError("Scored runner/outcome keys do not match exactly")
        probabilities = pd.to_numeric(frame[column], errors="raise").to_numpy(float)
        if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
            raise ValueError("Invalid win marginals")
        grouped = frame.groupby(["fold_id", "race_id"], sort=False)
        if not np.allclose(grouped[column].sum(), 1, atol=1e-8, rtol=0):
            raise ValueError("Win marginals must already sum to one per race")
        if not frame.target_win.isin([0, 1]).all() or not grouped.target_win.sum().eq(1).all():
            raise ValueError("Stored development outcomes require one winner per race")
        frames.append(frame)
        provenance.append({"attempt_id": attempt, "recipe_hash": result.get("recipe_hash"),
                           "prediction_sha256": hashlib.sha256(prediction_bytes).hexdigest(),
                           "code_revision": lineage.get("code_revision"),
                           "environment_hash": lineage.get("environment_hash")})
    mixed = frames[0].copy()
    mixed["paper_probability"] = sum(weight * frame[column].to_numpy(float)
                                     for weight, frame in zip(request.model_weights, frames))
    races = list(mixed.groupby(["fold_id", "race_id"], sort=True))
    races.sort(key=lambda item: (str(item[1].date.min()), item[0]))
    return request, races, identities[0], provenance, len(mixed)


def validate_paper_research(request: PaperResearchRequest | dict, *, campaign_dir: Path,
                            terminal_results: Sequence[dict[str, Any]]) -> dict:
    """Validate frozen development inputs without running outcome simulations."""
    request, races, comparison, provenance, rows = _load_paper_inputs(
        request, campaign_dir=campaign_dir, terminal_results=terminal_results)
    return {"comparison": comparison, "evaluation_key": _digest(comparison),
            "model_ids": provenance, "rows": rows, "available_races": len(races),
            "requested_races": min(len(races), request.max_races)}


def evaluate_paper_research(request: PaperResearchRequest | dict, *, campaign_dir: Path,
                            terminal_results: Sequence[dict[str, Any]]) -> dict:
    """Evaluate frozen current-campaign development outputs, never protected data."""
    request, races, comparison, provenance, _ = _load_paper_inputs(
        request, campaign_dir=campaign_dir, terminal_results=terminal_results)
    reports, skipped = [], []
    for index, ((fold, race_id), frame) in enumerate(races[:request.max_races]):
        try:
            reports.append(_race_report(request, frame, fold, race_id, index))
        except ValueError as exc:
            skipped.append({"fold_id": fold, "race_id": race_id, "reason": str(exc)})
    output = {"schema_version": 1, "status": "completed" if reports else "unsupported",
              "paper_only": True, "executable_evidence": False,
              "request_id": request.request_id, "evidence_id": request.evidence_id,
              "request": request.model_dump(mode="json"), "model_ids": provenance,
              "comparison": comparison, "evaluation_key": _digest(comparison),
              "probability_basis": request.probability_basis,
              "scenario_payout_contract": {"source": "explicit request inputs only; no observed quote", "convention": "total decimal return", "currency": "HKD", "takeout_subtracted_again": False} if request.quote_mode == "scenario" else None,
              "ex_post_market_tainted": request.probability_basis == "blended",
              "assumptions": "Weighted normalized win marginals -> assumed Plackett-Luce conditional orders; not independent place products. Top-strength ticket per pool, no outcome-based ticket selection. Development reuse is not independent strategy validation.",
              "coverage": {"available_races": len(races), "requested_races": min(len(races), request.max_races),
                           "evaluated_races": len(reports), "skipped_races": skipped,
                           "truncated_races": max(0, len(races)-request.max_races)},
              "races": reports,
              "resource_bounds": {"maximum_joint_outcomes": 1000000, "maximum_prediction_bytes_per_attempt": 128*1024**2,
                                  "maximum_runners_per_race": 32, "maximum_tickets_per_race": len(request.pools)},
              "unsupported_requirements": ["Official effective pool rules/quote history are not supplied; rule counts and HKD10 units are hypothetical inputs.",
                  "No ex-post dividends, realized profit or protected confirmation inputs are read.",
                  "No independent chronological strategy selection/backtest or native packaged joint model replay in this action."]}
    return json.loads(json.dumps(output, default=str, allow_nan=False))


def _race_report(request, frame, fold, race_id, index):
    strengths = frame.paper_probability.to_numpy(float)
    strengths = strengths / strengths.sum()
    ids = tuple(frame.horse_no)
    if len(ids) > 32:
        raise ValueError("Paper race field exceeds bounded 32 runners")
    order = tuple(ids[i] for i in sorted(range(len(ids)), key=lambda i: (-strengths[i], ids[i])))
    tickets = [Ticket(race_id, pool, order[:POOL_SIZE[pool]]) for pool in request.pools]
    at = datetime(2000, 1, 1, tzinfo=timezone.utc)
    rules = RuleInputs("paper-hypothetical-v1", "explicit paper action assumptions; NOT verified official rules",
        "HKD", at-timedelta(days=1), at+timedelta(days=1), request.paid_places,
        request.qpl_places, Decimal(10), Decimal(10), verified=False,
        scratch_policy="refund_selected", available_pools=request.pools)
    depth = required_depth(tickets, rules)
    if depth > len(ids):
        raise ValueError("Requested pool/place depth exceeds field; provide explicit smaller counts")
    if math.perm(len(ids), depth) <= request.simulations:
        scenarios = conditional_choice_scenarios(ids, strengths, depth=depth, max_orders=request.simulations)
        method = "exact_prefix_enumeration"
    else:
        rank = win_to_joint(strengths, ids, method="sampled", seed=(request.seed+index) % 2**32,
                            draws=request.simulations, chunk_size=min(4096, request.simulations))
        scenarios = scenarios_from_rank_distribution(rank, source_id="paper-action:PL")
        method = "fixed_budget_sampled_PL"
    probabilities = ticket_probabilities(tickets, scenarios, rules)
    evs = []
    for ticket, probability in zip(tickets, probabilities):
        quote = None
        if request.quote_mode == "scenario":
            quote = Quote(ticket=ticket, value=Decimal(str(request.scenario_decimal_returns[ticket.pool])),
                          convention="decimal_return", unit_stake=Decimal(1), currency="HKD",
                          quoted_at=at, available_at=at, status=QuoteStatus.SCENARIO,
                          source_hash=_digest(request.scenario_decimal_returns), rule_version=rules.version,
                          uncertainty="Explicit hypothetical payout; NOT an observed market quote")
        evs.append(evaluate_ticket(ticket, probability, request.stake, "HKD", rules, at=at,
            race_start=at+timedelta(hours=1), quote=quote, quote_mode=request.quote_mode,
            probability_metadata=scenarios.metadata))
    sizing, sizing_note = None, "No payout: fair prices cannot fund Kelly sizing"
    if request.quote_mode == "scenario" and request.bankroll is not None:
        sizing_note = "Hypothetical scenario-only sizing; not executable or an official wager recommendation"
        if method != "exact_prefix_enumeration":
            sizing_note += "; unavailable because fixed-budget samples lack convergence/tail evidence"
        elif any(ev.decimal_return <= 1 for ev in evs):
            sizing_note += "; unavailable because total returns must exceed one"
        else:
            # The existing solver accepts executable contracts only. This private,
            # hypothetical copy satisfies its numerical boundary, never public evidence.
            solver_rules = replace(rules, verified=True)
            solver_reports = [replace(ev, mode="quoted", executable_evidence=True) for ev in evs]
            policy = StakePolicy(request.bankroll, "HKD", **(request.kelly_policy or KellySizingPolicy()).model_dump())
            gross = joint_return_matrix(tickets, scenarios, solver_rules, [ev.decimal_return for ev in evs])
            sizing = asdict(size_portfolio(solver_reports, scenarios, gross, policy, solver_rules))
            sizing.update(mode="hypothetical_scenario", executable_evidence=False,
                          assumptions=sizing["assumptions"] + "; hypothetical payouts and rule inputs, no real market quote")
    winner = frame.loc[frame.target_win.eq(1)].paper_probability.iloc[0]
    return {"fold_id": fold, "race_id": race_id, "method": method,
            "joint_assumptions": scenarios.assumptions, "distribution_metadata": scenarios.metadata,
            "probability_bounds": "Exact PL under stated assumption" if method == "exact_prefix_enumeration" else "Empirical fixed-budget PL; unseen outcomes are not proven zero and sizing is disabled",
            "rules": asdict(rules), "reports": [asdict(ev) for ev in evs],
            "sizing": sizing, "sizing_note": sizing_note,
            "development_win_log_loss": None if winner <= 0 else -math.log(winner),
            "joint_outcome_metrics": None,
            "joint_metrics_note": "Stored predictions contain winner only, not complete finish orders"}
