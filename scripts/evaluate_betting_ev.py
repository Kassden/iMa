"""Deterministic, offline-only paper EV / joint Kelly / chronological settlement CLI.

Input JSON: rules (RuleInputs fields), race_id, decision_at, race_start,
tickets [{pool, runners, quote?}], and either joint {orders, probabilities,
assumptions, source_id} or runners/strengths for the legacy conditional-choice
baseline. Quotes use Quote fields except ticket. --replay accepts records with
the same predeclared tickets plus finish and dividends, never selects from finals.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timedelta
from decimal import Decimal
import json
import hashlib
import math
from pathlib import Path

from ima.betting_contracts import Quote, RuleInputs, Ticket, pool_name
from ima.betting_ev import (conditional_choice_scenarios, evaluate_ticket, joint_return_matrix,
                            required_depth, scenarios_from_orders, scenarios_from_rank_distribution,
                            ticket_probabilities)
from ima.betting_settlement import Finish, settle_ticket
from ima.betting_stakes import StakePolicy, size_portfolio, size_single


def timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def rules_from_json(data):
    values = dict(data)
    for key in ("effective_from", "effective_to"):
        values[key] = timestamp(values[key])
    return RuleInputs(**values)


def quote_from_json(data, ticket):
    if data is None:
        return None
    values = dict(data)
    for key in ("quoted_at", "available_at"):
        values[key] = timestamp(values[key])
    return Quote(ticket=ticket, **values)


def evaluate_payload(data, *, stake="10", currency="HKD", pools=None, quote_mode="quoted",
                     max_age_seconds=300, bankroll=None, kelly_fraction=0.25, portfolio=False):
    rules = rules_from_json(data["rules"])
    at, start = timestamp(data["decision_at"]), timestamp(data["race_start"])
    selected = [t for t in data["tickets"] if not pools or pool_name(t["pool"]) in pools]
    tickets = tuple(Ticket(data["race_id"], t["pool"], tuple(t["runners"])) for t in selected)
    if not tickets or len(set(tickets)) != len(tickets):
        raise ValueError("Empty or duplicate ticket selection")
    if "rank_distribution" in data:
        from ima.rank_distributions import RankDistribution
        raw = data["rank_distribution"]
        distribution = RankDistribution(tuple(raw["runner_ids"]),
            {tuple(row["order"]): row["probability"] for row in raw["outcomes"]},
            raw.get("metadata", {}))
        if len(distribution.order_probabilities) != len(raw["outcomes"]):
            raise ValueError("Duplicate rank outcome rows")
        scenarios = scenarios_from_rank_distribution(distribution, source_id=raw["source_id"])
    elif "joint" in data:
        joint = data["joint"]
        scenarios = scenarios_from_orders(joint["orders"], joint["probabilities"],
                                         assumptions=joint["assumptions"], source_id=joint["source_id"])
    else:
        scenarios = conditional_choice_scenarios(data["runners"], data["strengths"],
                                                depth=required_depth(tickets, rules))
    probabilities = ticket_probabilities(tickets, scenarios, rules)
    quotes = tuple(quote_from_json(row.get("quote"), t) for t, row in zip(tickets, selected))
    if not math.isfinite(max_age_seconds) or max_age_seconds < 0:
        raise ValueError("Quote max age must be finite and nonnegative")
    reports = tuple(evaluate_ticket(t, p, stake, currency, rules, at=at, race_start=start,
        quote=q, quote_mode=quote_mode, max_age=timedelta(seconds=max_age_seconds),
        probability_metadata=scenarios.metadata)
        for t, p, q in zip(tickets, probabilities, quotes))
    output = {"paper_only": True, "assumptions": scenarios.assumptions,
              "scenario_source": scenarios.source_id, "scenario_count": len(scenarios.finishes),
              "distribution_metadata": scenarios.metadata, "rules": asdict(rules),
              "decision_at": at, "race_start": start,
              "quote_evidence": [asdict(q) if q else None for q in quotes],
              "reports": [asdict(r) for r in reports],
              "scenario_hash": hashlib.sha256(json.dumps(asdict(scenarios), sort_keys=True,
                                                          default=str).encode()).hexdigest()}
    if bankroll is not None:
        policy_data = dict(data.get("stake_policy", {}))
        policy_data.update(bankroll=bankroll, currency=currency, kelly_fraction=kelly_fraction)
        policy = StakePolicy(**policy_data)
        output["stake_policy"] = asdict(policy)
        if all(r.executable_evidence for r in reports):
            if portfolio:
                gross = joint_return_matrix(tickets, scenarios, rules, [r.decimal_return for r in reports])
                output["sizing"] = asdict(size_portfolio(reports, scenarios, gross, policy, rules))
            else:
                output["sizing"] = [asdict(size_single(r, policy, rules)) for r in reports]
                output["sizing_note"] = "Isolated single-ticket alternatives; allocations must not be added"
        else:
            output["sizing"] = None
            output["sizing_note"] = "Unavailable pre-race evidence or unverified rules; no stake recommendation"
    return output


def render_text(result):
    if "reports" not in result:
        return json.dumps(result, indent=2, default=str, allow_nan=False)
    lines = ["PAPER ONLY", f"Assumptions: {result['assumptions']}",
             "Pool      Selection        P(hit)    Fair d    Quote d      EV       Mode"]
    for report in result["reports"]:
        def fmt(value):
            return "-" if value is None else f"{value:.4f}"
        t = report["ticket"]
        lines.append(f"{t['pool']:<9} {','.join(t['runners']):<16} {report['probability']:>7.4f} "
                     f"{fmt(report['fair_decimal_return']):>9} {fmt(report['decimal_return']):>10} "
                     f"{fmt(report['expected_profit']):>9} {report['mode']}")
    lines.append(f"Currency: {result['reports'][0]['currency']}; rule version: {result['rules']['version']}")
    if "sizing" in result:
        lines.append("Sizing: " + json.dumps(result["sizing"], default=str, allow_nan=False))
    if result.get("sizing_note"):
        lines.append(result["sizing_note"])
    return "\n".join(lines)


def replay_payload(data, *, bankroll, currency, strategy="fixed", kelly_fraction=0.25):
    """Externally predeclared ticket candidates; size only from pre-race evidence.

    Fixed, fractional and full-Kelly controls share the same candidate set and
    frozen risk caps. Settlement prices never enter evaluate_payload.
    """
    if strategy not in {"fixed", "fractional_kelly", "full_kelly"}:
        raise ValueError("Unknown frozen replay strategy")
    balance = Decimal(str(bankroll))
    if balance <= 0:
        raise ValueError("Positive replay bankroll required")
    initial, peak, drawdown, previous = balance, balance, Decimal(0), None
    day_spend = {}
    rows = []
    for record in data["records"]:
        at, start = timestamp(record["decision_at"]), timestamp(record["race_start"])
        if previous is not None and at <= previous:
            raise ValueError("Replay decision must follow previous completed settlement")
        settled_at = timestamp(record["settled_at"])
        if settled_at < start:
            raise ValueError("Settlement precedes race")
        rules = rules_from_json(record["rules"])
        finish = Finish(**record["finish"])
        ticket_rows = [dict(r) for r in record["tickets"]]
        frozen_policy = dict(record.get("stake_policy", {}))
        if frozen_policy.get("remaining_day_budget") is not None:
            frozen_policy["remaining_day_budget"] = max(Decimal(0),
                Decimal(str(frozen_policy["remaining_day_budget"])) - day_spend.get(at.date(), Decimal(0)))
        replay_policy = StakePolicy(**(frozen_policy | {"bankroll": balance, "currency": currency,
            "kelly_fraction": 1.0 if strategy == "full_kelly" else kelly_fraction}))
        if strategy != "fixed":
            sized_record = dict(record)
            sized_record["stake_policy"] = frozen_policy
            evaluated = evaluate_payload(sized_record, stake=str(rules.minimum_stake), currency=currency, bankroll=balance,
                kelly_fraction=1.0 if strategy == "full_kelly" else kelly_fraction, portfolio=True)
            sizing = evaluated["sizing"]
            if sizing is None or not sizing["success"]:
                raise ValueError("Replay policy lacks eligible prices or solver validation")
            for row, amount in zip(ticket_rows, sizing["stakes"]):
                row["stake"] = amount
        selections = [(Ticket(record["race_id"], r["pool"], tuple(r["runners"])), r)
                      for r in ticket_rows]
        if len({t for t, _ in selections}) != len(selections):
            raise ValueError("Duplicate physical replay tickets")
        dividends = {}
        for row in record.get("dividends", []):
            ticket = Ticket(record["race_id"], row["pool"], tuple(row["runners"]))
            quote = quote_from_json(row["quote"], ticket)
            if quote.available_at > settled_at:
                raise ValueError("Dividend not yet available at settlement readback")
            if ticket in dividends:
                raise ValueError("Duplicate dividend evidence")
            dividends[ticket] = quote
        cost = sum((Decimal(str(r["stake"])) for _, r in selections), Decimal(0))
        if cost > balance:
            raise ValueError("Replay stake exceeds available cash")
        if cost > replay_policy.cap_amount:
            raise ValueError("Replay stake exceeds frozen race/day/open/cash caps")
        if any(Decimal(str(r["stake"])) > Decimal(str(replay_policy.max_ticket_fraction)) * balance
               for _, r in selections):
            raise ValueError("Replay stake exceeds frozen per-ticket cap")
        day_spend[at.date()] = day_spend.get(at.date(), Decimal(0)) + cost
        settlements = [settle_ticket(t, r["stake"], finish, rules, dividends,
            selected_at=at, race_start=start, currency=currency) for t, r in selections]
        if any(s.net_profit is None for s in settlements):
            raise ValueError("Unresolved settlement; cannot fabricate bankroll/ROI")
        profit = sum((s.net_profit for s in settlements), Decimal(0))
        old = balance
        balance += profit
        peak = max(peak, balance)
        drawdown = max(drawdown, (peak - balance) / peak)
        rows.append({"race_id": record["race_id"], "stake": cost, "net_profit": profit,
                     "bankroll": balance, "realized_log_growth": math.log(float(balance / old))
                     if balance > 0 else None, "settlements": [asdict(s) for s in settlements]})
        previous = settled_at
        if balance <= 0:
            break
    total_cost = sum((r["stake"] for r in rows), Decimal(0))
    return {"paper_only": True, "mode": "predeclared_" + strategy + "_ex_post_replay",
            "initial_bankroll": initial, "final_bankroll": balance,
            "total_stake": total_cost, "net_profit": balance-initial,
            "roi": (balance-initial)/total_cost if total_cost else None,
            "maximum_drawdown": drawdown, "records": rows,
            "provenance_limit": "Recorded decision timestamps are assertions, not independent commitment proof"}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline paper pool EV and correlated Kelly research")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--stake", default="10")
    parser.add_argument("--currency", default="HKD")
    parser.add_argument("--pool", action="append")
    parser.add_argument("--quote-mode", choices=("quoted", "fair_price", "scenario", "ex_post"), default="quoted")
    parser.add_argument("--max-quote-age-seconds", type=float, default=300)
    parser.add_argument("--bankroll")
    parser.add_argument("--kelly-fraction", type=float, default=0.25)
    parser.add_argument("--portfolio", action="store_true")
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--replay-policy", choices=("fixed", "fractional_kelly", "full_kelly"), default="fixed")
    parser.add_argument("--compare-policies", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--format", choices=("json", "text"), default="json")
    args = parser.parse_args(argv)
    if args.portfolio and args.bankroll is None:
        parser.error("--portfolio requires --bankroll")
    if args.compare_policies and not args.replay:
        parser.error("--compare-policies requires --replay")
    try:
        data = json.loads(args.input.read_text(encoding="utf-8"))
        if args.replay:
            if args.bankroll is None:
                raise ValueError("--replay requires --bankroll")
            if args.compare_policies:
                result = {strategy: replay_payload(data, bankroll=args.bankroll, currency=args.currency,
                          strategy=strategy, kelly_fraction=args.kelly_fraction)
                          for strategy in ("fixed", "fractional_kelly", "full_kelly")}
            else:
                result = replay_payload(data, bankroll=args.bankroll, currency=args.currency,
                                        strategy=args.replay_policy, kelly_fraction=args.kelly_fraction)
        else:
            result = evaluate_payload(data, stake=args.stake, currency=args.currency,
                pools={pool_name(p) for p in args.pool} if args.pool else None,
                quote_mode=args.quote_mode, max_age_seconds=args.max_quote_age_seconds,
                bankroll=args.bankroll, kelly_fraction=args.kelly_fraction, portfolio=args.portfolio)
        rendered = render_text(result) if args.format == "text" else json.dumps(result, indent=2,
            default=lambda o: str(o) if isinstance(o, Decimal) else o.isoformat(), allow_nan=False)
        if args.output:
            args.output.write_text(rendered + "\n", encoding="utf-8")
        else:
            print(rendered)
        return 0
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
