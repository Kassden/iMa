"""Explicit per-combination settlement, including ties and selected-runner refunds."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Mapping

from ima.betting_contracts import POOL_SIZE, UNORDERED, Quote, QuoteStatus, RuleInputs, Ticket, aware


@dataclass(frozen=True)
class Finish:
    """Tied groups occupy consecutive physical slots; nonfinishers are omitted."""
    groups: tuple[tuple[str, ...], ...]
    scratches: tuple[str, ...] = ()
    abandoned: bool = False

    def __post_init__(self):
        groups = tuple(tuple(str(r) for r in g) for g in self.groups)
        scratches = tuple(str(r) for r in self.scratches)
        runners = [r for group in groups for r in group]
        if any(not g for g in groups) or len(runners) != len(set(runners)):
            raise ValueError("Invalid finish groups")
        if len(set(scratches)) != len(scratches) or set(runners) & set(scratches):
            raise ValueError("Scratched runner cannot finish")
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "scratches", scratches)


def ticket_wins(ticket: Ticket, finish: Finish, rules: RuleInputs) -> bool:
    if finish.abandoned or set(ticket.runners) & set(finish.scratches):
        return False
    slots = []
    for group in finish.groups:
        slots.extend([set(group)] * len(group))
    if ticket.pool in {"PLACE", "QPL"}:
        count = rules.paid_places if ticket.pool == "PLACE" else rules.qpl_places
        eligible = set().union(*slots[:count]) if slots else set()
        return set(ticket.runners).issubset(eligible)
    count = POOL_SIZE[ticket.pool]
    if len(slots) < count:
        return False
    if ticket.pool in UNORDERED:
        # A tied group can straddle the payout boundary. Each admissible
        # combination must fill every top slot, not just belong to their union.
        from itertools import permutations
        return any(all(r in slots[i] for i, r in enumerate(order))
                   for order in permutations(ticket.runners))
    return all(r in slots[i] for i, r in enumerate(ticket.runners))


@dataclass(frozen=True)
class Settlement:
    ticket: Ticket
    stake: Decimal
    gross_return: Decimal | None
    net_profit: Decimal | None
    status: str
    evidence: str


def settle_ticket(ticket: Ticket, stake, finish: Finish, rules: RuleInputs,
                  dividends: Mapping[Ticket, Quote], *, selected_at: datetime,
                  race_start: datetime, currency: str) -> Settlement:
    rules.validate_ticket(ticket, selected_at, currency)
    aware(race_start)
    stake = rules.validate_stake(stake)
    if selected_at >= race_start:
        raise ValueError("Ticket must be independently selected before race start")
    if not stake:
        return Settlement(ticket, stake, Decimal(0), Decimal(0), "no_bet", "Zero physical stake")
    refunded = finish.abandoned or bool(set(ticket.runners) & set(finish.scratches))
    if refunded:
        if rules.scratch_policy != "refund_selected":
            return Settlement(ticket, stake, None, None, "unsupported_refund", rules.source)
        return Settlement(ticket, stake, stake, Decimal(0), "refund", rules.source)
    if not ticket_wins(ticket, finish, rules):
        return Settlement(ticket, stake, Decimal(0), -stake, "lost", rules.source)
    quote = dividends.get(ticket)
    if quote is None:
        return Settlement(ticket, stake, None, None, "missing_dividend", rules.source)
    quote.validate(ticket, rules, currency)
    if quote.status != QuoteStatus.FINAL or quote.quoted_at < race_start:
        raise ValueError("Settlement requires ex-post final dividend evidence")
    # Final published dividends already include pool deductions/dead-heat rules.
    if quote.convention.value == "dividend":
        gross = stake * quote.value / quote.unit_stake
    elif quote.convention.value == "net_odds":
        gross = stake * (quote.value + 1)
    else:
        gross = stake * quote.value
    return Settlement(ticket, stake, gross, gross - stake, "won_ex_post", quote.source_hash)
