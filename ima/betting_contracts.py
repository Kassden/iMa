"""Paper-only money and rule contracts; no implicit jurisdiction/date defaults."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from enum import Enum
import math

from ima.pools import canonical_pool_name

POOL_SIZE = {"WIN": 1, "PLACE": 1, "QIN": 2, "QPL": 2, "FORECAST": 2,
             "TRI": 3, "TIERCE": 3, "FIRST4": 4, "QUARTET": 4}
UNORDERED = frozenset({"QIN", "QPL", "TRI", "FIRST4"})


def pool_name(value: str) -> str:
    if value.strip().upper() in {"FORECAST", "FCT"}:
        return "FORECAST"
    return canonical_pool_name(value)


def money(value: Decimal | str | float | int) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValueError("Invalid money value") from error
    if not result.is_finite() or result < 0:
        raise ValueError("Money must be finite and nonnegative")
    return result


def aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Timestamp must include timezone")
    return value


@dataclass(frozen=True)
class Ticket:
    race_id: str
    pool: str
    runners: tuple[str, ...]

    def __post_init__(self):
        pool = pool_name(self.pool)
        runners = tuple(str(r) for r in self.runners)
        if not self.race_id or len(runners) != POOL_SIZE[pool]:
            raise ValueError("Invalid race or selection length")
        if any(not r for r in runners) or len(set(runners)) != len(runners):
            raise ValueError("Selections must be distinct nonempty runner IDs")
        object.__setattr__(self, "pool", pool)
        object.__setattr__(self, "runners", tuple(sorted(runners)) if pool in UNORDERED else runners)


@dataclass(frozen=True)
class RuleInputs:
    """Caller-supplied effective rules, including their affirmative evidence.

    A citation alone is not validation. Brackets, flexi expansions and special
    consolation pools require external normalization and explicit dividends.
    """
    version: str
    source: str
    currency: str
    effective_from: datetime
    effective_to: datetime
    paid_places: int
    qpl_places: int
    minimum_stake: Decimal
    stake_increment: Decimal
    verified: bool = False
    scratch_policy: str = "unverified"
    available_pools: tuple[str, ...] = tuple(POOL_SIZE)

    def __post_init__(self):
        if not self.version or not self.source or len(self.currency) != 3:
            raise ValueError("Rule version, provenance and ISO currency are required")
        if aware(self.effective_from) >= aware(self.effective_to):
            raise ValueError("Invalid effective rule interval")
        if self.paid_places not in {1, 2, 3} or self.qpl_places not in {2, 3}:
            raise ValueError("Explicit supported place counts required")
        for name in ("minimum_stake", "stake_increment"):
            value = money(getattr(self, name))
            if value <= 0:
                raise ValueError("Wager units must be positive")
            object.__setattr__(self, name, value)
        if self.minimum_stake % self.stake_increment:
            raise ValueError("Minimum stake must be a whole wager increment")
        if self.scratch_policy not in {"unverified", "refund_selected"}:
            raise ValueError("Unsupported scratch policy")
        object.__setattr__(self, "available_pools", tuple(pool_name(p) for p in self.available_pools))

    def validate_ticket(self, ticket: Ticket, at: datetime, currency: str):
        if not self.effective_from <= aware(at) < self.effective_to:
            raise ValueError("Rules not effective at decision time")
        if currency != self.currency or ticket.pool not in self.available_pools:
            raise ValueError("Currency mismatch or unavailable pool")

    def validate_stake(self, stake):
        stake = money(stake)
        if stake and (stake < self.minimum_stake or stake % self.stake_increment):
            raise ValueError("Stake violates physical ticket units")
        return stake


class QuoteConvention(str, Enum):
    DECIMAL_RETURN = "decimal_return"
    NET_ODDS = "net_odds"
    DIVIDEND = "dividend"


class QuoteStatus(str, Enum):
    PRE_RACE = "pre_race"
    FINAL = "final"
    SCENARIO = "scenario"


@dataclass(frozen=True)
class Quote:
    ticket: Ticket
    value: Decimal
    convention: QuoteConvention
    unit_stake: Decimal
    currency: str
    quoted_at: datetime
    available_at: datetime
    source_hash: str
    rule_version: str
    status: QuoteStatus = QuoteStatus.PRE_RACE
    close_status: str = "open"
    indicative: bool = True
    uncertainty: str = "Unmodeled pari-mutuel dividend movement"

    def __post_init__(self):
        object.__setattr__(self, "value", money(self.value))
        object.__setattr__(self, "unit_stake", money(self.unit_stake))
        object.__setattr__(self, "convention", QuoteConvention(self.convention))
        object.__setattr__(self, "status", QuoteStatus(self.status))
        if self.unit_stake <= 0 or len(self.currency) != 3:
            raise ValueError("Positive dividend unit and ISO currency required")
        if aware(self.available_at) < aware(self.quoted_at):
            raise ValueError("Availability precedes quote timestamp")
        if not self.source_hash or not self.rule_version:
            raise ValueError("Quote provenance required")
        if self.close_status not in {"open", "closed", "unknown"}:
            raise ValueError("Invalid close status")

    @property
    def decimal_return(self) -> float:
        if self.convention == QuoteConvention.DIVIDEND:
            value = self.value / self.unit_stake
        elif self.convention == QuoteConvention.NET_ODDS:
            value = self.value + 1
        else:
            value = self.value
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("Return multiple outside numerical range")
        return result

    def validate(self, ticket: Ticket, rules: RuleInputs, currency: str):
        if self.ticket != ticket or self.currency != currency or self.rule_version != rules.version:
            raise ValueError("Quote ticket, currency or effective rule mismatch")

    def executable_at(self, at: datetime, race_start: datetime, max_age: timedelta) -> bool:
        aware(at)
        aware(race_start)
        if max_age.total_seconds() < 0:
            raise ValueError("Quote max age must be nonnegative")
        return (self.status == QuoteStatus.PRE_RACE and self.close_status == "open"
                and self.available_at <= at < race_start
                and self.quoted_at < race_start and at - self.quoted_at <= max_age)
