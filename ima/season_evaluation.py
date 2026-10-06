"""Fixed HKD10 straight-ticket paper evaluation, with no training or service I/O.

Parent API: ``evaluate_season(frame, settlements, pools=SUPPORTED_POOLS,
top_n=1, quotes=None, exponents=OrderExponents()) -> SeasonEvaluation``.
Exponents must be frozen from earlier calibration races by the caller; this
module never fits them. Frame columns are race_id, date,
horse_no, result and model_probability. The caller must supply the COMPLETE
official starter field and probabilities frozen before racing. Optional
``field_size`` verifies the supplied starter count. ``pool_result_status(race,
pool)`` returns "eligible" or an exclusion reason, without needing probabilities.
Each pool requires unique, complete ranks 1..its paid-position cutoff. Lower
dead heats are allowed. Null results require explicit known nonfinisher codes
PU/UR/FE/DNF/DISQ/TNP in finishing_status; no ranks are invented for nonfinishers.
Unknown missing positions and invalid numeric ranks quarantine all pools.
Nonnegative finite strengths are normalized within each race (positive total).

Settlements map race IDs to lists of {pool, winning_combination,
dividend_hkd_per_10}, or are a flat list of those records with race_id added.
The parent can flatten collector {metadata, runners, dividends} payloads into
either form. Combinations accept runner sequences or strings such as
"1/2/3". Supply ALL paid combinations, including every PLACE/QPL combination.
Dividends are gross HKD returns inclusive of stake. Missing, invalid, or
inconsistent pool settlements exclude the whole pool/race. A record with
status="refund"/"dead_heat" or refund=True/dead_heat=True excludes that pool/race.
Record dead_heat must mean a tie affecting that pool's paid positions, not a
race-wide lower-position tie; adapters must compute that distinction.
Identical duplicate records are deduplicated; conflicting duplicates exclude it.

Optional quotes use a SEPARATE race-ID mapping of {pool, combination,
payout_hkd_per_10, quote_scope: "full_ticket", is_pre_race: True}. Those explicit
provenance assertions are the caller's responsibility. Only a positive finite
gross quote for that exact ticket supplies estimated_ev_hkd = p*quote - 10.
Final winner-only dividends never become quotes, including for losing tickets.
Quotes and results cannot affect selection: rank_pool_combinations selects
top_n by probability using rank_combinations mathematics, with deterministic
runner-number tie ordering, before settlement.

Return fields: ledger (one row per selected ticket, including exclusions),
with combination serialized as a slash-separated string (WIN "1", QIN "1/2"),
summary, by_pool, bankroll_curve. Summary stake/gross/profit/ROI and hit_rate
cover SETTLED tickets only; nsettled/nmissing/nexcluded count tickets, nraces
counts distinct candidate races. planned_stake_hkd covers all selected tickets.
ROI is profit/stake, hit_rate is wins/settled tickets; undefined rates are None.
The bankroll assumes HKD1000 initially, HKD10 per ticket, no compounding or Kelly,
and excludes unknown/unsupported settlements. It is a hypothetical settled-only
path, ordered by date then natural race ID (R2 before R10); callers with arbitrary
IDs must provide race-start timestamps to establish same-day chronology.
min_bankroll_capital_required_hkd includes all simultaneous stakes before
each race's returns; maximum_drawdown_hkd measures peaks to race-end balances.
can_finance_with_1000 flags whether the starting capital funds the whole schedule.
Negative balances are hypothetical only and require additional external funding;
no continued stakes or compound growth are claimed for an unfunded schedule.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from ima.pools import (
    SUPPORTED_POOLS, UNORDERED_POOLS, OrderExponents, canonical_pool_name, paid_place_count,
    rank_pool_combinations,
)


STAKE_HKD = 10.0
STARTING_BANKROLL_HKD = 1000.0
POOL_SIZE = {"WIN": 1, "PLACE": 1, "QIN": 2, "QPL": 2, "TRI": 3,
             "TIERCE": 3, "FIRST4": 4, "QUARTET": 4}
KNOWN_NONFINISHERS = {"PU", "UR", "FE", "DNF", "DISQ", "TNP"}
LEDGER_COLUMNS = [
    "race_id", "date", "pool", "rank", "combination", "stake_hkd", "probability",
    "break_even_dividend_hkd_per_10", "payout_quote_hkd_per_10", "quote_status",
    "estimated_ev_hkd", "status", "hit", "realized_gross_hkd", "realized_net_hkd",
]


@dataclass(frozen=True)
class SeasonEvaluation:
    ledger: pd.DataFrame
    summary: dict
    by_pool: dict[str, dict]
    bankroll_curve: pd.DataFrame


def _runner(value) -> str:
    text = str(value).strip()
    if not re.fullmatch(r"\d+(?:\.0+)?", text) or int(text.split(".")[0]) < 1:
        raise ValueError(f"Invalid horse number: {value!r}")
    return str(int(text.split(".")[0]))


def _combination(value, pool: str) -> tuple[str, ...]:
    if isinstance(value, str):
        values = re.split(r"[/,\s-]+", value.strip())
    elif isinstance(value, (int, float, np.number)):
        values = [value]
    else:
        values = list(value)
    runners = tuple(_runner(v) for v in values)
    if len(runners) != POOL_SIZE[pool] or len(set(runners)) != len(runners):
        raise ValueError("Invalid combination size or duplicate runners")
    if pool in UNORDERED_POOLS or pool == "QPL":
        runners = tuple(sorted(runners))
    return runners


def _positive(value) -> float | None:
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) and number > 0 else None


def _natural_order(value: str) -> tuple:
    return tuple((0, int(part)) if part.isdigit() else (1, part)
                 for part in re.split(r"(\d+)", value))


def _records(mapping: Mapping | Iterable[Mapping] | None) -> dict[str, dict[str, list[dict]]]:
    if mapping is not None and not isinstance(mapping, Mapping):
        flat = {}
        for record in mapping:
            if "race_id" not in record or pd.isna(record["race_id"]):
                raise ValueError("Flat records require race_id")
            flat.setdefault(str(record["race_id"]), []).append(record)
        mapping = flat
    grouped = {}
    for race_id, records in (mapping or {}).items():
        key = str(race_id)
        if key in grouped:
            raise ValueError("Duplicate normalized race ID")
        grouped[key] = {}
        for record in records:
            pool = canonical_pool_name(record["pool"])
            grouped[key].setdefault(pool, []).append(record)
    return grouped


def pool_result_status(race: pd.DataFrame, pool: str) -> str:
    """Validate full starter evidence and only this pool's relevant finish ranks.

    Requires result; optionally uses finishing_status and field_size. Parent
    preparation can retain a race if any requested pool is eligible, while the
    evaluator records exclusions separately for pools with ambiguous paid ranks.
    This helper never reads model probabilities or dividends.
    """
    pool = canonical_pool_name(pool)
    if "result" not in race or race.empty:
        return "excluded_incomplete_race"
    if "field_size" in race:
        sizes = pd.to_numeric(race["field_size"], errors="coerce")
        if not (sizes == len(race)).all():
            return "excluded_incomplete_race"
    results = pd.to_numeric(race["result"], errors="coerce")
    statuses = race.get("finishing_status", pd.Series("", index=race.index)).fillna("")
    nonfinishers = statuses.astype(str).str.strip().str.upper().isin(KNOWN_NONFINISHERS)
    if (results.isna() & ~(race["result"].isna() & nonfinishers)).any():
        return "excluded_incomplete_race"
    if (results.notna() & nonfinishers).any():
        return "excluded_incomplete_race"
    finite = results.dropna()
    if (not np.isfinite(finite.to_numpy(dtype=float)).all()
            or (finite < 1).any() or (finite > len(race)).any()
            or (finite != np.floor(finite)).any()):
        return "excluded_incomplete_race"
    cutoff = paid_place_count(len(race)) if pool in {"PLACE", "QPL"} else POOL_SIZE[pool]
    paid = finite[finite <= cutoff]
    if paid.duplicated().any():
        return "excluded_dead_heat"
    if sorted(paid.tolist()) != list(range(1, cutoff + 1)):
        return "excluded_incomplete_race"
    return "eligible"


def _winners(race: pd.DataFrame, pool: str) -> set[tuple[str, ...]]:
    finish = race.dropna(subset=["result"]).sort_values("result")["horse_no"].tolist()
    if pool == "PLACE":
        return {(runner,) for runner in finish[:paid_place_count(len(race))]}
    if pool == "QPL":
        return {_combination(pair, pool) for pair in
                itertools.combinations(finish[:paid_place_count(len(race))], 2)}
    return {_combination(finish[:POOL_SIZE[pool]], pool)}


def _settlement(records: list[dict], expected: set[tuple[str, ...]] | None, pool: str):
    for record in records:
        status = str(record.get("status") or "").lower().replace("-", "_").replace(" ", "_")
        if record.get("refund") or status in {"refund", "refunded", "unsupported_refund"}:
            return "excluded_refund", {}
        if record.get("dead_heat") or status in {"dead_heat", "deadheat", "unsupported_dead_heat"}:
            return "excluded_dead_heat", {}
        if status not in {"", "payable", "final", "settled"}:
            return "excluded_unsupported_settlement", {}
    if expected is None:
        return "excluded_incomplete_race", {}
    if not records:
        return "missing_settlement", {}
    dividends = {}
    for record in records:
        try:
            combination = _combination(record["winning_combination"], pool)
        except (KeyError, TypeError, ValueError):
            return "excluded_invalid_settlement", {}
        dividend = _positive(record.get("dividend_hkd_per_10"))
        if dividend is None:
            return "missing_dividend", {}
        if combination in dividends and dividends[combination] != dividend:
            return "excluded_conflicting_settlement", {}
        dividends[combination] = dividend
    if set(dividends) != expected:
        return "missing_settlement" if set(dividends) < expected else "excluded_invalid_settlement", {}
    return "settled", dividends


def _quote(records: list[dict], combination: tuple[str, ...], pool: str):
    matches = []
    for record in records:
        try:
            if _combination(record.get("combination"), pool) == combination:
                matches.append(record)
        except (TypeError, ValueError):
            continue
    if not matches:
        return None, "missing"
    values = []
    for record in matches:
        value = _positive(record.get("payout_hkd_per_10"))
        if (record.get("quote_scope") != "full_ticket"
                or record.get("is_pre_race") is not True or value is None
                or "winning_combination" in record
                or str(record.get("status", "")).lower() in {"final", "settled", "payable"}):
            return None, "invalid_provenance_or_value"
        values.append(value)
    if len(set(values)) != 1:
        return None, "conflicting"
    return values[0], "valid_full_ticket"


def _summary(ledger: pd.DataFrame) -> dict:
    settled = ledger[ledger["status"] == "settled"]
    stake = float(settled["stake_hkd"].sum())
    gross = float(settled["realized_gross_hkd"].sum())
    nsettled = len(settled)
    nmissing = int(ledger["status"].str.startswith("missing").sum())
    profit = gross - stake
    return {
        "stake_hkd": stake, "gross_hkd": gross, "profit_hkd": profit,
        "roi": profit / stake if stake else None,
        "hit_rate": float(settled["hit"].sum()) / nsettled if nsettled else None,
        "nraces": int(ledger["race_id"].nunique()), "nsettled": nsettled,
        "nmissing": nmissing, "nexcluded": len(ledger) - nsettled - nmissing,
        "planned_stake_hkd": float(ledger["stake_hkd"].sum()),
    }


def evaluate_season(
    frame: pd.DataFrame,
    settlements: Mapping | Iterable[Mapping],
    *,
    pools: Iterable[str] = SUPPORTED_POOLS,
    top_n: int = 1,
    quotes: Mapping | Iterable[Mapping] | None = None,
    exponents: OrderExponents = OrderExponents(),
) -> SeasonEvaluation:
    """Evaluate frozen top-probability tickets; see module docstring for schema.

    Invalid runner IDs, duplicate runners, dates, or probabilities raise ValueError.
    Invalid/missing official outcomes exclude rather than silently drop starters.
    Inputs are copied; no caller frames or records are mutated.
    """
    required = {"race_id", "date", "horse_no", "result", "model_probability"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(frame.columns))}")
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
        raise ValueError("top_n must be a positive integer")
    if _positive(exponents.second) is None or _positive(exponents.third) is None:
        raise ValueError("Order exponents must be positive and finite")
    pools = tuple(canonical_pool_name(pool) for pool in pools)
    if len(set(pools)) != len(pools):
        raise ValueError("Duplicate pools")
    data = frame.copy(deep=True)
    if data["race_id"].isna().any():
        raise ValueError("Missing race ID")
    data["race_id"] = data["race_id"].astype(str)
    data["horse_no"] = data["horse_no"].map(_runner)
    data["date"] = pd.to_datetime(data["date"], errors="coerce", utc=True)
    if data["date"].isna().any():
        raise ValueError("Invalid race date")
    data["model_probability"] = pd.to_numeric(data["model_probability"], errors="coerce")
    probabilities = data["model_probability"].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Probabilities must be finite and nonnegative")
    official, quoted = _records(settlements), _records(quotes)
    rows = []
    for race_id, race in data.groupby("race_id", sort=False):
        if race["horse_no"].duplicated().any():
            raise ValueError(f"Duplicate runners in {race_id}")
        if race["date"].nunique() != 1:
            raise ValueError(f"Multiple dates for {race_id}")
        race = race.iloc[np.argsort(race["horse_no"].astype(int).to_numpy())].copy()
        strengths = race["model_probability"].to_numpy(dtype=float)
        total = float(strengths.sum())
        if not np.isfinite(total) or total <= 0:
            raise ValueError(f"Race {race_id} has no positive probability mass")
        strengths = strengths / total
        # Freeze all selections before reading results or settlement dividends.
        rankings = rank_pool_combinations(
            race["horse_no"].tolist(), strengths,
            [pool for pool in pools if len(race) >= POOL_SIZE[pool]], exponents,
        )
        selected = {pool: tickets[:top_n] for pool, tickets in rankings.items()}
        for pool, tickets in selected.items():
            outcome_status = pool_result_status(race, pool)
            official_race = race.assign(result=pd.to_numeric(race["result"], errors="coerce"))
            status, dividends = _settlement(
                official.get(race_id, {}).get(pool, []),
                _winners(official_race, pool) if outcome_status == "eligible" else None, pool,
            )
            if status == "excluded_incomplete_race":
                status = outcome_status
            for rank, ticket in enumerate(tickets, 1):
                payout, quote_status = _quote(quoted.get(race_id, {}).get(pool, []),
                                             ticket.runners, pool)
                hit = ticket.runners in dividends if status == "settled" else None
                gross = dividends.get(ticket.runners, 0.0) if status == "settled" else None
                p = ticket.probability
                rows.append({
                    "race_id": race_id, "date": race["date"].iloc[0], "pool": pool,
                    "rank": rank, "combination": "/".join(ticket.runners), "stake_hkd": STAKE_HKD,
                    "probability": p,
                    "break_even_dividend_hkd_per_10": STAKE_HKD / p if p > 0 else float("inf"),
                    "payout_quote_hkd_per_10": payout, "quote_status": quote_status,
                    "estimated_ev_hkd": p * payout - STAKE_HKD if payout is not None else None,
                    "status": status, "hit": hit, "realized_gross_hkd": gross,
                    "realized_net_hkd": gross - STAKE_HKD if gross is not None else None,
                })
    ledger = pd.DataFrame(rows, columns=LEDGER_COLUMNS)
    if not ledger.empty:
        dates = data.groupby("race_id")["date"].first().to_dict()
        race_order = {race_id: index for index, race_id in enumerate(sorted(
            dates, key=lambda race_id: (dates[race_id], _natural_order(race_id)),
        ))}
        ledger["_race_order"] = ledger["race_id"].map(race_order)
        ledger = ledger.sort_values(["_race_order", "pool", "rank"], kind="stable").drop(
            columns="_race_order",
        ).reset_index(drop=True)
    summary = _summary(ledger)
    by_pool = {pool: _summary(ledger[ledger["pool"] == pool]) for pool in pools}
    settled = ledger[ledger["status"] == "settled"]
    curve = settled.groupby(["date", "race_id"], sort=False)[
        ["stake_hkd", "realized_gross_hkd", "realized_net_hkd"]
    ].sum().reset_index()
    curve["bankroll_hkd"] = STARTING_BANKROLL_HKD + curve["realized_net_hkd"].cumsum()
    prior_profit = curve["realized_net_hkd"].cumsum() - curve["realized_net_hkd"]
    capital_required = max(0.0, float((curve["stake_hkd"] - prior_profit).max())) if len(curve) else 0.0
    curve["capital_required_hkd"] = (curve["stake_hkd"] - prior_profit).clip(lower=0).cummax()
    curve["financed_by_starting_1000"] = curve["capital_required_hkd"] <= STARTING_BANKROLL_HKD
    peaks = curve["bankroll_hkd"].cummax().clip(lower=STARTING_BANKROLL_HKD)
    drawdown = float((peaks - curve["bankroll_hkd"]).max()) if len(curve) else 0.0
    summary.update({
        "starting_bankroll_hkd": STARTING_BANKROLL_HKD,
        "ending_bankroll_hkd": STARTING_BANKROLL_HKD + summary["profit_hkd"],
        "hypothetical_fixed_stake_return": summary["profit_hkd"] / STARTING_BANKROLL_HKD,
        "min_bankroll_capital_required_hkd": capital_required,
        "maximum_drawdown_hkd": drawdown,
        "can_finance_with_1000": capital_required <= STARTING_BANKROLL_HKD,
        "additional_capital_required_hkd": max(0.0, capital_required - STARTING_BANKROLL_HKD),
        "bankroll_assumption": "Hypothetical settled-only fixed stakes; unfunded schedule requires additional capital; no compounding or Kelly",
    })
    return SeasonEvaluation(ledger, summary, by_pool, curve)
