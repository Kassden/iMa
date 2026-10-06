"""Fractional Kelly and cash-constrained joint-scenario paper portfolios."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR
import math
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import linprog, minimize

from ima.betting_contracts import Quote, RuleInputs, Ticket, money
from ima.betting_ev import EVReport, JointScenarios, joint_return_matrix, ticket_probabilities
from ima.betting_settlement import ticket_wins


@dataclass(frozen=True)
class StakePolicy:
    bankroll: Decimal
    currency: str
    kelly_fraction: float = 0.25
    max_race_fraction: float = 0.1
    max_ticket_fraction: float = 0.05
    minimum_cash_fraction: float = 0.5
    remaining_day_budget: Decimal | None = None
    maximum_open_exposure: Decimal | None = None
    current_open_exposure: Decimal = Decimal(0)
    payout_multiplier: float = 1.0
    uncertainty_policy: str = "Provided coherent joint scenarios; no independent marginal haircut"
    own_impact_policy: str = "negligible"

    def __post_init__(self):
        for name in ("bankroll", "current_open_exposure", "remaining_day_budget", "maximum_open_exposure"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, money(value))
        if self.bankroll <= 0 or not math.isfinite(float(self.bankroll)) or len(self.currency) != 3:
            raise ValueError("Positive bankroll and ISO currency required")
        for name in ("kelly_fraction", "max_race_fraction", "max_ticket_fraction",
                     "minimum_cash_fraction", "payout_multiplier"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Policy fractions must be finite in [0, 1]")
        if self.own_impact_policy != "negligible":
            raise ValueError("Stake-dependent pool impact needs observed turnover and a separate model")
        if not self.uncertainty_policy:
            raise ValueError("Uncertainty assumption required")

    @property
    def cap_amount(self) -> Decimal:
        cap = min(Decimal(str(self.max_race_fraction)) * self.bankroll,
                  (1-Decimal(str(self.minimum_cash_fraction))) * self.bankroll,
                  Decimal("0.9999999999") * self.bankroll)
        if self.remaining_day_budget is not None:
            cap = min(cap, self.remaining_day_budget)
        if self.maximum_open_exposure is not None:
            cap = min(cap, self.maximum_open_exposure-self.current_open_exposure)
        return max(Decimal(0), cap)

    @property
    def total_cap(self) -> float:
        return float(self.cap_amount / self.bankroll)


def physical_stake(fraction: float, policy: StakePolicy, rules: RuleInputs) -> Decimal:
    if not math.isfinite(fraction) or fraction < 0:
        raise ValueError("Invalid stake fraction")
    amount = min(Decimal(str(fraction)) * policy.bankroll, policy.cap_amount,
                 Decimal(str(policy.max_ticket_fraction)) * policy.bankroll)
    rounded = (amount / rules.stake_increment).to_integral_value(rounding=ROUND_FLOOR) * rules.stake_increment
    return Decimal(0) if rounded < rules.minimum_stake else rounded


def full_kelly_fraction(probability: float, decimal_return: float) -> float:
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Probability must be in [0, 1]")
    if not math.isfinite(decimal_return) or decimal_return <= 1:
        raise ValueError("Kelly requires total return d > 1")
    return max(0.0, min(1.0, (probability * decimal_return - 1) / (decimal_return - 1)))


def binary_log_growth(probability: float, decimal_return: float, fraction: float) -> float:
    if not 0 <= fraction < 1:
        raise ValueError("Cash must remain positive")
    return (probability * math.log1p(fraction * (decimal_return - 1))
            + (1 - probability) * math.log1p(-fraction))


@dataclass(frozen=True)
class StakeResult:
    tickets: tuple[Ticket, ...]
    stakes: tuple[Decimal, ...]
    full_fractions: tuple[float, ...]
    rounded_fractions: tuple[float, ...]
    expected_log_growth: float
    expected_profit: float
    success: bool
    status: str
    solver_message: str
    constraint_residual: float
    all_lose_probability: float
    scenario_source: str
    assumptions: str
    quote_sources: tuple[str, ...]
    optimality_gap: float | None = 0.0
    paper_only: bool = True


def _validate_reports(reports: Sequence[EVReport], policy: StakePolicy, rules: RuleInputs):
    if policy.currency != rules.currency:
        raise ValueError("Bankroll/rule currency mismatch")
    if not rules.verified:
        raise ValueError("Sizing requires verified effective rules")
    if not reports or len({r.ticket for r in reports}) != len(reports):
        raise ValueError("Empty or duplicate tickets; consolidate identical physical exposures first")
    if len({r.ticket.race_id for r in reports}) != 1:
        raise ValueError("Portfolio must describe one joint race")
    for report in reports:
        if (not report.executable_evidence or report.mode != "quoted"
                or report.decimal_return is None or report.decimal_return <= 1
                or not math.isfinite(report.decimal_return)
                or report.currency != policy.currency or report.rule_version != rules.version):
            raise ValueError("Sizing requires eligible pre-race quotes and verified effective rules")
        metadata = report.probability_metadata
        if metadata.get("draws") is not None and (not metadata.get("converged") or report.probability >= 1):
            raise ValueError("Sampled single-ticket sizing requires convergence evidence and observed losing tails")


def size_single(report: EVReport, policy: StakePolicy, rules: RuleInputs) -> StakeResult:
    _validate_reports((report,), policy, rules)
    d = report.decimal_return * policy.payout_multiplier
    full = full_kelly_fraction(report.probability, d)
    fraction = min(full * policy.kelly_fraction, policy.total_cap, policy.max_ticket_fraction)
    stake = physical_stake(fraction, policy, rules)
    rounded = float(stake / policy.bankroll)
    return StakeResult((report.ticket,), (stake,), (full,), (rounded,),
        binary_log_growth(report.probability, d, rounded),
        float(stake) * (report.probability * d - 1), True, "optimal", "analytic binary Kelly", 0,
        1 - report.probability, "binary", policy.uncertainty_policy + "; negligible own impact",
        (report.quote_source,))


def size_portfolio(reports: Sequence[EVReport], scenarios: JointScenarios,
                   gross_returns: np.ndarray, policy: StakePolicy, rules: RuleInputs,
                   *, max_iterations: int = 1000, tolerance: float = 1e-10,
                   scenario_dividends: Sequence[Mapping[Ticket, Quote]] | None = None) -> StakeResult:
    """Maximize E[log(1 + sum f_j (d_sj - 1))] including cash.

    All tickets share the same outcome rows. Callers supplying dead-heat/refund
    returns must use the settlement adapter; zero-mass tails do not bound loss.
    Fractional Kelly scales the constrained continuous solution, then rounds down.
    """
    reports = tuple(reports)
    _validate_reports(reports, policy, rules)
    p = np.asarray(scenarios.probabilities, dtype=float)
    gross = np.asarray(gross_returns, dtype=float)
    n = len(reports)
    if (gross.shape != (len(p), n) or not np.isfinite(gross).all() or (gross < 0).any()
            or max_iterations < 1 or tolerance <= 0 or not math.isfinite(tolerance)):
        raise ValueError("Invalid joint returns or solver controls")
    implied = ticket_probabilities([r.ticket for r in reports], scenarios, rules)
    if any(not math.isclose(r.probability, probability, abs_tol=1e-10, rel_tol=0)
           for r, probability in zip(reports, implied)):
        raise ValueError("Quote-report probabilities disagree with the joint outcome distribution")
    expected = joint_return_matrix([r.ticket for r in reports], scenarios, rules,
                                   [r.decimal_return for r in reports],
                                   scenario_dividends=scenario_dividends)
    if not np.allclose(gross, expected, rtol=1e-12, atol=1e-12):
        raise ValueError("Return matrix differs from normalized quote/scenario dividend units")
    for i, finish in enumerate(scenarios.finishes):
        for j, report in enumerate(reports):
            refunded = finish.abandoned or bool(set(report.ticket.runners) & set(finish.scratches))
            if refunded:
                if rules.scratch_policy != "refund_selected" or gross[i, j] != 1:
                    raise ValueError("Joint return matrix violates refund rules")
            elif not ticket_wins(report.ticket, finish, rules) and gross[i, j] != 0:
                raise ValueError("Joint return matrix pays a losing ticket")
    # Stress paid dividends only. Full refunds (d=1) remain full refunds.
    stressed = np.where(gross == 1, 1, gross * policy.payout_multiplier)
    net = stressed - 1
    all_lose = float(p[np.all(gross == 0, axis=1)].sum())
    if scenarios.metadata.get("draws") is not None:
        if not scenarios.metadata.get("converged") or all_lose <= 0:
            raise ValueError("Sampled sizing requires convergence evidence and observed losing tails")
    cap = policy.total_cap
    bound = min(policy.max_ticket_fraction, cap)

    def objective(f):
        wealth = 1 + net @ f
        if (wealth <= 0).any():
            return float("inf")
        return -float(p @ np.log(wealth))

    def gradient(f):
        return -(net.T @ (p / (1 + net @ f)))

    result = minimize(objective, np.zeros(n), jac=gradient, method="SLSQP",
        bounds=[(0, bound)] * n,
        constraints=[{"type": "ineq", "fun": lambda f: cap - f.sum(),
                      "jac": lambda f: -np.ones(n)}],
        options={"maxiter": max_iterations, "ftol": min(tolerance, 1e-13)})
    f = np.asarray(result.x)
    residual = max(0.0, float(-f.min()), float(f.max() - bound), float(f.sum() - cap))
    valid = (result.success and np.isfinite(f).all() and residual <= 1e-8
             and np.all(1 + net @ f > 0) and math.isfinite(objective(f))
             and objective(f) <= tolerance)
    gap = float("inf")
    if valid:
        # Independent first-order concave-objective bound over the feasible box.
        g = -gradient(f)
        check = linprog(-g, A_ub=np.ones((1, n)), b_ub=[cap], bounds=[(0, bound)] * n,
                        method="highs")
        if check.success:
            gap = max(0.0, float(g @ (check.x - f)))
        valid = check.success and gap <= max(1e-7, tolerance * 100)
    if valid:
        # Remove tiny constraint overshoots before physical rounding.
        f = np.clip(f, 0, bound)
        if f.sum() > cap:
            f *= cap / f.sum()
        stakes = tuple(physical_stake(float(x * policy.kelly_fraction), policy, rules) for x in f)
        if sum(stakes) > policy.cap_amount:
            # Decimal readback of numerical cash caps is stricter than float tolerance.
            amounts = list(stakes)
            for j in reversed(range(n)):
                excess = sum(amounts) - policy.cap_amount
                if excess <= 0:
                    break
                units = (excess/rules.stake_increment).to_integral_value(rounding="ROUND_CEILING")
                amounts[j] = max(Decimal(0), amounts[j] - units*rules.stake_increment)
                if amounts[j] < rules.minimum_stake:
                    amounts[j] = Decimal(0)
            stakes = tuple(amounts)
        rounded = np.array([float(s / policy.bankroll) for s in stakes])
        growth = -objective(rounded)
        # Independent flooring changes hedge ratios. Cash beats a negative-growth rounded portfolio.
        if growth < 0:
            stakes, rounded, growth = (Decimal(0),) * n, np.zeros(n), 0.0
        status = "optimal" if any(stakes) else "cash"
    else:
        stakes, rounded, growth, status = (Decimal(0),) * n, np.zeros(n), 0.0, "solver_failed"
        if not np.isfinite(f).all():
            f = np.zeros(n)
        if not math.isfinite(residual):
            residual = 1.0
        if not math.isfinite(gap):
            gap = None
    message = str(result.message)
    if result.success and not valid:
        message += "; independent objective/constraint/optimality validation failed"
    return StakeResult(tuple(r.ticket for r in reports), stakes, tuple(float(x) for x in f),
        tuple(float(x) for x in rounded), float(growth),
        float(policy.bankroll) * float(p @ (net @ rounded)), bool(valid), status,
        message, residual, all_lose, scenarios.source_id,
        scenarios.assumptions + "; " + policy.uncertainty_policy + "; negligible own impact",
        tuple(r.quote_source for r in reports), gap)
