"""Coherent complete finish orders; iid chunked simulation with error evidence."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from itertools import permutations

import numpy as np
from scipy.stats import beta

from .performance_distributions import PerformanceDistribution


@dataclass
class RankDistribution:
    runner_ids: tuple[str, ...]
    order_probabilities: dict[tuple[str, ...], float]
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if len(set(self.runner_ids)) != len(self.runner_ids) or not self.runner_ids:
            raise ValueError("Runner IDs must be nonempty and unique")
        if any(set(order) != set(self.runner_ids) or len(order) != len(self.runner_ids)
               for order in self.order_probabilities):
            raise ValueError("Each outcome must contain the complete field")
        probabilities = np.asarray(list(self.order_probabilities.values()))
        if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1, atol=1e-12):
            raise ValueError("Finish-order probabilities must normalize")

    def probability(self, *, first=None, top_k=None, unordered=None):
        return float(sum(p for order, p in self.order_probabilities.items()
            if (first is None or order[0] == str(first))
            and (unordered is None or set(order[:len(unordered)]) == set(map(str, unordered)))
            and (top_k is None or str(top_k[0]) in order[:int(top_k[1])])) )

    def win_probabilities(self):
        return np.array([self.probability(first=runner) for runner in self.runner_ids])

    def place_probability(self, runner_id, places=3):
        if not 1 <= places <= len(self.runner_ids):
            raise ValueError("Invalid place count")
        return self.probability(top_k=(runner_id, places))

    def quinella_probability(self, runners):
        if len(set(runners)) != 2:
            raise ValueError("Quinella requires two distinct runners")
        return self.probability(unordered=runners)

    def trio_probability(self, runners):
        if len(set(runners)) != 3:
            raise ValueError("Trio requires three distinct runners")
        return self.probability(unordered=runners)

    def order_probability(self, order):
        order = tuple(map(str, order))
        return float(sum(p for outcome, p in self.order_probabilities.items() if outcome[:len(order)] == order))

    def probability_interval(self, probability, confidence=0.95):
        if not 0 <= probability <= 1 or not 0 < confidence < 1:
            raise ValueError("Invalid interval probability/confidence")
        n = self.metadata.get("draws")
        if n is None:
            return float(probability), float(probability)
        count = int(round(probability*n))
        alpha = 1-confidence
        lower = 0.0 if count == 0 else beta.ppf(alpha/2, count, n-count+1)
        upper = 1.0 if count == n else beta.ppf(1-alpha/2, count+1, n-count)
        return float(lower), float(upper)


def plackett_luce_distribution(probabilities, runner_ids, *, max_runners=8):
    """Exact small-field enumeration, explicitly inferred from win marginals."""
    p = np.asarray(probabilities, float)
    runners = tuple(map(str, runner_ids))
    if p.shape != (len(runners),) or len(p) > max_runners or not np.isfinite(p).all() or np.any(p < 0) or p.sum() <= 0:
        raise ValueError("Invalid or excessively large exact Plackett-Luce field")
    p = p/p.sum()
    if np.any(p == 0):
        raise ValueError("Zero strengths cannot identify complete PL orders; declare a smoothing policy")
    outcomes = {}
    for order in permutations(range(len(p))):
        value = 1.0
        for k, i in enumerate(order[:-1]):
            value *= p[i]/p[list(order[k:])].sum()
        outcomes[tuple(runners[i] for i in order)] = float(value)
    return RankDistribution(runners, outcomes, {"method": "exact_plackett_luce",
        "assumption": "sequential proportional strengths; joint orders not identified by marginals"})


def simulate_rank_distribution(distribution: PerformanceDistribution, runner_ids, *, seed=42,
                               draws=65536, chunk_size=4096, covariance=None):
    runners = tuple(map(str, runner_ids))
    n = len(runners)
    if n != len(distribution.location) or len(set(runners)) != n or draws < 2 or chunk_size < 1:
        raise ValueError("Invalid simulation dimensions or budget")
    factor = None
    if covariance is not None:
        covariance = np.asarray(covariance, float)
        if covariance.shape != (n, n) or not np.isfinite(covariance).all() or not np.allclose(covariance, covariance.T):
            raise ValueError("Invalid covariance")
        eigenvalues, vectors = np.linalg.eigh(covariance)
        if eigenvalues.min() < -1e-10 or not np.allclose(np.diag(covariance), 1):
            raise ValueError("Correlation matrix must be positive semidefinite with diagonal one")
        factor = vectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0)))
    random = np.random.default_rng(seed)

    def sample(count):
        z = random.standard_normal((count, n))
        if factor is not None:
            z = z @ factor.T
        performances = distribution.location + distribution.scale*z
        return -performances if distribution.direction == "higher" else performances

    counts = _sample_order_counts(runners, draws, chunk_size, sample)
    return RankDistribution(runners, {order: count/draws for order, count in counts.items()},
        {"method": "iid_gaussian", "seed": int(seed), "draws": draws,
         "chunk_size": chunk_size, "rng": "numpy.PCG64", "numpy_version": np.__version__,
         "coordinate": distribution.coordinate, "dependence": "independent" if covariance is None else "declared_correlation",
         "correlation": None if covariance is None else covariance.tolist(),
         "converged": False, "convergence_status": "fixed_budget_only; no independent replicate check",
         "unseen_outcomes": "zero count is not zero probability; use probability_interval",
         "dead_heats": "continuous baseline excludes ties; settlement rules external"})


def _sample_order_counts(runners, draws, chunk_size, sample):
    counts = Counter()
    for start in range(0, draws, chunk_size):
        orders = np.argsort(sample(min(chunk_size, draws-start)), axis=1, kind="stable")
        counts.update(tuple(runners[i] for i in order) for order in orders)
    return counts


def sample_plackett_luce_distribution(probabilities, runner_ids, *, seed=42,
                                     draws=65536, chunk_size=4096):
    """Same chunked order engine; Gumbel strengths induce Plackett-Luce orders."""
    p = np.asarray(probabilities, float)
    runners = tuple(map(str, runner_ids))
    if p.shape != (len(runners),) or not len(p) or len(set(runners)) != len(runners) or not np.isfinite(p).all() or np.any(p <= 0) or draws < 2 or chunk_size < 1:
        raise ValueError("Sampled Plackett-Luce needs positive strengths and valid budget")
    p = p/p.sum()
    random = np.random.default_rng(seed)
    counts = _sample_order_counts(runners, draws, chunk_size,
        lambda count: -(np.log(p)+random.gumbel(size=(count, len(p)))))
    return RankDistribution(runners, {order: count/draws for order, count in counts.items()},
        {"method": "iid_plackett_luce", "seed": seed, "draws": draws, "chunk_size": chunk_size,
         "rng": "numpy.PCG64", "numpy_version": np.__version__,
         "assumption": "sequential proportional strengths; joint orders not identified by marginals",
         "unseen_outcomes": "zero count is not zero probability; use probability_interval",
         "converged": False, "convergence_status": "fixed_budget_only; no independent replicate check"})


def converged_rank_distribution(distribution, runner_ids, *, seed=42, tolerance=0.01,
                                min_draws=8192, max_draws=131072, chunk_size=4096):
    """Independent iid replicates; report unmet tolerance at the resource limit."""
    if tolerance <= 0 or min_draws < 2 or max_draws < min_draws:
        raise ValueError("Invalid convergence budget")
    draws = min_draws
    while True:
        a = simulate_rank_distribution(distribution, runner_ids, seed=seed, draws=draws, chunk_size=chunk_size)
        b = simulate_rank_distribution(distribution, runner_ids, seed=seed+1, draws=draws, chunk_size=chunk_size)
        keys = set(a.order_probabilities) | set(b.order_probabilities)
        difference = max(abs(a.order_probabilities.get(k, 0)-b.order_probabilities.get(k, 0)) for k in keys)
        # Simultaneous Hoeffding bound covers all complete orders, including unobserved ones.
        from scipy.special import gammaln
        bound = float(np.sqrt((np.log(40)+gammaln(len(a.runner_ids)+1))/(2*draws)))
        ok = max(difference, bound) <= tolerance
        if ok or draws >= max_draws:
            a.metadata.update({"independent_seed": seed+1, "replicate_max_difference": difference,
                "simultaneous_error_bound_95": bound, "tolerance": tolerance, "converged": ok})
            return a
        draws = min(2*draws, max_draws)
