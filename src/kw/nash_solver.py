"""Algorithm 1: damped best-response iteration for stationary Nash equilibria.

Alternates (architecture.md §6.1) between (1) the population law's fixed
point `mu` under the current policy (`distributions.invariant`), and (2) a
per-type Bellman best response (`bellman.rvi_solve` / `greedy_policy`,
selecting the incumbent entry at indifference via `current=`)
compared against the current policy's own gain (`bellman.policy_eval`) to
form a gain gap. The policy is updated by a damped convex combination toward
the best response; the loop stops once the worst-case gain gap is at most
`eps_br` and the population law's one-step residual at the fixed point is at
most `eps_dist`.
"""

from dataclasses import dataclass

import numpy as np

from kw.bellman import greedy_policy, policy_eval, rvi_solve
from kw.distributions import aggregate, default_mu0, invariant, one_step, reciprocation
from kw.economies import Economy


@dataclass
class NashResult:
    alpha: np.ndarray
    beta: np.ndarray
    mu: np.ndarray
    Q: np.ndarray
    g: np.ndarray
    gain_gap_history: list[float]
    dist_history: list[float]
    converged: bool


def _gains(econ: Economy, alpha: np.ndarray, beta: np.ndarray, mu: np.ndarray) -> np.ndarray:
    agg = aggregate(econ, mu)
    Q = reciprocation(econ, mu, alpha)
    g = np.zeros(econ.n_types, dtype=np.float64)
    for i in range(econ.n_types):
        g[i], _ = policy_eval(econ, i, alpha[i], beta[i], agg, Q)
    return g


def solve_nash(
    econ: Economy,
    alpha0: np.ndarray,
    beta0: np.ndarray,
    mu0: np.ndarray | None = None,
    rho: float = 0.3,
    eps_br: float = 1e-9,
    eps_dist: float = 1e-12,
    max_iter: int = 2000,
) -> NashResult:
    alpha, beta = alpha0.copy(), beta0.copy()
    mu = default_mu0(econ) if mu0 is None else mu0.copy()
    gaps: list[float] = []
    dists: list[float] = []
    for _ in range(max_iter):
        mu = invariant(econ, mu, alpha, beta)
        agg = aggregate(econ, mu)
        Q = reciprocation(econ, mu, alpha)
        alpha_br, beta_br = np.zeros_like(alpha), np.zeros_like(beta)
        gap = 0.0
        for i in range(econ.n_types):
            g_star, V, W = rvi_solve(econ, i, agg, Q)
            alpha_br[i], beta_br[i] = greedy_policy(
                econ, i, Q, V, W, current=(alpha[i], beta[i])
            )
            g_pol, _ = policy_eval(econ, i, alpha[i], beta[i], agg, Q)
            gap = max(gap, g_star - g_pol)
        dist = np.abs(one_step(econ, mu, alpha, beta) - mu).max()
        gaps.append(gap)
        dists.append(dist)
        if gap <= eps_br and dist <= eps_dist:
            return NashResult(alpha, beta, mu, Q, _gains(econ, alpha, beta, mu), gaps, dists, True)
        alpha = (1 - rho) * alpha + rho * alpha_br
        beta = (1 - rho) * beta + rho * beta_br
    return NashResult(alpha, beta, mu, Q, _gains(econ, alpha, beta, mu), gaps, dists, False)
