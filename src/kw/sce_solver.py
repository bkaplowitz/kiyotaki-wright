"""Algorithm 2: restricted-information self-confirming equilibrium (SCE) fixed point.

A genuinely different fixed point from Algorithm 1 (architecture.md §6.2, paper
Algorithm 2, pp. 26-28). State is (sigma_m, theta_m, mu_m) with subjective
environments theta_i = (m_i, q_i) in place of the objective (mu, Q). Per
iteration:

1. Per type, solve the *subjective* Bellman equations under theta_i
   (`bellman.rvi_solve_subjective`) and damp the policy toward the subjective
   greedy best response (`bellman.greedy_policy`, passing q_i as its `Q`
   argument and `current=(alpha_i, beta_i)` for the minimal-movement
   Delta(argmax) tie-break — the mechanism that makes a skeptical q_i == 0
   self-confirming: every offer margin is exactly tied at 0, so the tie-break
   keeps whatever policy was seeded there, never contradicted by "experience"
   the type never gathers, paper p. 44-45, p. 51, p. 54).
2. Recompute the objective invariant distribution mu and reciprocation Q under
   the newly damped policy (`distributions.invariant`/`reciprocation`).
3. Belief update: m_i <- mu unconditionally for every type (a meeting always
   reveals the partner's good, whether or not a trade is offered — paper
   Algorithm 2 lines 14-16); q_i(k,j) <- Q(k,j) only at on-path triples
   (i,k,j) with mu_i(k)*mu(j)*alpha_i(k,j) > 1e-14 (a reciprocal-acceptance
   rate is only observed when the type actually offers — lines 17-23);
   off-path q_i is left frozen. Neither m nor q is damped (paper Algorithm 2
   sets them outright, unlike the policy).

Residuals, evaluated each iteration at the pre-update state (sigma_m, theta_m,
mu_m), mirror the paper's s_m/b_m/d_m (§6.2, eq. after Algorithm 2 line 8):
s_m is the worst-case subjective optimality gap (subjective RVI gain minus
the subjective policy_eval gain of the current policy); b_m is the worst-case
on-path belief error |q_i - Q|; d_m is the population law's one-step residual
at mu_m. The solver stops when max(s_m, b_m) <= eps_sce and d_m <= eps_dist.
"""

from dataclasses import dataclass

import numpy as np

from kw.bellman import greedy_policy, policy_eval_subjective, rvi_solve_subjective
from kw.distributions import aggregate, default_mu0, invariant, one_step, reciprocation
from kw.economies import Economy


@dataclass
class SCEResult:
    alpha: np.ndarray
    beta: np.ndarray
    mu: np.ndarray
    m: np.ndarray
    q: np.ndarray
    s_history: list[float]
    b_history: list[float]
    d_history: list[float]
    converged: bool


_MU_TOL = 1e-15
"""invariant() tolerance inside the SCE loop, tighter than the 1e-12 default.

The exact fixed points here have hard zeros in mu (e.g. Economy C's type 2
holds no fiat at the frozen low-trade point), but forward iteration only
approaches them along a geometric tail, terminating with dust of order
tol / (per-period outflow rate) — at the default 1e-12 and C's slowest
rate ~0.013 that is ~7e-11. The dust leaks into Q via the reciprocation
kernel, gets learned as an on-path belief q_i(k,j) ~ 7e-11, and turns a
type's belief-chain into one whose only escape from a state is that dust
arc; RVI's Bellman residual then plateaus at ~dust * gain-gap ~ 7e-10,
above its 1e-10 tolerance, and fails loud. Tightening to 1e-15 puts the
plateau near 7e-13 (dust ~7e-14) — two orders below the RVI tolerance —
for ~2,300 extra forward iterations per invariant call."""


def _onpath_mask(mu: np.ndarray, agg: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """O_m = {(i,k,j): mu_i(k) mu(j) alpha_i(k,j) > 1e-14} (paper Algorithm 2, O_m)."""
    return mu[:, :, None] * agg[None, None, :] * alpha > 1e-14


def solve_sce(
    econ: Economy,
    alpha0: np.ndarray,
    beta0: np.ndarray,
    m0: np.ndarray,
    q0: np.ndarray,
    rho: float = 0.3,
    eps_sce: float = 1e-9,
    eps_dist: float = 1e-12,
    max_iter: int = 5000,
) -> SCEResult:
    alpha, beta = alpha0.copy(), beta0.copy()
    m, q = m0.copy(), q0.copy()
    s_hist: list[float] = []
    b_hist: list[float] = []
    d_hist: list[float] = []
    mu = invariant(econ, default_mu0(econ), alpha, beta, tol=_MU_TOL)
    agg = aggregate(econ, mu)
    Q = reciprocation(econ, mu, alpha)
    for _ in range(max_iter):
        alpha_br, beta_br = np.zeros_like(alpha), np.zeros_like(beta)
        s = 0.0
        for i in range(econ.n_types):
            g_star, V, W = rvi_solve_subjective(econ, i, m[i], q[i])
            alpha_br[i], beta_br[i] = greedy_policy(
                econ, i, q[i], V, W, current=(alpha[i], beta[i])
            )
            g_pol, _ = policy_eval_subjective(econ, i, alpha[i], beta[i], m[i], q[i])
            s = max(s, g_star - g_pol)

        onpath = _onpath_mask(mu, agg, alpha)
        b = float(np.abs(q - Q[None, :, :])[onpath].max()) if onpath.any() else 0.0
        d = float(np.abs(one_step(econ, mu, alpha, beta) - mu).max())
        s_hist.append(s)
        b_hist.append(b)
        d_hist.append(d)

        if s <= eps_sce and b <= eps_sce and d <= eps_dist:
            return SCEResult(alpha, beta, mu, m, q, s_hist, b_hist, d_hist, True)

        alpha = (1 - rho) * alpha + rho * alpha_br
        beta = (1 - rho) * beta + rho * beta_br

        mu = invariant(econ, mu, alpha, beta, tol=_MU_TOL)
        agg = aggregate(econ, mu)
        Q = reciprocation(econ, mu, alpha)
        m = np.broadcast_to(agg, (econ.n_types, econ.n_goods)).copy()
        onpath_new = _onpath_mask(mu, agg, alpha)
        q = np.where(onpath_new, Q[None, :, :], q)

    return SCEResult(alpha, beta, mu, m, q, s_hist, b_hist, d_hist, False)
