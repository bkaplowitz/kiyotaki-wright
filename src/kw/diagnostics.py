"""Outcome diagnostics: BR gain, SCE certificate, and outcome classification.

`br_gain` is the paper's diagnostic (architecture.md §6.2) reported alongside
every restricted-information SCE: it re-evaluates a policy profile under the
*objective* environment (rvi_solve/policy_eval, not the subjective variants
`solve_sce` uses internally) and reports how much any type could still gain
by deviating, given full information about (mu, Q). A stationary Nash
equilibrium has br_gain == 0; a self-confirming equilibrium that is not Nash
has br_gain > 0 for at least one type (paper §7's H2 claim).

`stationarity` is the population law's one-step residual at mu, ||F_sigma(mu)
- mu||_inf — the same sup-norm metric sce_solver.py's d_history and
nash_solver.py's dist_history track, so a value <= Thresholds.dist carries
the same convergence semantics as those solvers' eps_dist stopping rule.

`sce_certificate` is the approximate-SCE analogue of `br_gain`: given beliefs
theta_hat = (m_hat, q_hat) (e.g. a learner's own, or an SCE solver's
converged beliefs), it reports the worst-case subjective optimality gap s
(rvi_solve_subjective vs policy_eval_subjective under theta_hat, mirroring
sce_solver.py's s_m) and the worst-case on-path belief error b_belief
(|q_hat - Q^sigma| at the objective reciprocation kernel implied by mu,
mirroring sce_solver.py's b_m) — both evaluated once at the caller's
(alpha, beta, mu, theta_hat) rather than iterated to a fixed point.

`classify_run` combines the three into architecture.md §8's outcome
classifier: NASH (b <= nash_gain and d <= dist), SCE_TRAP (a Nash gap is
present but mu is stationary and (alpha, beta) is an approximate SCE under
its own beliefs), or UNSETTLED otherwise.
"""

from dataclasses import dataclass

import numpy as np

from kw.bellman import policy_eval, policy_eval_subjective, rvi_solve, rvi_solve_subjective
from kw.distributions import aggregate, default_mu0, invariant, one_step, reciprocation
from kw.economies import Economy


def br_gain(
    econ: Economy,
    alpha: np.ndarray,
    beta: np.ndarray,
    mu: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Complete-information best-response gain of policy (alpha, beta) at mu.

    If `mu` is not given, uses the invariant distribution of (alpha, beta)
    from the default f-endowment start. Per type: the objective optimal gain
    (`rvi_solve`) minus the gain of the given policy (`policy_eval`), both
    under the objective (agg, Q) implied by mu. Returns (max_i gap, per-type
    gap array); gap == 0 (up to solver tolerance) means the policy is already
    a complete-information best response at mu, i.e. Nash.
    """
    if mu is None:
        mu = invariant(econ, default_mu0(econ), alpha, beta)
    agg = aggregate(econ, mu)
    Q = reciprocation(econ, mu, alpha)
    gaps = np.zeros(econ.n_types, dtype=np.float64)
    for i in range(econ.n_types):
        g_star, _, _ = rvi_solve(econ, i, agg, Q)
        g_pol, _ = policy_eval(econ, i, alpha[i], beta[i], agg, Q)
        gaps[i] = g_star - g_pol
    return float(gaps.max()), gaps


def stationarity(econ: Economy, alpha: np.ndarray, beta: np.ndarray, mu: np.ndarray) -> float:
    """Population law's one-step residual d = ||F_sigma(mu) - mu||_inf at mu.

    Sup-norm over all (type, good) entries, identical to the metric
    sce_solver.py's d_history and nash_solver.py's dist_history compute each
    iteration — so this function's value against a `Thresholds.dist` cutoff
    carries the same convergence semantics as those solvers' eps_dist rule.
    """
    return float(np.abs(one_step(econ, mu, alpha, beta) - mu).max())


def tv_distance(mu_a: np.ndarray, mu_b: np.ndarray, nu: np.ndarray) -> float:
    """Aggregate total variation distance between two (I,G) type-level laws.

    nu (I,) are type-share weights (`econ.nu`); nu @ mu_a and nu @ mu_b are
    the aggregate good-level marginals (distributions.aggregate's formula)
    implied by mu_a and mu_b, and TV is half the L1 distance between them.
    """
    return float(0.5 * np.abs(nu @ mu_a - nu @ mu_b).sum())


def sce_certificate(
    econ: Economy,
    alpha: np.ndarray,
    beta: np.ndarray,
    m_hat: np.ndarray,
    q_hat: np.ndarray,
    mu: np.ndarray | None = None,
) -> tuple[float, float]:
    """Approximate-SCE certificate of (alpha, beta) under beliefs (m_hat, q_hat).

    m_hat is (I,G); q_hat is (I,G,G) per-type beliefs (as `SCEResult.q`
    provides) or (G,G) pooled beliefs, broadcast to per-type here.

    s = max_i [rvi_solve_subjective(m_hat_i, q_hat_i) gain minus
    policy_eval_subjective(alpha_i, beta_i, m_hat_i, q_hat_i) gain]: the
    worst-case subjective optimality gap of the policy under its own beliefs
    (mirrors sce_solver.py's s_m).

    b_belief = max over on-path (i,k,j) of |q_hat_i(k,j) - Q^sigma(k,j)|,
    where Q^sigma = reciprocation(econ, mu, alpha) is the objective
    reciprocation kernel implied by mu, and on-path is
    mu_i(k) * aggregate(mu)(j) * alpha_i(k,j) > 1e-14 (same predicate as
    sce_solver.py's private `_onpath_mask`, recomputed locally rather than
    imported so this module has no dependency on sce_solver's internals).
    0.0 if the on-path set is empty. If `mu` is not given, uses the
    invariant distribution of (alpha, beta) from the default f-endowment
    start, mirroring `br_gain`'s default.
    """
    if mu is None:
        mu = invariant(econ, default_mu0(econ), alpha, beta)
    if q_hat.ndim == 2:
        q_hat = np.broadcast_to(q_hat, (econ.n_types, econ.n_goods, econ.n_goods))

    s = 0.0
    for i in range(econ.n_types):
        g_star, _, _ = rvi_solve_subjective(econ, i, m_hat[i], q_hat[i])
        g_pol, _ = policy_eval_subjective(econ, i, alpha[i], beta[i], m_hat[i], q_hat[i])
        s = max(s, g_star - g_pol)

    agg = aggregate(econ, mu)
    Q_sigma = reciprocation(econ, mu, alpha)
    onpath = mu[:, :, None] * agg[None, None, :] * alpha > 1e-14
    b_belief = float(np.abs(q_hat - Q_sigma[None, :, :])[onpath].max()) if onpath.any() else 0.0
    return s, b_belief


@dataclass(frozen=True)
class Thresholds:
    """Outcome-classifier cutoffs (architecture.md §8)."""

    nash_gain: float = 0.05
    dist: float = 1e-3
    sce_s: float = 0.05
    sce_belief: float = 0.02


def classify_run(
    econ: Economy,
    alpha: np.ndarray,
    beta: np.ndarray,
    mu: np.ndarray,
    m_hat: np.ndarray | None,
    q_hat: np.ndarray | None,
    thresholds: Thresholds = Thresholds(),
) -> str:
    """Outcome classifier (architecture.md §8): "NASH" | "SCE_TRAP" | "UNSETTLED".

    NASH: the complete-information BR gain b is within `thresholds.nash_gain`
    and mu is stationary (d within `thresholds.dist`). SCE_TRAP: b exceeds
    nash_gain but mu is stationary, and beliefs (m_hat, q_hat) are given and
    certify (alpha, beta) as an approximate SCE (`sce_certificate`'s s, b_belief
    within `thresholds.sce_s`, `thresholds.sce_belief`). Everything else —
    including b > nash_gain with m_hat or q_hat missing — is UNSETTLED.
    """
    b, _ = br_gain(econ, alpha, beta, mu)
    d = stationarity(econ, alpha, beta, mu)
    if b <= thresholds.nash_gain and d <= thresholds.dist:
        return "NASH"
    if d <= thresholds.dist and b > thresholds.nash_gain and m_hat is not None and q_hat is not None:
        s, b_belief = sce_certificate(econ, alpha, beta, m_hat, q_hat, mu=mu)
        if s <= thresholds.sce_s and b_belief <= thresholds.sce_belief:
            return "SCE_TRAP"
    return "UNSETTLED"
