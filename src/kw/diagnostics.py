"""Complete-information best-response gain: the SCE-is-not-Nash certificate.

`br_gain` is the paper's diagnostic (architecture.md §6.2) reported alongside
every restricted-information SCE: it re-evaluates a policy profile under the
*objective* environment (rvi_solve/policy_eval, not the subjective variants
`solve_sce` uses internally) and reports how much any type could still gain
by deviating, given full information about (mu, Q). A stationary Nash
equilibrium has br_gain == 0; a self-confirming equilibrium that is not Nash
has br_gain > 0 for at least one type (paper §7's H2 claim). This module is
deliberately minimal — Task 11 extends it with further diagnostics.
"""

import numpy as np

from kw.bellman import policy_eval, rvi_solve
from kw.distributions import aggregate, invariant, reciprocation
from kw.economies import Economy


def _default_mu0(econ: Economy) -> np.ndarray:
    """Everyone holds f[i]; economy C splits fiat share M0 on idx 0, remainder on f[i]."""
    mu0 = np.zeros((econ.n_types, econ.n_goods), dtype=np.float64)
    if econ.fiat_idx is not None:
        mu0[:, econ.fiat_idx] = econ.M0
        mu0[np.arange(econ.n_types), econ.f] = 1.0 - econ.M0
    else:
        mu0[np.arange(econ.n_types), econ.f] = 1.0
    return mu0


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
        mu = invariant(econ, _default_mu0(econ), alpha, beta)
    agg = aggregate(econ, mu)
    Q = reciprocation(econ, mu, alpha)
    gaps = np.zeros(econ.n_types, dtype=np.float64)
    for i in range(econ.n_types):
        g_star, _, _ = rvi_solve(econ, i, agg, Q)
        g_pol, _ = policy_eval(econ, i, alpha[i], beta[i], agg, Q)
        gaps[i] = g_star - g_pol
    return float(gaps.max()), gaps
