"""Population holdings-distribution dynamics: F_sigma and its fixed point.

Vectorized implementation of (B1)-(B2) reciprocation and (F1)-(F4) one-step
population law (architecture.md §5.3). Pure functions of `Economy`, the
current distribution `mu (I,G)`, and policy `(alpha, beta)`.
"""

import numpy as np

from kw.economies import Economy


def aggregate(econ: Economy, mu: np.ndarray) -> np.ndarray:
    """mu(k) = sum_i nu_i mu_i(k), the good-k population marginal (B1)."""
    return (econ.nu[:, None] * mu).sum(axis=0)


def reciprocation(econ: Economy, mu: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Reciprocation kernel Q(k,j) = sum_i eta_i(j) alpha_i(j,k) (B2).

    eta[i,k] = nu[i]*mu[i,k]/agg[k], zeroed where agg[k]==0 so Q[:, j] = 0
    whenever mu(j) = 0 (no type holds j to reciprocate an offer into it).
    """
    agg = aggregate(econ, mu)
    eta = np.where(agg > 0, econ.nu[:, None] * mu / np.where(agg > 0, agg, 1.0), 0.0)
    Q = np.einsum("ij,ijk->kj", eta, alpha)
    return Q


def one_step(econ: Economy, mu: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Apply F_sigma once: trade flows (F1)-(F2) then consumption/production (F3)."""
    agg = (econ.nu[:, None] * mu).sum(axis=0)                    # μ(k)
    Q = reciprocation(econ, mu, alpha)                           # (B2)
    F = mu[:, :, None] * agg[None, None, :] * alpha * Q[None]    # (F1)
    F[:, np.arange(econ.n_goods), np.arange(econ.n_goods)] = 0.0
    mu_plus = mu - F.sum(axis=2) + F.sum(axis=1)                 # (F2)
    consumed = mu_plus * beta                                    # (F3)
    mu_next = mu_plus - consumed
    mu_next[np.arange(econ.n_types), econ.f] += consumed.sum(axis=1)
    return mu_next


def invariant(
    econ: Economy,
    mu0: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    tol: float = 1e-12,
    max_iter: int = 40_000_000,
) -> np.ndarray:
    """Fixed point of F_sigma via damped forward iteration from mu0.

    Iterates mu_{n+1} = F_sigma(mu_n) and stops when ||mu_{n+1}-mu_n||_inf <=
    tol. If the raw update does not shrink the residual (a sign of
    oscillation between iterates rather than monotone convergence), damping
    kicks in: mu_{n+1} = 0.5*mu_n + 0.5*F_sigma(mu_n). Raises RuntimeError if
    max_iter is exhausted without reaching tol.

    The default cap is sized for near-deterministic learned policies whose
    damped map contracts with spectral gap ~1e-6 per iteration (measured
    ~11.1M iterations to tol on extension E3's slowest economy-C cell);
    3.6x that measured need. Converged calls stop on the tolerance test,
    never the cap, so raising it cannot change any previously converged
    result.
    """
    mu = mu0.copy()
    prev_residual = np.inf
    damping = 1.0
    for _ in range(max_iter):
        mu_raw = one_step(econ, mu, alpha, beta)
        residual = np.max(np.abs(mu_raw - mu))
        if residual <= tol:
            return mu_raw
        if residual > prev_residual:
            damping = 0.5
        mu_next = (1.0 - damping) * mu + damping * mu_raw
        prev_residual = residual
        mu = mu_next
    raise RuntimeError(
        f"invariant: F_sigma did not converge within {max_iter} iterations "
        f"(residual={prev_residual:.3e} > tol={tol:.3e})"
    )
