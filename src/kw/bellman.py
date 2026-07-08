"""Average-reward Bellman kernel: RVI, greedy extraction, policy evaluation.

Per-type relative value iteration on (B3)-(B4) (architecture.md §5.4), bias
normalized at the reference good k_ref = f(i) (§12 item 5). Objective
variants take the population marginal `agg (G,)` and reciprocation kernel
`Q (G,G)`; the subjective variants for Solver 2 (architecture.md §6.2) take a
type's own beliefs `m_i (G,)`, `q_i (G,G)` in their place — kernel T_q of
(S2)-(S3) has the same algebraic form, so both share one iteration core.

Stopping rule: alongside the standard RVI criterion (sup-norm on successive
normalized V and on g), the loop also stops once the per-state Bellman
residual D = TV - V has stabilized state-wise. Under multichain dynamics
(e.g. skeptical beliefs q_i = 0, where a type's consumption good is a
higher-gain absorbing state than k_ref) relative values of states in
recurrent classes with gain != g(k_ref) drift linearly forever, but D
converges to the vector of per-state optimal gains; g = D[k_ref] is then the
exact gain at the reference good, which is the scalar the paper's per-type
gain means. Multichain inputs are legitimate (paper fn. 4; architecture
§5.4), so this exit returns normally rather than failing.
"""

from collections.abc import Callable

import numpy as np

from kw.economies import Economy
from kw.environment import next_holding_cons, reward_cons_table


def _max_cons_backup(econ: Economy, i: int) -> Callable[[np.ndarray], np.ndarray]:
    """(B3): W[h] = max over feasible gamma of r[i,h,gamma] + V[c(h,gamma)]."""
    r = reward_cons_table(econ)[i]
    c = next_holding_cons(econ)[i]

    def backup(V: np.ndarray) -> np.ndarray:
        return (r + V[c]).max(axis=1)

    return backup


def _max_trade_backup(
    econ: Economy, i: int, agg: np.ndarray, Q: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    """(B4): TV[k] = sum_j agg[j] max_l {(1-l)W[k] + l(Q[k,j]W[j] + (1-Q[k,j])W[k])}."""
    feasible = econ.lambda_feasible[i]

    def backup(W: np.ndarray) -> np.ndarray:
        offer = Q * W[None, :] + (1.0 - Q) * W[:, None]
        offer = np.where(feasible, offer, -np.inf)
        return np.maximum(W[:, None], offer) @ agg

    return backup


def _avg_cons_backup(
    econ: Economy, i: int, beta_i: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    r = reward_cons_table(econ)[i]
    c = next_holding_cons(econ)[i]
    r_cons = np.where(np.isfinite(r[:, 1]), r[:, 1], 0.0)

    def backup(V: np.ndarray) -> np.ndarray:
        stay = r[:, 0] + V[c[:, 0]]
        cons = r_cons + V[c[:, 1]]
        return (1.0 - beta_i) * stay + beta_i * cons

    return backup


def _avg_trade_backup(
    alpha_i: np.ndarray, agg: np.ndarray, Q: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    def backup(W: np.ndarray) -> np.ndarray:
        offer = Q * W[None, :] + (1.0 - Q) * W[:, None]
        return ((1.0 - alpha_i) * W[:, None] + alpha_i * offer) @ agg

    return backup


def _run_rvi(
    econ: Economy,
    i: int,
    cons_backup: Callable[[np.ndarray], np.ndarray],
    trade_backup: Callable[[np.ndarray], np.ndarray],
    tol: float,
    max_iter: int,
    label: str,
) -> tuple[float, np.ndarray, np.ndarray]:
    kref = int(econ.f[i])
    V = np.zeros(econ.n_goods, dtype=np.float64)
    g = 0.0
    D_prev = np.full(econ.n_goods, np.inf)
    for _ in range(max_iter):
        W = cons_backup(V)
        TV = trade_backup(W)
        D = TV - V
        g_new = float(TV[kref])
        V_new = TV - g_new
        unichain_done = np.max(np.abs(V_new - V)) <= tol and abs(g_new - g) <= tol
        # Multichain exit: D stabilizes state-wise even though V drifts in
        # recurrent classes whose gain differs from g(kref) (module docstring).
        multichain_done = np.max(np.abs(D - D_prev)) <= tol
        if unichain_done or multichain_done:
            return g_new, V_new, cons_backup(V_new)
        V, g, D_prev = V_new, g_new, D
    raise RuntimeError(
        f"{label}: RVI did not converge within {max_iter} iterations "
        f"(residual={np.max(np.abs(D - D_prev)):.3e} > tol={tol:.3e})"
    )


def rvi_solve(
    econ: Economy,
    i: int,
    agg: np.ndarray,
    Q: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 200_000,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Optimal gain, bias V (V[f[i]] = 0), and W for type i under (agg, Q)."""
    return _run_rvi(
        econ,
        i,
        _max_cons_backup(econ, i),
        _max_trade_backup(econ, i, agg, Q),
        tol,
        max_iter,
        "rvi_solve",
    )


def rvi_solve_subjective(
    econ: Economy,
    i: int,
    m_i: np.ndarray,
    q_i: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 200_000,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Subjective RVI under type i's own beliefs (m_i, q_i), kernel T_q (S2)-(S3)."""
    return _run_rvi(
        econ,
        i,
        _max_cons_backup(econ, i),
        _max_trade_backup(econ, i, m_i, q_i),
        tol,
        max_iter,
        "rvi_solve_subjective",
    )


def greedy_policy(
    econ: Economy,
    i: int,
    Q: np.ndarray,
    V: np.ndarray,
    W: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pointwise best response (BR-e)/(BR-c); ties break to action 0.

    Offer iff Q(k,j)*(W(j) - W(k)) > 0, masked to feasible lambda; consume
    iff r[i,h,1] + V[f[i]] > r[i,h,0] + V[h] (infeasible gamma has reward
    -inf, so it never wins).
    """
    gain = Q * (W[None, :] - W[:, None])
    alpha_i = ((gain > 0) & econ.lambda_feasible[i]).astype(np.float64)

    r = reward_cons_table(econ)[i]
    c = next_holding_cons(econ)[i]
    values = r + V[c]
    beta_i = (values[:, 1] > values[:, 0]).astype(np.float64)
    return alpha_i, beta_i


def policy_eval(
    econ: Economy,
    i: int,
    alpha_i: np.ndarray,
    beta_i: np.ndarray,
    agg: np.ndarray,
    Q: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 200_000,
) -> tuple[float, np.ndarray]:
    """Gain and bias of fixed (alpha_i, beta_i): RVI on the sigma-averaged operator."""
    g, V, _ = _run_rvi(
        econ,
        i,
        _avg_cons_backup(econ, i, beta_i),
        _avg_trade_backup(alpha_i, agg, Q),
        tol,
        max_iter,
        "policy_eval",
    )
    return g, V


def policy_eval_subjective(
    econ: Economy,
    i: int,
    alpha_i: np.ndarray,
    beta_i: np.ndarray,
    m_i: np.ndarray,
    q_i: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 200_000,
) -> tuple[float, np.ndarray]:
    """Fixed-policy gain and bias under type i's own beliefs (m_i, q_i)."""
    g, V, _ = _run_rvi(
        econ,
        i,
        _avg_cons_backup(econ, i, beta_i),
        _avg_trade_backup(alpha_i, m_i, q_i),
        tol,
        max_iter,
        "policy_eval_subjective",
    )
    return g, V
