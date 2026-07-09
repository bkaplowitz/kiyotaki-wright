"""PUCT planner: (L2)-(L3) DP tables plus deterministic (P1)-(P2) visit allocation (architecture.md §7.2).

Key implementation fact (paper p. 34): within one search the model is frozen
and the action value l_x is visit-independent, so the DP tables never change
after their first computation for a given root. The planner is therefore two
closed-form pieces glued by one small deterministic loop:

`dp_tables` computes V-hat(h,d), W-hat(h,d) for d = 0..H via the mutual
recursion (L2)-(L3): W-hat(h,d) = (1-beta(h))*Psi^c(h,0;d) + beta(h)*Psi^c(h,1;d);
V-hat(h,d) = sum_j m-hat(j)*[(1-alpha(h,j))*Psi^t(h,j,0;d) + alpha(h,j)*Psi^t(h,j,1;d)];
Psi^t(k,j,l;d) = r-hat^t + sum_h' P-hat^t(h'|k,j,l)*W-hat(h',d);
Psi^c(h,g;d) = r-hat^c + sum_h' P-hat^c(h'|h,g)*V-hat(h',d-1);
terminal V-hat(.,0) = W-hat(.,0) = v (the learner's current per-type value
table). `greedy=True` (known-environment change #2) replaces the prior-weighted
interior mixes with maxes: W-hat(h,d) = max_g Psi^c(h,g;d), V-hat(h,d) =
sum_j m-hat(j)*max_l Psi^t(h,j,l;d). Cost O(|G|^2 * H) per (type, model
snapshot).

`puct_select` is the literal (P1)-(P2) deterministic bandit loop over the two
root actions: min-max normalize the two root values to S_x in [0,1] (S_x =
1/2 when exactly tied), mask infeasible actions to S_x = -inf, then run
exactly `n_sims` steps of argmax_x S_x + c_puct*prior_x*sqrt(N_tot+1)/(1+N_x)
(ties broken to the lowest index, N starting at zero and S frozen after
normalization). Returns the visit policy pi ∝ N^(1/tau) (P2) and the max
root action value over feasible actions.

Two model backends share one `ModelBackend` lookup protocol, each method a
per-type slice: `EmpiricalBackend` reads (m-hat, P-hat^t, P-hat^c, r-hat^t,
r-hat^c) from the Task 8 `EmpiricalModel` (fully empirical case; q-hat is not
part of this protocol since the trade kernel comes entirely from
`p_trade`). `RulesBackend` (paper §6.4, Task 15) is not implemented here.

`Planner` wraps a backend with the learner's current (alpha, beta, v) tables
and `LearnerConfig` knobs, and memoizes the deterministic search per root
(paper §6.3.5): the planner is constructed once per iteration; the *first*
`query_trade(i,k,j)` / `query_cons(i,h)` for a given root computes (pi,
value) from the backend as of that moment and caches it; later queries of
the same root in the same iteration reuse the cache even though the model
has since moved. Different roots first queried at different times may
legitimately see different model snapshots. `refresh(alpha, beta, v)`
installs new policy/value tables and clears the cache, once per iteration.
"""

from typing import Protocol

import numpy as np

from kw.economies import Economy
from kw.empirical_model import EmpiricalModel
from kw.learners.base import LearnerConfig


class ModelBackend(Protocol):
    """Per-type slice of the learner's model, as consumed by `dp_tables`."""

    def m(self, i: int) -> np.ndarray: ...

    def p_trade(self, i: int) -> np.ndarray: ...

    def p_cons(self, i: int) -> np.ndarray: ...

    def r_trade(self, i: int) -> np.ndarray: ...

    def r_cons(self, i: int) -> np.ndarray: ...


class EmpiricalBackend:
    """`ModelBackend` over the Task 8 `EmpiricalModel`, sliced per type at query time."""

    def __init__(self, model: EmpiricalModel) -> None:
        self.model = model

    def m(self, i: int) -> np.ndarray:
        return self.model.m_hat()[i]

    def p_trade(self, i: int) -> np.ndarray:
        return self.model.p_trade()[i]

    def p_cons(self, i: int) -> np.ndarray:
        return self.model.p_cons()[i]

    def r_trade(self, i: int) -> np.ndarray:
        return self.model.r_trade()[i]

    def r_cons(self, i: int) -> np.ndarray:
        return self.model.r_cons()[i]


def dp_tables(
    econ: Economy,
    be: ModelBackend,
    i: int,
    alpha_i: np.ndarray,
    beta_i: np.ndarray,
    v_i: np.ndarray,
    horizon: int,
    greedy: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    V = np.tile(v_i, (horizon + 1, 1))
    W = V.copy()
    pt, pc = be.p_trade(i), be.p_cons(i)
    rt, rc = be.r_trade(i), be.r_cons(i)
    m = be.m(i)
    for d in range(1, horizon + 1):
        psi_c = rc + pc @ V[d - 1]  # (G,2)
        W[d] = (
            psi_c.max(1)
            if greedy
            else (1 - beta_i) * psi_c[:, 0] + beta_i * psi_c[:, 1]
        )
        psi_t = rt + np.einsum("kjlh,h->kjl", pt, W[d])  # (G,G,2)
        V[d] = (
            m
            * (
                psi_t.max(2)
                if greedy
                else (1 - alpha_i) * psi_t[:, :, 0] + alpha_i * psi_t[:, :, 1]
            )
        ).sum(1)
    psi_c_top = rc + pc @ V[horizon - 1]
    psi_t_top = rt + np.einsum("kjlh,h->kjl", pt, W[horizon])
    return psi_t_top, psi_c_top


def puct_select(
    values: np.ndarray,
    prior: np.ndarray,
    feasible: np.ndarray,
    n_sims: int,
    c_puct: float,
    tau: float,
) -> tuple[np.ndarray, float]:
    values = np.asarray(values, dtype=np.float64)
    prior = np.asarray(prior, dtype=np.float64)
    feasible = np.asarray(feasible, dtype=bool)

    lo, hi = values.min(), values.max()
    S = np.full_like(values, 0.5) if hi == lo else (values - lo) / (hi - lo)
    S = np.where(feasible, S, -np.inf)

    N = np.zeros_like(values)
    for _ in range(n_sims):
        U = S + c_puct * prior * np.sqrt(N.sum() + 1) / (1 + N)
        x = int(np.argmax(U))
        N[x] += 1.0

    pi = N ** (1.0 / tau)
    pi = pi / pi.sum()
    root_value = float(values[feasible].max())
    return pi, root_value


class Planner:
    """Memoized PUCT search over trade and consumption roots for one iteration."""

    def __init__(
        self,
        econ: Economy,
        backend: ModelBackend,
        alpha: np.ndarray,
        beta: np.ndarray,
        v: np.ndarray,
        cfg: LearnerConfig,
        greedy: bool = False,
    ) -> None:
        self.econ = econ
        self.backend = backend
        self.alpha = alpha
        self.beta = beta
        self.v = v
        self.cfg = cfg
        self.greedy = greedy
        self._trade_cache: dict[tuple[int, int, int], tuple[np.ndarray, float]] = {}
        self._cons_cache: dict[tuple[int, int], tuple[np.ndarray, float]] = {}

    def refresh(self, alpha: np.ndarray, beta: np.ndarray, v: np.ndarray) -> None:
        self.alpha = alpha
        self.beta = beta
        self.v = v
        self._trade_cache.clear()
        self._cons_cache.clear()

    def query_trade(self, i: int, k: int, j: int) -> tuple[np.ndarray, float]:
        key = (i, k, j)
        if key not in self._trade_cache:
            psi_t, _ = dp_tables(
                self.econ, self.backend, i, self.alpha[i], self.beta[i], self.v[i],
                self.cfg.horizon, greedy=self.greedy,
            )
            values = psi_t[k, j]
            prior = np.array([1.0 - self.alpha[i, k, j], self.alpha[i, k, j]])
            feasible = np.array([True, bool(self.econ.lambda_feasible[i, k, j])])
            self._trade_cache[key] = puct_select(
                values, prior, feasible, self.cfg.n_sims, self.cfg.c_puct, self.cfg.tau
            )
        return self._trade_cache[key]

    def query_cons(self, i: int, h: int) -> tuple[np.ndarray, float]:
        key = (i, h)
        if key not in self._cons_cache:
            _, psi_c = dp_tables(
                self.econ, self.backend, i, self.alpha[i], self.beta[i], self.v[i],
                self.cfg.horizon, greedy=self.greedy,
            )
            values = psi_c[h]
            prior = np.array([1.0 - self.beta[i, h], self.beta[i, h]])
            feasible = np.array([True, bool(self.econ.gamma_feasible[i, h])])
            self._cons_cache[key] = puct_select(
                values, prior, feasible, self.cfg.n_sims, self.cfg.c_puct, self.cfg.tau
            )
        return self._cons_cache[key]
