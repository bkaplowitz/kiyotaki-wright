"""Empirical count model M-hat, the learner's stand-in for the true trading kernel
(architecture.md §7.1).

Six count tables, all initialized at a prior and updated online as
`TradeRecord`/`ConsRecord` events arrive (paper §6.3.1):

| table          | shape         | ratio          | prior per cell        |
|----------------|---------------|----------------|------------------------|
| `C[i,j]`       | (I,G)         | `m_hat`        | 1.0 (flat)             |
| `O[k,j]`       | (G,G)         | `q_hat` denom  | kappa0                 |
| `A_acc[k,j]`   | (G,G)         | `q_hat` numer  | kappa0/2               |
| `N_T[i,k,j,l,:]` | (I,G,G,2,G) | `p_trade`      | 1.0 (flat)             |
| `N_C[i,h,g,:]` | (I,G,2,G)     | `p_cons`       | 1.0 (flat)             |
| `B_T,S_T[i,k,j,l]` | (I,G,G,2) | `r_trade`      | B0=kappa0, S0=0        |
| `B_C,S_C[i,h,g]`   | (I,G,2)   | `r_cons`       | B0=kappa0, S0=0        |

kappa0 governs only the paper's two Beta/reward-average priors: the
offer/acceptance ratio q_hat (A_acc starts at kappa0/2 so q_hat^0 = 1/2) and
the reward averages (numerator sum starts at 0, denominator count starts at
kappa0, so r_hat^0 = 0). The plain categorical ratios (m_hat, p_trade,
p_cons) are ordinary add-one Dirichlet priors with a flat pseudo-count of
1.0 per outcome, independent of kappa0 — any positive uniform pseudo-count
gives the same uniform-over-G starting ratio; 1.0 is the simplest choice and
is documented here rather than tied to kappa0 because it plays no role in
the paper's kappa0-indexed extension sweep (extensions §5.3, E3).

`observe_meeting(i, j_good)` feeds `C`. `observe_offer(k, j, accepted)` feeds
`O`/`A_acc` and requires a true offer (k != j): a same-good meeting never
presents an offer/acceptance decision, so callers must not route those
through this method (they still generate a `TradeRecord` for
`observe_trade`, since the physical transition h_next=k is still a valid
data point for `p_trade`). `observe_trade`/`observe_cons` feed the paired
transition and reward tables for the two decision points; `r_trade` is
carried for structural symmetry with `r_cons` (paper lists both B_T,S_T and
B_C,S_C) but is provably identically zero here, since every `TradeRecord` in
this economy carries r=0 (all reward accrues at consumption, paper §6.3.1).

EMA hook `ema_gain` (extension E1, architecture §7.1, extensions §5.1):
"multiplies every count table by (1-ema_gain) once per record-arrival in its
cell" — the decay is local to the cell (or row, for row-normalized tables)
the record touches, applied to the numerator and its paired
denominator/count table together, immediately before the increment. At the
default `ema_gain=0.0` the multiply is a no-op and the tables are exactly
the paper's cumulative counts.
"""

import numpy as np

from kw.economies import Economy
from kw.learners.base import ConsRecord, TradeRecord

_CAT_PRIOR = 1.0


class EmpiricalModel:
    def __init__(self, econ: Economy, kappa0: float = 2.0, ema_gain: float = 0.0) -> None:
        self.econ = econ
        self.kappa0 = kappa0
        self.ema_gain = ema_gain
        n_types, n_goods = econ.n_types, econ.n_goods

        self._C = np.full((n_types, n_goods), _CAT_PRIOR, dtype=np.float64)
        self._O = np.full((n_goods, n_goods), kappa0, dtype=np.float64)
        self._A_acc = np.full((n_goods, n_goods), kappa0 / 2.0, dtype=np.float64)
        self._N_T = np.full((n_types, n_goods, n_goods, 2, n_goods), _CAT_PRIOR, dtype=np.float64)
        self._B_T = np.full((n_types, n_goods, n_goods, 2), kappa0, dtype=np.float64)
        self._S_T = np.zeros((n_types, n_goods, n_goods, 2), dtype=np.float64)
        self._N_C = np.full((n_types, n_goods, 2, n_goods), _CAT_PRIOR, dtype=np.float64)
        self._B_C = np.full((n_types, n_goods, 2), kappa0, dtype=np.float64)
        self._S_C = np.zeros((n_types, n_goods, 2), dtype=np.float64)

    def observe_meeting(self, i: int, j_good: int) -> None:
        """Type i meets a partner offering j_good: decays and updates C[i, :]."""
        decay = 1.0 - self.ema_gain
        self._C[i, :] *= decay
        self._C[i, j_good] += 1.0

    def observe_offer(self, k: int, j: int, accepted: bool) -> None:
        """A true offer (k != j) was made and either accepted or rejected."""
        if k == j:
            raise ValueError(f"observe_offer requires a true offer k != j; got k=j={k}")
        decay = 1.0 - self.ema_gain
        self._O[k, j] *= decay
        self._A_acc[k, j] *= decay
        self._O[k, j] += 1.0
        if accepted:
            self._A_acc[k, j] += 1.0

    def observe_trade(self, rec: TradeRecord) -> None:
        decay = 1.0 - self.ema_gain
        self._N_T[rec.i, rec.k, rec.j, rec.lam, :] *= decay
        self._N_T[rec.i, rec.k, rec.j, rec.lam, rec.h_next] += 1.0
        self._B_T[rec.i, rec.k, rec.j, rec.lam] *= decay
        self._S_T[rec.i, rec.k, rec.j, rec.lam] *= decay
        self._B_T[rec.i, rec.k, rec.j, rec.lam] += 1.0
        self._S_T[rec.i, rec.k, rec.j, rec.lam] += rec.r

    def observe_cons(self, rec: ConsRecord) -> None:
        decay = 1.0 - self.ema_gain
        self._N_C[rec.i, rec.h, rec.gamma, :] *= decay
        self._N_C[rec.i, rec.h, rec.gamma, rec.h_next] += 1.0
        self._B_C[rec.i, rec.h, rec.gamma] *= decay
        self._S_C[rec.i, rec.h, rec.gamma] *= decay
        self._B_C[rec.i, rec.h, rec.gamma] += 1.0
        self._S_C[rec.i, rec.h, rec.gamma] += rec.r

    def m_hat(self) -> np.ndarray:
        """(I,G) meeting-frequency estimate m_hat_i(j) = C[i,j] / sum_l C[i,l]."""
        return self._C / self._C.sum(axis=1, keepdims=True)

    def q_hat(self) -> np.ndarray:
        """(G,G) offer/acceptance estimate q_hat(k,j) = A_acc[k,j] / O[k,j], pooled over offerer type."""
        return self._A_acc / self._O

    def p_trade(self) -> np.ndarray:
        """(I,G,G,2,G) trade transition estimate P_hat^T_i(h'|k,j,lam)."""
        return self._N_T / self._N_T.sum(axis=-1, keepdims=True)

    def p_cons(self) -> np.ndarray:
        """(I,G,2,G) consumption transition estimate P_hat^C_i(h'|h,gamma)."""
        return self._N_C / self._N_C.sum(axis=-1, keepdims=True)

    def r_trade(self) -> np.ndarray:
        """(I,G,G,2) trade reward estimate; identically 0 since TradeRecord.r is always 0."""
        return self._S_T / self._B_T

    def r_cons(self) -> np.ndarray:
        """(I,G,2) consumption reward estimate r_hat^C_i(h,gamma) = S_C / B_C."""
        return self._S_C / self._B_C


class ModelBank:
    """agent_id -> EmpiricalModel indirection for table ownership (architecture.md §7.1, E2).

    `n_banks=1` pools every agent into one shared model (paper default,
    all agents map to bank 0); `n_banks=n_agents` gives each agent a
    private model (extension E2). Both reduce to `agent_id % n_banks`: 0 for
    every agent when `n_banks=1`, the identity when `n_banks=n_agents` and
    agent ids run 0..n_agents-1.
    """

    def __init__(self, econ: Economy, n_banks: int, **kw) -> None:
        self.n_banks = n_banks
        self.banks = [EmpiricalModel(econ, **kw) for _ in range(n_banks)]

    def bank_for(self, agent_id: int) -> EmpiricalModel:
        return self.banks[agent_id % self.n_banks]

    def __getitem__(self, agent_id: int) -> EmpiricalModel:
        return self.bank_for(agent_id)
