"""Differential (average-reward) tabular Q-learning baseline (extensions/architecture.md §4).

`QLearner` implements the same `Learner` protocol (§7.5) as `MuZeroLearner`, so
`simulate.run_learning` drives it identically, but it replaces the PUCT
self-play/DP-planner machinery with a plain tabular Q-learning update over the
two-stage (trade, then consumption) period structure. There is no empirical
model, no planner, no policy/value mixing: greedy behavior is read directly
off the Q-tables and both tables are updated on-line, one meeting at a time.

Two population-shared tables (per-agent banks are out of scope here, see
below): `QT[i,k,j,lam]` for the trade-stage root (type i, currently holding
k, meets a partner offering j, considers action lam in {0,1}), and
`QC[i,h,gamma]` for the consumption-stage root (type i, now holding h after
trade, considers action gamma in {0,1}). `gbar[i]` is the running average-
reward estimate used to center the consumption-stage TD target (Wan-Naik-
Sutton 2021 differential Q-learning, adapted to this two-stage period).
Infeasible cells (`~econ.lambda_feasible`, and `~econ.gamma_feasible` on the
gamma=1 column only -- gamma=0/hold is always feasible) are pinned at -inf
so they never win an argmax and are excluded from the greedy/deployment
policy and from the `end_iteration` delta.

Update equations (ext §4), applied per observed transition:
  trade stage        delta_T = max_gamma QC[i,h+,gamma] - QT[i,k,j,lam]   (r_T = 0)
                      QT += q_lr * delta_T
  consumption stage   delta_C = r_C - gbar[i] + max_lam' QT[i,h++,j',lam'] - QC[i,h,gamma]
                      QC += q_lr * delta_C ;  gbar[i] += eta * q_lr * delta_C

The consumption-stage TD cannot complete when `observe_cons` fires: the
bootstrap term `max_lam' QT[i,h++,j',lam']` needs the *next* meeting's
offered good j', which the (agent, i, h, gamma, h_next, r) record does not
carry. `observe_cons` therefore only stashes a pending per-agent record;
`observe_meeting_agent(agent, i, k, j_good)` -- an optional hook on the
`Learner` protocol (§7.5) that `simulate.py`'s `_run_trade_round` calls,
guarded by `hasattr`, for both agents of every meeting, before either agent's
`trade_action` -- looks up and completes any pending record for that agent,
using the just-revealed (k, j_good) as (i, h++, j'). `MuZeroLearner` does not
implement this hook (its meeting counts already come through
`observe_meeting`), so the guard leaves it untouched.

When the revealed meeting is same-good (k == j_good), (i, h++, j') is not a
trade root at all -- `lambda_feasible` requires k != j, so `QT[i,h++,j',:]`
is exactly the -inf infeasibility mask, not a merely-unvisited zero, and
`max` over it must not be used as a bootstrap target (it would poison QC/
gbar with -inf, later surfacing as nan on an inf - inf cancellation). A
same-good meeting has no trade decision to make (adjudication 3, mirrored
from `MuZeroLearner`): the agent proceeds directly to the consumption stage
still holding h++, so the bootstrap falls back to `max_gamma QC[i,h++,gamma]`
-- the same reduction the trade-stage TD already performs for its own
bootstrap, one level down.

`per_agent=True` is accepted but raises `NotImplementedError`, matching
`MuZeroLearner`: routing per-agent banks needs an agent id that the
`trade_action`/`cons_action` protocol does not carry (deferred, per the
architecture's per-agent extension). `dyna != "none"` also raises
`NotImplementedError` naming Task 16, which is where the Dyna planning
backups (`extensions/architecture.md` §4's K=20-backup variants) belong.

RNG discipline (§12.2): action sampling draws from the per-call `rng`
argument threaded by `simulate.run_learning`. The constructor `rng` is
stored but reserved, matching `MuZeroLearner` -- it seeds nothing here.
"""

import numpy as np

from kw.economies import Economy
from kw.learners.base import ConsRecord, LearnerConfig, TradeRecord


class QLearner:
    """Differential average-reward tabular Q-learning (Learner protocol, §7.5)."""

    def __init__(
        self,
        econ: Economy,
        cfg: LearnerConfig,
        rng: np.random.Generator,
        dyna: str = "none",
        q_lr: float = 0.1,
        eta: float = 0.5,
        dyna_k: int = 20,
        per_agent: bool = False,
    ) -> None:
        if per_agent:
            raise NotImplementedError(
                "per_agent banks need an agent id, absent from the trade/cons action protocol"
            )
        if dyna != "none":
            raise NotImplementedError(f"dyna={dyna!r} planning backups are Task 16")
        self.econ = econ
        self.cfg = cfg
        self.rng = rng
        self.dyna = dyna
        self.q_lr = q_lr
        self.eta = eta
        self.dyna_k = dyna_k
        self.per_agent = per_agent
        self.eps = cfg.eps_floor

        n_types, n_goods = econ.n_types, econ.n_goods
        self.QT = np.zeros((n_types, n_goods, n_goods, 2), dtype=np.float64)
        self.QT[~econ.lambda_feasible] = -np.inf
        self.QC = np.zeros((n_types, n_goods, 2), dtype=np.float64)
        self.QC[..., 1] = np.where(econ.gamma_feasible, 0.0, -np.inf)
        self.gbar = np.zeros(n_types, dtype=np.float64)

        self._pending: dict[int, ConsRecord] = {}
        self._trade_visited: set[tuple[int, int, int]] = set()
        self._cons_visited: set[tuple[int, int]] = set()
        self._alpha_prev, self._beta_prev = self._greedy_alpha_beta()

    def observe_meeting(self, i: int, j_good: int) -> None:
        """No-op: this action protocol carries no agent id, and this learner's
        pending-transition bookkeeping is driven by `observe_meeting_agent`
        instead (see the module docstring)."""

    def trade_action(self, i: int, k: int, j: int, rng: np.random.Generator) -> int:
        if k == j:
            return 0
        self._trade_visited.add((i, k, j))
        if rng.random() < self.eps:
            return int(rng.integers(0, 2))
        return int(self.QT[i, k, j, 1] > self.QT[i, k, j, 0])

    def cons_action(self, i: int, h: int, rng: np.random.Generator) -> int:
        self._cons_visited.add((i, h))
        feasible1 = bool(self.econ.gamma_feasible[i, h])
        if rng.random() < self.eps:
            return int(rng.integers(0, 2)) if feasible1 else 0
        return int(feasible1 and self.QC[i, h, 1] > self.QC[i, h, 0])

    def observe_trade(self, rec: TradeRecord) -> None:
        if not self.econ.lambda_feasible[rec.i, rec.k, rec.j]:
            return
        target = float(np.max(self.QC[rec.i, rec.h_next, :]))
        delta_t = target - self.QT[rec.i, rec.k, rec.j, rec.lam]
        self.QT[rec.i, rec.k, rec.j, rec.lam] += self.q_lr * delta_t

    def observe_cons(self, rec: ConsRecord) -> None:
        self._pending[rec.agent] = rec

    def observe_meeting_agent(self, agent: int, i: int, k: int, j_good: int) -> None:
        rec = self._pending.pop(agent, None)
        if rec is None:
            return
        if k == j_good:
            target = float(np.max(self.QC[i, k, :]))
        else:
            target = float(np.max(self.QT[i, k, j_good, :]))
        delta_c = rec.r - self.gbar[rec.i] + target - self.QC[rec.i, rec.h, rec.gamma]
        self.QC[rec.i, rec.h, rec.gamma] += self.q_lr * delta_c
        self.gbar[rec.i] += self.eta * self.q_lr * delta_c

    def _greedy_alpha_beta(self) -> tuple[np.ndarray, np.ndarray]:
        econ = self.econ
        alpha = np.where(econ.lambda_feasible & (self.QT[..., 1] > self.QT[..., 0]), 1.0, 0.0)
        beta = np.where(econ.gamma_feasible & (self.QC[..., 1] > self.QC[..., 0]), 1.0, 0.0)
        return alpha, beta

    @property
    def alpha(self) -> np.ndarray:
        return self._greedy_alpha_beta()[0]

    @property
    def beta(self) -> np.ndarray:
        return self._greedy_alpha_beta()[1]

    def end_iteration(self, n: int) -> float:
        econ = self.econ
        alpha_new, beta_new = self._greedy_alpha_beta()

        trade_feasible = [(i, k, j) for (i, k, j) in self._trade_visited if econ.lambda_feasible[i, k, j]]
        sum_t = sum(abs(alpha_new[i, k, j] - self._alpha_prev[i, k, j]) for (i, k, j) in trade_feasible)
        n_t = len(trade_feasible)

        cons_feasible = [(i, h) for (i, h) in self._cons_visited if econ.gamma_feasible[i, h]]
        sum_c = sum(abs(beta_new[i, h] - self._beta_prev[i, h]) for (i, h) in cons_feasible)
        n_c = len(cons_feasible)

        delta = (sum_t + sum_c) / max(1, n_t + n_c)

        self._alpha_prev, self._beta_prev = alpha_new, beta_new
        self._trade_visited, self._cons_visited = set(), set()
        return float(delta)

    def deployment_policy(self) -> tuple[np.ndarray, np.ndarray]:
        return self._greedy_alpha_beta()
