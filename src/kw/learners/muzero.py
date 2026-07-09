"""Tabular PUCT self-play learner: paper Algorithms 3-4 (empirical + known-env).

This is the paper's MuZero-style learner (architecture.md §7.2-§7.3). It owns
*only* the pieces the fairness contract (§7.5) assigns to a `Learner`: routing
each meeting/trade/consumption observation into an empirical count model,
sampling actions from a PUCT visit-policy, and running the end-of-iteration
policy/value update (Algorithm 4). Everything numerically load-bearing is
delegated: the `EmpiricalModel`/`ModelBank` (Task 8) accumulate M-hat, and one
`Planner` (Task 9) computes the (L2)-(L3) DP tables and the (P1)-(P2) PUCT
allocation and memoizes each root's (pi, value) for the iteration (§6.3.5).
The learner never reimplements a DP recursion or a bandit loop.

Two backend regimes (chosen by `known_rules`, wired identically into the
training and deployment planners via `_make_backend`):
  - Fully empirical (`known_rules=False`, all three flags off): the trade kernel
    comes entirely from the empirical `p_trade` count table, so `observe_offer`/
    q-hat is never fed (fidelity 12.10 — q-hat is a `RulesBackend` input, absent
    here) and same-good meetings (k == j) short-circuit to no-offer without
    touching the model or planner (adjudication 3).
  - Known environment (`known_rules=True`, paper §6.4): a `RulesBackend` keeps
    the learned m-hat/q-hat but takes the transition/reward kernels from the true
    rules; `observe_trade` additionally feeds `observe_offer` (opposite-direction
    acceptance evidence, still skipping k == j) so q-hat can move off its prior.
`per_agent=True` is accepted but raises `NotImplementedError` (the action
protocol carries no agent id to route per-agent banks).

RNG discipline (§12.2): action sampling draws from the per-call `rng` argument
threaded by `simulate.run_learning`, so a run is reproduced by (econ, cfg,
seed). The constructor `rng` is stored but reserved — it seeds nothing in the
fully empirical loop, where all stochasticity is either the shared matching RNG
(matching, action Bernoullis) or the deterministic PUCT allocation.

End-of-iteration update (Algorithm 4, all indices per the controller
adjudications): with ζ_n = max(zeta_min, zeta0 * zeta_decay^(n-1)) for 1-based
n (adj 2),
  - V-bar_i(h): call-count-weighted mean of memoized root values, pooling every
    trade root (i, k=h, j) over j with the cons root (i, h) into the own-holding
    bin h (adj 4); unvisited (i,h) keep the old v[i,h] as stand-in.
  - alpha[i,k,j] <- clip_eps[(1-ζ)alpha + ζ pi(1)] on visited feasible trade
    roots; beta[i,h] likewise on visited gamma-feasible cons roots (adj 5),
    clip_eps = [eps_floor, 1-eps_floor]; infeasible cells stay exactly 0.
  - v[i,h] <- (1-ζ)v + ζ V-bar on visited (i,h); `centered_values` instead
    targets V-bar_i(h) - V-bar_i(f_i) and pins v[i, f_i] = 0 (adj 6).
  - Delta-bar_n = (Σ|Δalpha| over queried trade roots + Σ|Δbeta| over queried
    gamma-feasible cons roots) / max(1, #trade + #cons), POST-clip (adj 3).
Tables update only after all rounds of the iteration have been played, then the
planner is refreshed (cache cleared, new tables installed) for the next
iteration (adj 9), so the planner never sees a table mutated mid-search.

`deployment_policy` (adj 1/8): a fresh planner pass over every feasible root at
(h_dep, n_sims_dep, tau_dep) using the current (alpha, beta, v) as priors and
the same backup mode, emitting alpha* = pi_dep(1|i,k,j), beta* = pi_dep(1|i,h)
UNCLIPPED (infeasible cells exactly 0). No records, no model update, a fresh
`Planner` so no training-memoization leaks in.
"""

import dataclasses

import numpy as np

from kw.economies import Economy
from kw.empirical_model import ModelBank
from kw.learners.base import ConsRecord, LearnerConfig, TradeRecord
from kw.planner import EmpiricalBackend, Planner, RulesBackend


class MuZeroLearner:
    """PUCT self-play learner over the empirical model (Learner protocol, §7.5)."""

    def __init__(
        self,
        econ: Economy,
        cfg: LearnerConfig,
        rng: np.random.Generator,
        known_rules: bool = False,
        greedy_backup: bool = False,
        centered_values: bool = False,
        ema_gain: float = 0.0,
        per_agent: bool = False,
    ) -> None:
        if per_agent:
            raise NotImplementedError(
                "per_agent banks need an agent id, absent from the trade/cons action protocol"
            )
        self.econ = econ
        self.cfg = cfg
        self.rng = rng
        self.known_rules = known_rules
        self.greedy_backup = greedy_backup
        self.centered_values = centered_values
        self.eps = cfg.eps_floor

        self.alpha = np.where(econ.lambda_feasible, 0.5, 0.0)
        self.beta = np.where(econ.gamma_feasible, 0.5, 0.0)
        self.v = np.zeros((econ.n_types, econ.n_goods), dtype=np.float64)

        self.bank = ModelBank(econ, n_banks=1, kappa0=cfg.kappa0, ema_gain=ema_gain)
        self.model = self.bank[0]
        self.planner = Planner(
            econ, self._make_backend(), self.alpha, self.beta, self.v,
            cfg, greedy=greedy_backup,
        )

        # (i,k,j) -> [count, pi(1), root_value]; (i,h) -> [count, pi(1), root_value].
        self._trade_visits: dict[tuple[int, int, int], list] = {}
        self._cons_visits: dict[tuple[int, int], list] = {}

    def _make_backend(self):
        """Backend for both the training and deployment planners: the known-
        environment variant plugs learned (m-hat, q-hat) into the true rules
        (`RulesBackend`); the fully-empirical variant reads every kernel from
        counts (`EmpiricalBackend`). Both wrap the same shared model."""
        if self.known_rules:
            return RulesBackend(self.econ, self.model)
        return EmpiricalBackend(self.model)

    def observe_meeting(self, i: int, j_good: int) -> None:
        self.model.observe_meeting(i, j_good)

    def trade_action(self, i: int, k: int, j: int, rng: np.random.Generator) -> int:
        if k == j:
            return 0
        pi, value = self.planner.query_trade(i, k, j)
        rec = self._trade_visits.get((i, k, j))
        if rec is None:
            self._trade_visits[(i, k, j)] = [1, float(pi[1]), float(value)]
        else:
            rec[0] += 1
        return int(rng.random() < pi[1])

    def cons_action(self, i: int, h: int, rng: np.random.Generator) -> int:
        pi, value = self.planner.query_cons(i, h)
        rec = self._cons_visits.get((i, h))
        if rec is None:
            self._cons_visits[(i, h)] = [1, float(pi[1]), float(value)]
        else:
            rec[0] += 1
        return int(rng.random() < pi[1])

    def observe_trade(self, rec: TradeRecord) -> None:
        self.model.observe_trade(rec)
        # Known-environment only: the acceptance decision by the holder of rec.k
        # (offered rec.j) is evidence for the opposite-direction query q-hat[rec.j,
        # rec.k] that a holder of rec.j makes about a partner holding rec.k
        # (adjudication; fidelity 12.10 -- the empirical variant never feeds q-hat).
        if self.known_rules and rec.k != rec.j:
            self.model.observe_offer(k=rec.j, j=rec.k, accepted=bool(rec.lam))

    def observe_cons(self, rec: ConsRecord) -> None:
        self.model.observe_cons(rec)

    def end_iteration(self, n: int) -> float:
        cfg, eps, econ = self.cfg, self.eps, self.econ
        zeta = max(cfg.zeta_min, cfg.zeta0 * cfg.zeta_decay ** (n - 1))
        n_types, n_goods, f = econ.n_types, econ.n_goods, econ.f

        alpha_old = self.alpha.copy()
        beta_old = self.beta.copy()

        # V-bar: call-count-weighted mean of memoized root values, binned by own-holding.
        num = np.zeros((n_types, n_goods), dtype=np.float64)
        cnt = np.zeros((n_types, n_goods), dtype=np.float64)
        for (i, k, _j), (c, _pi1, val) in self._trade_visits.items():
            num[i, k] += c * val
            cnt[i, k] += c
        for (i, h), (c, _pi1, val) in self._cons_visits.items():
            num[i, h] += c * val
            cnt[i, h] += c
        visited = cnt > 0
        vbar = self.v.copy()
        vbar[visited] = num[visited] / cnt[visited]

        # Policy updates on visited feasible roots (infeasible cells stay 0).
        for (i, k, j), (_c, pi1, _val) in self._trade_visits.items():
            if econ.lambda_feasible[i, k, j]:
                mixed = (1.0 - zeta) * self.alpha[i, k, j] + zeta * pi1
                self.alpha[i, k, j] = min(max(mixed, eps), 1.0 - eps)
        for (i, h), (_c, pi1, _val) in self._cons_visits.items():
            if econ.gamma_feasible[i, h]:
                mixed = (1.0 - zeta) * self.beta[i, h] + zeta * pi1
                self.beta[i, h] = min(max(mixed, eps), 1.0 - eps)

        # Value update (optionally centered on the production good f_i, which is pinned to 0).
        if self.centered_values:
            target = vbar - vbar[np.arange(n_types), f][:, None]
        else:
            target = vbar
        new_v = self.v.copy()
        new_v[visited] = (1.0 - zeta) * self.v[visited] + zeta * target[visited]
        if self.centered_values:
            new_v[np.arange(n_types), f] = 0.0
        self.v = new_v

        # Delta-bar_n: mean absolute post-clip policy change over queried roots.
        sum_t = sum(
            abs(self.alpha[i, k, j] - alpha_old[i, k, j]) for (i, k, j) in self._trade_visits
        )
        n_t = len(self._trade_visits)
        cons_feasible = [(i, h) for (i, h) in self._cons_visits if econ.gamma_feasible[i, h]]
        sum_c = sum(abs(self.beta[i, h] - beta_old[i, h]) for (i, h) in cons_feasible)
        n_c = len(cons_feasible)
        delta = (sum_t + sum_c) / max(1, n_t + n_c)

        self.planner.refresh(self.alpha, self.beta, self.v)
        self._trade_visits = {}
        self._cons_visits = {}
        return float(delta)

    def deployment_policy(self) -> tuple[np.ndarray, np.ndarray]:
        cfg, econ = self.cfg, self.econ
        dep_cfg = dataclasses.replace(
            cfg, horizon=cfg.h_dep, n_sims=cfg.n_sims_dep, tau=cfg.tau_dep
        )
        planner = Planner(
            econ, self._make_backend(), self.alpha, self.beta, self.v,
            dep_cfg, greedy=self.greedy_backup,
        )
        n_types, n_goods = econ.n_types, econ.n_goods
        alpha_star = np.zeros((n_types, n_goods, n_goods), dtype=np.float64)
        beta_star = np.zeros((n_types, n_goods), dtype=np.float64)
        for i in range(n_types):
            for k in range(n_goods):
                for j in range(n_goods):
                    if econ.lambda_feasible[i, k, j]:
                        pi, _ = planner.query_trade(i, k, j)
                        alpha_star[i, k, j] = pi[1]
            for h in range(n_goods):
                pi, _ = planner.query_cons(i, h)
                beta_star[i, h] = pi[1]
        return alpha_star, beta_star
