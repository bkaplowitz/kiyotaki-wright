"""Learner records, hyperparameters, and the shared learner protocol (architecture.md §7.3, §7.5).

`TradeRecord`/`ConsRecord` are the two experience units a learner emits as it
plays: one per bilateral meeting decision, one per consumption decision.
`LearnerConfig` is Table 1 of the paper, transcribed field-for-field as a
frozen dataclass so every learning run is fully specified by one immutable
object. `Learner` is the fairness-contract protocol (architecture.md §7.5):
`simulate.py` drives population dynamics, matching, and stopping identically
for every learner, and only ever calls through this interface, which is why
extension learners (§7.4) plug in without touching `simulate.py`.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class TradeRecord:
    agent: int
    i: int
    k: int
    j: int
    lam: int
    h_next: int
    r: float


@dataclass(frozen=True)
class ConsRecord:
    agent: int
    i: int
    h: int
    gamma: int
    h_next: int
    r: float


@dataclass(frozen=True)
class LearnerConfig:
    """Table 1 hyperparameters (architecture.md §7.3), paper defaults."""

    n_agents: int = 1800
    rounds: int = 35
    n_iters_max: int = 180
    n_min: int = 100
    delta_stop: float = 0.003
    patience: int = 8
    n_sims: int = 32
    horizon: int = 10
    c_puct: float = 1.5
    tau: float = 1.0
    zeta0: float = 0.45
    zeta_decay: float = 0.94
    zeta_min: float = 0.01
    kappa0: float = 2.0
    eps_floor: float = 0.03
    h_dep: int = 14
    n_sims_dep: int = 96
    tau_dep: float = 0.35
    eval_rounds: int = 120
    eval_interval: int = 5


class Learner(Protocol):
    """The fairness-contract interface every learner (replication or extension) implements.

    `simulate.py` owns population init, matching, trade resolution, the
    (D1) stopping rule, deployment extraction, and fixed-policy evaluation
    identically across learners (architecture.md §7.5); a `Learner`
    contributes only action selection, record ingestion, end-of-iteration
    bookkeeping, and its deployment policy.
    """

    def observe_meeting(self, i: int, j_good: int, agent: int = 0) -> None:
        """Own type `i` saw a matched partner holding `j_good`, before trade_action (Alg 3 l.5).

        `agent` is the true simulator agent id (Task 18, extension E2): the
        harness always passes it so per-agent learners can route observations
        and action queries to private model banks; shared learners ignore it.
        """
        ...

    def trade_action(
        self, i: int, k: int, j: int, rng: np.random.Generator, agent: int = 0
    ) -> int: ...

    def cons_action(self, i: int, h: int, rng: np.random.Generator, agent: int = 0) -> int: ...

    def observe_trade(self, rec: TradeRecord) -> None: ...

    def observe_cons(self, rec: ConsRecord) -> None: ...

    def end_iteration(self, n: int) -> float: ...

    def deployment_policy(self) -> tuple[np.ndarray, np.ndarray]: ...
