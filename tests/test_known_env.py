import sys
from pathlib import Path

import numpy as np
import pytest
from kw.diagnostics import br_gain, stationarity
from kw.economies import get_economy
from kw.empirical_model import EmpiricalModel
from kw.environment import next_holding_cons, reward_cons_table
from kw.learners.base import LearnerConfig
from kw.learners.muzero import MuZeroLearner
from kw.planner import RulesBackend
from kw.simulate import run_learning

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from run_learning import LEARNER_PRESETS  # noqa: E402

DRY = LearnerConfig(n_agents=60, rounds=5, n_iters_max=6, n_min=2, patience=2)
KNOWN_FLAGS = LEARNER_PRESETS["muzero-known"]


def test_rules_backend_trade_kernel():
    e = get_economy("A1")
    m = EmpiricalModel(e)
    be = RulesBackend(e, m)
    pt = be.p_trade(0)
    assert np.isclose(pt[0, 1, 1, 1], 0.5) and np.isclose(pt[0, 1, 1, 0], 0.5)  # q̂=1/2 prior
    assert pt[0, 1, 0, 0] == 1.0


def test_rules_backend_trade_kernel_reflects_learned_q():
    e = get_economy("A1")
    m = EmpiricalModel(e)
    for _ in range(9):
        m.observe_offer(k=0, j=1, accepted=True)
    be = RulesBackend(e, m)
    q = m.q_hat()[0, 1]
    pt = be.p_trade(0)
    assert np.isclose(pt[0, 1, 1, 1], q)
    assert np.isclose(pt[0, 1, 1, 0], 1.0 - q)
    for k in range(e.n_goods):  # same-good rows stay onehot(k) at either action
        assert np.isclose(pt[k, k, 0, k], 1.0)
        assert np.isclose(pt[k, k, 1, k], 1.0)
    assert np.allclose(pt.sum(axis=-1), 1.0)


def test_rules_backend_cons_and_reward_kernels():
    e = get_economy("C")
    m = EmpiricalModel(e)
    be = RulesBackend(e, m)
    nh = next_holding_cons(e)
    for i in range(e.n_types):
        pc = be.p_cons(i)
        assert pc.shape == (e.n_goods, 2, e.n_goods)
        for h in range(e.n_goods):
            for g in range(2):
                assert pc[h, g, nh[i, h, g]] == 1.0
        assert np.allclose(pc.sum(axis=-1), 1.0)
        assert np.all(be.r_trade(i) == 0.0)
        rc, true = be.r_cons(i), reward_cons_table(e)[i]
        assert np.all(np.isfinite(rc))                       # -inf -> large negative
        finite = np.isfinite(true)
        assert np.allclose(rc[finite], true[finite])
        assert np.all(rc[~finite] < -1e6)


def test_dry_run_muzero_known_completes():
    assert KNOWN_FLAGS == {
        "known_rules": True,
        "greedy_backup": True,
        "centered_values": True,
    }
    for name in ["A1", "C"]:
        e = get_economy(name)
        lr = MuZeroLearner(e, DRY, np.random.default_rng(0), **KNOWN_FLAGS)
        res = run_learning(e, lr, DRY, seed=0)
        assert np.isfinite(res.alpha_star).all() and np.isfinite(res.beta_star).all()
        assert (res.alpha_star >= 0.0).all() and (res.alpha_star <= 1.0).all()
        assert (res.beta_star >= 0.0).all() and (res.beta_star <= 1.0).all()
        assert (res.alpha_star[~e.lambda_feasible] == 0.0).all()
        assert (res.beta_star[~e.gamma_feasible] == 0.0).all()
        assert not np.allclose(lr.model.q_hat(), 0.5)         # observe_offer feed fired


@pytest.mark.slow
def test_c_known_env_reaches_nash():
    e = get_economy("C")
    cfg = LearnerConfig()
    lr = MuZeroLearner(e, cfg, np.random.default_rng(0), **KNOWN_FLAGS)
    res = run_learning(e, lr, cfg, seed=0)
    # Primary H4 check (paper Table 2: gain 0): the known-environment variant
    # reaches a complete-information Nash. br_gain compares the deployed policy to
    # its *optimal* response at the policy's own invariant, so it stays meaningful.
    b, _ = br_gain(e, res.alpha_star, res.beta_star, res.mu_star)
    assert b < 0.1
    # Secondary: the settled holdings are near-stationary, measured at the finite-N
    # empirical terminal distribution (the metric run_learning saves). Its floor is
    # the O(1/sqrt(n_agents)) ~ 0.024 sampling noise at n_agents=1800, NOT the
    # paper's 1.74e-8 continuum figure. (That figure is stationarity at the computed
    # invariant mu_star, which the invariant() solver drives to ~1e-13 for *any*
    # policy -- Nash or not -- so it cannot certify Nash; do not assert against it.)
    stat = stationarity(e, res.alpha_star, res.beta_star, res.trace["agg_holdings"][-1])
    assert stat < 0.05
