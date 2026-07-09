import numpy as np
from kw.economies import get_economy
from kw.learners.base import LearnerConfig, TradeRecord, ConsRecord
from kw.extensions.qlearning import QLearner
from kw.simulate import run_learning

DRY = LearnerConfig(n_agents=60, rounds=5, n_iters_max=6, n_min=2, patience=2)


def test_trade_update():
    e = get_economy("A1")
    lr = QLearner(e, LearnerConfig(), np.random.default_rng(0), q_lr=0.5)
    lr.QC[0, 1, :] = [2.0, 6.0]
    lr.observe_trade(TradeRecord(agent=0, i=0, k=0, j=1, lam=1, h_next=1, r=0.0))
    assert np.isclose(lr.QT[0, 0, 1, 1], 0.5 * (6.0 - 0.0))


def test_cons_update_delayed_until_next_meeting():
    e = get_economy("A1")
    lr = QLearner(e, LearnerConfig(), np.random.default_rng(0), q_lr=0.5, eta=1.0)
    lr.QT[0, 1, 2, :] = [1.0, 3.0]
    lr.observe_cons(ConsRecord(agent=0, i=0, h=0, gamma=1, h_next=1, r=99.0))
    assert lr.QC[0, 0, 1] == 0.0                      # pending, not yet applied
    lr.observe_meeting_agent(agent=0, i=0, k=1, j_good=2)
    expect = 0.5 * (99.0 - 0.0 + 3.0 - 0.0)
    assert np.isclose(lr.QC[0, 0, 1], expect)
    assert np.isclose(lr.gbar[0], 0.5 * (99.0 + 3.0))


def test_dry_run_completes_all_econs():
    """Step 3: `run_learning` with `QLearner` on A1/C (and D for parity with
    MuZeroLearner's own dry-run test) completes and deployment arrays respect
    masks -- infeasible cells exactly 0, feasible cells a deterministic 0/1
    (greedy_policy is required to be hard, per the brief's interface line).
    """
    for name in ["A1", "C", "D"]:
        e = get_economy(name)
        rng = np.random.default_rng(0)
        lr = QLearner(e, DRY, rng)
        res = run_learning(e, lr, DRY, seed=0)
        assert np.isfinite(res.alpha_star).all()
        assert np.isfinite(res.beta_star).all()
        assert set(np.unique(res.alpha_star[e.lambda_feasible])) <= {0.0, 1.0}
        assert set(np.unique(res.beta_star[e.gamma_feasible])) <= {0.0, 1.0}
        assert (res.alpha_star[~e.lambda_feasible] == 0.0).all()
        assert (res.beta_star[~e.gamma_feasible] == 0.0).all()


def test_policies_respect_clip_and_masks():
    e = get_economy("C")
    lr = QLearner(e, DRY, np.random.default_rng(1))
    res = run_learning(e, lr, DRY, seed=1)
    assert (res.beta_star[:, 0] == 0).all()          # fiat never consumable
