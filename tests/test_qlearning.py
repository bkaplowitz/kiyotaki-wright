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


def test_cons_update_same_good_bootstraps_off_qc():
    """Same-good meeting (k == j_good): (i, k, j_good) is not a trade root --
    lambda_feasible requires k != j, so QT[i, k, j_good, :] is the -inf
    infeasibility mask. The pending consumption TD must instead bootstrap off
    max_gamma QC[i, k, :]."""
    e = get_economy("A1")
    assert not e.lambda_feasible[0, 1, 1]              # k == j_good == 1 is same-good, infeasible
    lr = QLearner(e, LearnerConfig(), np.random.default_rng(0), q_lr=0.5, eta=1.0)
    lr.QC[0, 1, :] = [2.0, 5.0]
    lr.observe_cons(ConsRecord(agent=0, i=0, h=0, gamma=1, h_next=1, r=99.0))
    assert lr.QC[0, 0, 1] == 0.0                        # pending, not yet applied
    lr.observe_meeting_agent(agent=0, i=0, k=1, j_good=1)  # same-good: k == j_good
    expect = 0.5 * (99.0 - 0.0 + 5.0 - 0.0)
    assert np.isclose(lr.QC[0, 0, 1], expect)
    assert np.isclose(lr.gbar[0], 0.5 * (99.0 + 5.0))
    assert np.isfinite(lr.QC).all()                     # canary: no -inf poisoning from the QT diagonal


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


def test_eps_start_decays_geometrically_to_floor():
    """Task 18 decay arm: `eps_start` anneals exploration epsilon on the zeta
    schedule shape, eps_n = max(eps_floor, eps_start * zeta_decay**(n-1));
    default (eps_start=None) keeps today's constant eps == cfg.eps_floor."""
    e = get_economy("A1")
    cfg = LearnerConfig()
    lr = QLearner(e, cfg, np.random.default_rng(0), eps_start=0.5)
    assert lr.eps == 0.5                              # iteration 1 explores at eps_start
    lr.end_iteration(1)
    assert np.isclose(lr.eps, 0.5 * cfg.zeta_decay)   # iteration 2's epsilon
    for n in range(2, 120):
        lr.end_iteration(n)
        assert lr.eps >= cfg.eps_floor
    assert lr.eps == cfg.eps_floor                    # floored, never below

    default = QLearner(e, cfg, np.random.default_rng(0))
    assert default.eps == cfg.eps_floor
    default.end_iteration(1)
    assert default.eps == cfg.eps_floor               # flag off: byte-identical constant


def test_q_lr_visit_decays_trade_step_per_cell():
    """Task 18 decay arm: `q_lr_visit` scales each real TD update by
    1/sqrt(n_visits) of the updated cell; flag off keeps the constant step."""
    e = get_economy("A1")
    rec = TradeRecord(agent=0, i=0, k=0, j=1, lam=1, h_next=1, r=0.0)

    lr = QLearner(e, LearnerConfig(), np.random.default_rng(0), q_lr=0.5, q_lr_visit=True)
    lr.QC[0, 1, :] = [2.0, 6.0]
    lr.observe_trade(rec)                             # visit 1: step 0.5/sqrt(1)
    assert np.isclose(lr.QT[0, 0, 1, 1], 0.5 * 6.0)
    lr.observe_trade(rec)                             # visit 2: step 0.5/sqrt(2)
    assert np.isclose(lr.QT[0, 0, 1, 1], 3.0 + (0.5 / np.sqrt(2.0)) * (6.0 - 3.0))

    const = QLearner(e, LearnerConfig(), np.random.default_rng(0), q_lr=0.5)
    const.QC[0, 1, :] = [2.0, 6.0]
    const.observe_trade(rec)
    const.observe_trade(rec)                          # flag off: constant 0.5 both times
    assert np.isclose(const.QT[0, 0, 1, 1], 3.0 + 0.5 * 3.0)


def test_q_lr_visit_decays_cons_step_and_gbar_together():
    """The consumption-stage TD and its gbar move share one decayed step, as in
    the constant-step rule gbar += eta * a * delta_C."""
    e = get_economy("A1")
    lr = QLearner(e, LearnerConfig(), np.random.default_rng(0), q_lr=0.5, eta=1.0,
                  q_lr_visit=True)
    lr.QT[0, 1, 2, :] = [1.0, 3.0]
    lr.observe_cons(ConsRecord(agent=0, i=0, h=0, gamma=1, h_next=1, r=99.0))
    lr.observe_meeting_agent(agent=0, i=0, k=1, j_good=2)  # visit 1: step 0.5
    assert np.isclose(lr.QC[0, 0, 1], 0.5 * (99.0 - 0.0 + 3.0 - 0.0))
    lr.gbar[0] = 0.0                                  # isolate the second step size
    lr.observe_cons(ConsRecord(agent=0, i=0, h=0, gamma=1, h_next=1, r=99.0))
    lr.observe_meeting_agent(agent=0, i=0, k=1, j_good=2)  # visit 2: step 0.5/sqrt(2)
    step2 = 0.5 / np.sqrt(2.0)
    delta2 = 99.0 - 0.0 + 3.0 - 51.0
    assert np.isclose(lr.QC[0, 0, 1], 51.0 + step2 * delta2)
    assert np.isclose(lr.gbar[0], step2 * delta2)
