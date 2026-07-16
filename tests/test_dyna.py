import numpy as np
import pytest

from kw.economies import get_economy
from kw.learners.base import ConsRecord, LearnerConfig, TradeRecord
from kw.extensions.qlearning import QLearner
from kw.simulate import run_learning

DRY = LearnerConfig(n_agents=60, rounds=5, n_iters_max=6, n_min=2, patience=2)


def test_unknown_dyna_raises():
    e = get_economy("A1")
    with pytest.raises(ValueError):
        QLearner(e, LearnerConfig(), np.random.default_rng(0), dyna="bogus")


def test_per_agent_still_not_implemented_with_dyna():
    e = get_economy("A1")
    with pytest.raises(NotImplementedError):
        QLearner(e, LearnerConfig(), np.random.default_rng(0), dyna="emp", per_agent=True)


def test_trade_backup_matches_hand_computed_target():
    """Deterministic model, dyna_k=1: the single planning backup at the only
    visited root (0, 0, 1) must match the hand-computed expected target
    r_hat^T + sum_h' P_hat^T(h'|k,j,lam) * max_gamma QC[i,h',gamma]."""
    e = get_economy("A1")
    rng = np.random.default_rng(0)
    lr = QLearner(e, LearnerConfig(), rng, dyna="emp", q_lr=0.5, dyna_k=1)

    lr.trade_action(0, 0, 1, rng)  # seeds the dyna pool with exactly root (0, 0, 1)

    lr.model._N_T[0, 0, 1, 0, :] = [1000.0, 0.0, 0.0]  # lam=0: deterministic h'=0
    lr.model._N_T[0, 0, 1, 1, :] = [0.0, 1000.0, 0.0]  # lam=1: deterministic h'=1
    lr.QC[0] = np.array([[1.0, 2.0], [3.0, 9.0], [5.0, 0.0]])  # qc_max = [2.0, 9.0, 5.0]

    trigger = TradeRecord(agent=0, i=0, k=1, j=2, lam=0, h_next=1, r=0.0)
    lr.observe_trade(trigger)  # unrelated root (0, 1, 2); triggers exactly one planning backup

    assert np.isclose(lr.QT[0, 0, 1, 0], 0.5 * 2.0)
    assert np.isclose(lr.QT[0, 0, 1, 1], 0.5 * 9.0)


def test_cons_backup_same_good_bootstraps_off_qc():
    """Deterministic model, dyna_k=1, eta=0 (gbar pinned at 0 so it does not
    confound the target): the cons-root backup's simulated next-meeting good
    j ~ m_hat_i is deterministically the same as the simulated post-
    consumption holding h''=1, so the expected backup must substitute
    max_gamma QC[0,1,:] for the permanently -inf max_lambda QT[0,1,1,:]
    diagonal (decision 4d) -- mirrors test_qlearning's
    test_cons_update_same_good_bootstraps_off_qc for the expected-backup case."""
    e = get_economy("A1")
    assert not e.lambda_feasible[0, 1, 1]
    rng = np.random.default_rng(0)
    lr = QLearner(e, LearnerConfig(), rng, dyna="emp", q_lr=0.4, eta=0.0, dyna_k=1)

    lr.cons_action(0, 0, rng)  # seeds the dyna pool with exactly cons root (0, 0)

    lr.model._C[0, :] = [0.0, 1000.0, 0.0]  # m_hat(0) deterministic at j=1
    lr.model._N_C[0, 0, 0, :] = [0.0, 1000.0, 0.0]  # p_cons(0)[h=0,gamma=0] -> h''=1
    lr.model._N_C[0, 0, 1, :] = [0.0, 1000.0, 0.0]  # p_cons(0)[h=0,gamma=1] -> h''=1
    lr.model._S_C[0, 0, 0] = 5.0
    lr.model._B_C[0, 0, 0] = 1.0  # r_cons(0)[0,0] = 5.0
    lr.model._S_C[0, 0, 1] = 20.0
    lr.model._B_C[0, 0, 1] = 1.0  # r_cons(0)[0,1] = 20.0
    lr.QC[0, 1, :] = [2.0, 6.0]  # max_gamma QC[0,1,:] = 6.0; QT[0,1,1,:] stays the -inf diagonal

    trigger = ConsRecord(agent=0, i=0, h=1, gamma=0, h_next=1, r=0.0)
    lr.observe_cons(trigger)
    lr.observe_meeting_agent(agent=0, i=0, k=1, j_good=1)  # completes trigger's own TD, then plans

    assert np.isclose(lr.QC[0, 0, 0], 0.4 * (5.0 + 6.0))
    assert np.isclose(lr.QC[0, 0, 1], 0.4 * (20.0 + 6.0))
    assert np.isfinite(lr.QC).all()  # canary: no -inf poisoning from the QT diagonal


def test_rules_backend_trade_backup_interpolates_by_qhat():
    """dyna="rules": p_trade(i)[k,j,1,:] = (1-q_hat)*onehot(k) + q_hat*onehot(j),
    so the lam=1 backup target must land strictly between the lam=0 target
    (qc_max[k], q_hat-independent) and the full-acceptance value qc_max[j],
    proportional to q_hat."""
    e = get_economy("A1")
    rng = np.random.default_rng(0)
    lr = QLearner(e, LearnerConfig(), rng, dyna="rules", q_lr=0.5, dyna_k=1)

    lr.trade_action(0, 0, 1, rng)  # seeds the dyna pool with exactly root (0, 0, 1)

    lr.model._O[0, 1] = 10.0
    lr.model._A_acc[0, 1] = 3.0  # q_hat[0,1] = 0.3
    lr.QC[0] = np.array([[1.0, 4.0], [9.0, 2.0], [0.0, 0.0]])  # qc_max = [4.0, 9.0, 0.0]

    trigger = TradeRecord(agent=0, i=0, k=1, j=2, lam=0, h_next=1, r=0.0)
    lr.observe_trade(trigger)  # unrelated root (0, 1, 2); triggers exactly one planning backup

    qc_max_k, qc_max_j = 4.0, 9.0
    raw_target1 = 0.7 * qc_max_k + 0.3 * qc_max_j
    assert qc_max_k < raw_target1 < qc_max_j
    assert np.isclose(lr.QT[0, 0, 1, 0], 0.5 * qc_max_k)
    assert np.isclose(lr.QT[0, 0, 1, 1], 0.5 * raw_target1)


def test_dyna_rules_feeds_offer_with_swapped_args():
    e = get_economy("A1")
    rng = np.random.default_rng(0)
    lr = QLearner(e, LearnerConfig(), rng, dyna="rules", q_lr=0.1, dyna_k=0)
    lr.observe_trade(TradeRecord(agent=0, i=0, k=0, j=1, lam=1, h_next=1, r=0.0))
    assert lr.model._O[1, 0] > 2.0 and lr.model._A_acc[1, 0] > 1.0  # swapped direction fed
    assert lr.model._O[0, 1] == 2.0 and lr.model._A_acc[0, 1] == 1.0  # un-swapped direction untouched


def test_dyna_emp_never_feeds_offer():
    e = get_economy("A1")
    rng = np.random.default_rng(0)
    lr = QLearner(e, LearnerConfig(), rng, dyna="emp", q_lr=0.1, dyna_k=0)
    lr.observe_trade(TradeRecord(agent=0, i=0, k=0, j=1, lam=1, h_next=1, r=0.0))
    assert (lr.model._O == 2.0).all()
    assert (lr.model._A_acc == 1.0).all()


def test_planning_uses_updated_kernel_after_model_observation():
    """Kernel-staleness guard: a planning backup that runs after a model
    observation must see the model's *current* kernels, not a snapshot taken
    before the observation. The first trigger backs root (0, 0, 1) up under a
    deterministic kernel A (h'=0 / h'=1); the counts are then hand-flipped to
    kernel B (h'=2 for both lam) and the identical trigger record is fed
    again -- the second backup's target must be qc_max[2] = 5.0 for both lam.
    A cache that survives `observe_trade`'s model feed would replay kernel A
    (targets 2.0 / 9.0) and fail both closing asserts."""
    e = get_economy("A1")
    rng = np.random.default_rng(0)
    lr = QLearner(e, LearnerConfig(), rng, dyna="emp", q_lr=0.5, dyna_k=1)

    lr.trade_action(0, 0, 1, rng)  # seeds the dyna pool with exactly root (0, 0, 1)
    lr.QC[0] = np.array([[1.0, 2.0], [3.0, 9.0], [5.0, 0.0]])  # qc_max = [2.0, 9.0, 5.0]
    trigger = TradeRecord(agent=0, i=0, k=1, j=2, lam=0, h_next=1, r=0.0)

    lr.model._N_T[0, 0, 1, 0, :] = [1000.0, 0.0, 0.0]  # kernel A, lam=0: h'=0
    lr.model._N_T[0, 0, 1, 1, :] = [0.0, 1000.0, 0.0]  # kernel A, lam=1: h'=1
    lr.observe_trade(trigger)
    assert np.isclose(lr.QT[0, 0, 1, 0], 0.5 * 2.0)
    assert np.isclose(lr.QT[0, 0, 1, 1], 0.5 * 9.0)

    lr.model._N_T[0, 0, 1, 0, :] = [0.0, 0.0, 1000.0]  # kernel B, lam=0: h'=2
    lr.model._N_T[0, 0, 1, 1, :] = [0.0, 0.0, 1000.0]  # kernel B, lam=1: h'=2
    lr.observe_trade(trigger)
    assert np.isclose(lr.QT[0, 0, 1, 0], 0.5 * (0.5 * 2.0) + 0.5 * 5.0)
    assert np.isclose(lr.QT[0, 0, 1, 1], 0.5 * (0.5 * 9.0) + 0.5 * 5.0)


@pytest.mark.parametrize("dyna", ["emp", "rules"])
def test_dyna_dry_run_completes_all_econs(dyna):
    for name in ["A1", "C", "D"]:
        e = get_economy(name)
        rng = np.random.default_rng(0)
        lr = QLearner(e, DRY, rng, dyna=dyna, dyna_k=5)
        res = run_learning(e, lr, DRY, seed=0)
        assert np.isfinite(res.alpha_star).all()
        assert np.isfinite(res.beta_star).all()
        assert set(np.unique(res.alpha_star[e.lambda_feasible])) <= {0.0, 1.0}
        assert set(np.unique(res.beta_star[e.gamma_feasible])) <= {0.0, 1.0}
        assert (res.alpha_star[~e.lambda_feasible] == 0.0).all()
        assert (res.beta_star[~e.gamma_feasible] == 0.0).all()
