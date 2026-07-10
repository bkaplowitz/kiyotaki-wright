import numpy as np
import pytest
from kw.economies import get_economy
from kw.learners.base import LearnerConfig
from kw.learners.muzero import MuZeroLearner
from kw.simulate import run_learning
from kw.diagnostics import br_gain

DRY = LearnerConfig(n_agents=60, rounds=5, n_iters_max=6, n_min=2, patience=2)


def test_dry_run_completes_all_econs():
    for name in ["A1", "C", "D"]:
        e = get_economy(name)
        rng = np.random.default_rng(0)
        lr = MuZeroLearner(e, DRY, rng)
        res = run_learning(e, lr, DRY, seed=0)
        assert np.isfinite(res.alpha_star).all()
        assert np.isfinite(res.beta_star).all()
        assert (res.alpha_star >= 0.0).all() and (res.alpha_star <= 1.0).all()
        assert (res.beta_star >= 0.0).all() and (res.beta_star <= 1.0).all()
        assert (res.alpha_star[~e.lambda_feasible] == 0.0).all()
        assert (res.beta_star[~e.gamma_feasible] == 0.0).all()


def test_policies_respect_clip_and_masks():
    e = get_economy("C")
    lr = MuZeroLearner(e, DRY, np.random.default_rng(1))
    res = run_learning(e, lr, DRY, seed=1)
    assert (res.beta_star[:, 0] == 0).all()          # fiat never consumable


def test_per_agent_banks_route_by_true_agent_id():
    """E2 (Task 18): per_agent=True gives every agent a private model bank
    (n_banks = cfg.n_agents) and observations route by the true agent id --
    feeding agent 5 must move bank 5 and no other."""
    e = get_economy("A1")
    lr = MuZeroLearner(e, DRY, np.random.default_rng(0), per_agent=True)
    assert lr.bank.n_banks == DRY.n_agents
    assert lr.bank[5] is not lr.bank[4]

    before = lr.bank[4].m_hat().copy()
    lr.observe_meeting(0, 1, agent=5)
    assert not np.allclose(lr.bank[5].m_hat(), before)   # bank 5 moved
    assert np.array_equal(lr.bank[4].m_hat(), before)    # bank 4 untouched

    from kw.learners.base import TradeRecord

    rec = TradeRecord(agent=7, i=0, k=0, j=1, lam=1, h_next=1, r=0.0)
    nt_before = lr.bank[6].p_trade().copy()
    lr.observe_trade(rec)
    assert not np.allclose(lr.bank[7].p_trade(), nt_before)
    assert np.array_equal(lr.bank[6].p_trade(), nt_before)


def test_per_agent_dry_run_completes_with_exact_per_bank_counts():
    """E2 end-to-end: `simulate.run_learning` threads true agent ids, so each
    private bank receives exactly one meeting observation per round (perfect
    matching), and deployment stays finite and mask-respecting."""
    e = get_economy("A1")
    lr = MuZeroLearner(e, DRY, np.random.default_rng(0), per_agent=True)
    res = run_learning(e, lr, DRY, seed=0)

    prior_mass = float(e.n_types * e.n_goods)            # C starts at 1.0 per cell
    expected = DRY.rounds * res.n_iters
    for bank in lr.bank.banks:
        assert bank._C.sum() - prior_mass == expected

    assert np.isfinite(res.alpha_star).all()
    assert np.isfinite(res.beta_star).all()
    assert (res.alpha_star >= 0.0).all() and (res.alpha_star <= 1.0).all()
    assert (res.alpha_star[~e.lambda_feasible] == 0.0).all()
    assert (res.beta_star[~e.gamma_feasible] == 0.0).all()


@pytest.mark.slow
def test_a1_full_run_reaches_nash():
    e = get_economy("A1")
    cfg = LearnerConfig()
    lr = MuZeroLearner(e, cfg, np.random.default_rng(0))
    res = run_learning(e, lr, cfg, seed=0)
    b, _ = br_gain(e, res.alpha_star, res.beta_star, res.mu_star)
    assert b < 0.1                                    # paper: 1.5e-12; seed slack allowed
    agg = (e.nu[:, None] * res.mu_star).sum(axis=0)
    assert abs(agg[0] - 0.5) < 0.05
