import numpy as np
from kw.economies import get_economy
from kw.empirical_model import EmpiricalModel, ModelBank
from kw.learners.base import TradeRecord, ConsRecord

def test_priors():
    m = EmpiricalModel(get_economy("A1"))
    assert np.allclose(m.q_hat(), 0.5)                      # A_acc=κ0/2, O=κ0
    assert np.allclose(m.m_hat(), 1 / 3)
    assert np.allclose(m.p_trade(), 1 / 3)                  # uniform Dirichlet-style prior
    assert np.allclose(m.r_trade(), 0.0) and np.allclose(m.r_cons(), 0.0)

def test_counts_update():
    e = get_economy("A1"); m = EmpiricalModel(e)
    m.observe_offer(k=0, j=1, accepted=True)
    m.observe_offer(k=0, j=1, accepted=False)
    assert np.isclose(m.q_hat()[0, 1], (1.0 + 1.0) / (2.0 + 2.0))   # (κ0/2 + 1)/(κ0 + 2)
    m.observe_trade(TradeRecord(0, 0, 0, 1, 1, 1, 0.0))
    p = m.p_trade()
    assert p[0, 0, 1, 1].argmax() == 1

def test_cons_reward_average():
    e = get_economy("A1"); m = EmpiricalModel(e)
    m.observe_cons(ConsRecord(0, 0, 0, 1, e.f[0], 99.0))
    assert np.isclose(m.r_cons()[0, 0, 1], (0.0 * 2.0 + 99.0) / 3.0)  # (S+r)/(B+1), B0=κ0

def test_ema_equals_cumulative_at_zero_gain():
    e = get_economy("A1")
    m0, m1 = EmpiricalModel(e, ema_gain=0.0), EmpiricalModel(e, ema_gain=0.05)
    for _ in range(30):
        m0.observe_offer(0, 1, True); m1.observe_offer(0, 1, True)
        m0.observe_offer(0, 1, False); m1.observe_offer(0, 1, False)
    assert np.isclose(m0.q_hat()[0, 1], 0.5, atol=0.02)
    assert m1.q_hat()[0, 1] != m0.q_hat()[0, 1]              # EMA down-weights the κ0 prior faster
    assert 0.4 < m1.q_hat()[0, 1] < 0.6

def test_model_bank_shared_and_per_agent():
    e = get_economy("A1")
    shared = ModelBank(e, n_banks=1)
    assert shared.bank_for(0) is shared.bank_for(41)

    per_agent = ModelBank(e, n_banks=3)
    assert per_agent.bank_for(0) is per_agent[0]
    assert per_agent.bank_for(0) is not per_agent.bank_for(1)
    assert per_agent.bank_for(1) is not per_agent.bank_for(2)
