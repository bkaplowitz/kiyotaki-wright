import numpy as np
from kw.economies import get_economy
from kw.environment import reward_cons_table, next_holding_cons, trade_outcome

def test_reward_a1():
    e = get_economy("A1")
    r = reward_cons_table(e)
    assert np.isclose(r[0, 0, 1], 100.0 - 1.0)
    assert np.isclose(r[0, 2, 0], -20.0)
    assert np.isclose(r[2, 2, 1], 100.0 - 0.1)

def test_fiat_not_consumable():
    r = reward_cons_table(get_economy("C"))
    assert np.isneginf(r[:, 0, 1]).all()

def test_c_consume_idx_not_type_idx():
    # consume_idx[0] == 1 (not the type index 0); reward must key off consume_idx.
    e = get_economy("C")
    r = reward_cons_table(e)
    assert np.isclose(r[0, 1, 1], 100.0 - e.s[e.f[0]])

def test_next_holding():
    e = get_economy("A1")
    nh = next_holding_cons(e)
    assert nh[0, 0, 1] == e.f[0] and nh[0, 2, 0] == 2

def test_trade_outcome():
    assert trade_outcome(0, 1, 1, 1) == (1, 0)
    assert trade_outcome(0, 1, 1, 0) == (0, 1)
    assert trade_outcome(1, 1, 1, 1) == (1, 1)
