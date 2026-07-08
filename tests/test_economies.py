import numpy as np
import pytest
from kw.economies import get_economy, ECONOMIES

def test_registry_names():
    assert set(ECONOMIES) == {"A1", "A2", "B", "C", "D"}

@pytest.mark.parametrize("name,n_goods,n_types", [
    ("A1", 3, 3), ("A2", 3, 3), ("B", 3, 3), ("C", 4, 3), ("D", 5, 5)])
def test_shapes(name, n_goods, n_types):
    e = get_economy(name)
    assert e.n_goods == n_goods and e.n_types == n_types
    assert e.nu.shape == (n_types,) and np.isclose(e.nu.sum(), 1.0)
    assert e.s.shape == (n_goods,) and e.f.shape == (n_types,)

def test_a1_values():
    e = get_economy("A1")
    assert np.allclose(e.s, [0.1, 1.0, 20.0])
    assert np.allclose(e.ubar, [100.0] * 3)
    assert list(e.f) == [1, 2, 0]            # type i produces f(i): 2,3,1 → idx 1,2,0
    assert list(e.consume_idx) == [0, 1, 2]

def test_a2_only_ubar_differs():
    a1, a2 = get_economy("A1"), get_economy("A2")
    assert np.allclose(a2.ubar, [500.0] * 3)
    assert np.allclose(a1.s, a2.s) and list(a1.f) == list(a2.f)

def test_b_values():
    e = get_economy("B")
    assert list(e.f) == [2, 0, 1] and np.allclose(e.s, [1.0, 4.0, 9.0])

def test_c_fiat():
    e = get_economy("C")
    assert e.fiat_idx == 0 and e.M0 == 0.32
    assert np.allclose(e.s, [0.0, 9.0, 14.0, 29.0])
    assert list(e.consume_idx) == [1, 2, 3] and list(e.f) == [2, 3, 1]
    assert not e.gamma_feasible[:, 0].any()

def test_d_values():
    e = get_economy("D")
    assert list(e.f) == [2, 3, 4, 0, 1] and np.allclose(e.ubar, [200.0] * 5)
    assert np.allclose(e.s, [1.0, 4.0, 9.0, 20.0, 30.0])

def test_same_good_trade_infeasible():
    e = get_economy("A1")
    assert not e.lambda_feasible[0, 1, 1]
    assert e.lambda_feasible[0, 1, 0]
