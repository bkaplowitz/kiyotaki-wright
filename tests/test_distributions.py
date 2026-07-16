import numpy as np
from kw.economies import get_economy
from kw.distributions import reciprocation, one_step, invariant

def _fundamental_a1(e):
    alpha = np.zeros((3, 3, 3))
    beta = np.zeros((3, 3))
    for i in range(3):
        beta[i, e.consume_idx[i]] = 1.0
        for k in range(3):
            for j in range(3):
                if k == j:
                    continue
                good_for_me = (j == e.consume_idx[i])
                cheaper = e.s[j] < e.s[k]
                alpha[i, k, j] = float(good_for_me or cheaper)
    return alpha, beta

def test_one_step_conserves_mass():
    e = get_economy("A1")
    alpha, beta = _fundamental_a1(e)
    mu = np.full((3, 3), 1 / 3)
    mu2 = one_step(e, mu, alpha, beta)
    assert np.allclose(mu2.sum(axis=1), 1.0) and (mu2 >= -1e-15).all()

def test_invariant_fundamental_a1():
    e = get_economy("A1")
    alpha, beta = _fundamental_a1(e)
    mu0 = np.zeros((3, 3))
    mu0[np.arange(3), e.f] = 1.0
    mu = invariant(e, mu0, alpha, beta)
    agg = (e.nu[:, None] * mu).sum(axis=0)
    assert np.allclose(agg, [0.5, 1 / 3, 1 / 6], atol=1e-9)
    assert np.allclose(mu[0], [0, 1, 0], atol=1e-9)
    assert np.allclose(mu[1], [0.5, 0, 0.5], atol=1e-9)
    assert np.allclose(mu[2], [1, 0, 0], atol=1e-9)

def test_q_zero_on_unheld_good():
    e = get_economy("A1")
    alpha, beta = _fundamental_a1(e)
    mu = np.zeros((3, 3))
    mu[:, 0] = 1.0
    Q = reciprocation(e, mu, alpha)
    assert (Q[:, 1] == 0).all() and (Q[:, 2] == 0).all()

def test_fiat_conserved_in_c():
    e = get_economy("C")
    rng = np.random.default_rng(0)
    alpha = rng.uniform(0, 1, (3, 4, 4)) * e.lambda_feasible
    beta = rng.uniform(0, 1, (3, 4)) * e.gamma_feasible
    mu0 = np.zeros((3, 4))
    mu0[:, 0] = 0.32
    mu0[np.arange(3), e.f] = 0.68
    mu = mu0
    for _ in range(50):
        mu = one_step(e, mu, alpha, beta)
        assert np.isclose((e.nu * mu[:, 0]).sum(), 0.32 * 1.0, atol=1e-12)
