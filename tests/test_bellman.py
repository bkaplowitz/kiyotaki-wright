import numpy as np
from kw.economies import get_economy
from kw.distributions import reciprocation, invariant
from kw.bellman import rvi_solve, greedy_policy, policy_eval, rvi_solve_subjective

def test_no_trade_gain_is_minus_storage():
    e = get_economy("A1")
    agg = np.full(3, 1 / 3)
    Q = np.zeros((3, 3))
    alpha_i = np.zeros((3, 3))
    beta_i = np.zeros(3)
    beta_i[e.consume_idx[0]] = 1.0
    g, V = policy_eval(e, 0, alpha_i, beta_i, agg, Q)
    assert np.isclose(g, -e.s[e.f[0]], atol=1e-9)     # type 1 stores good 2 forever: g = -1.0

def test_greedy_is_fundamental_at_fundamental_mu():
    e = get_economy("A1")
    from tests.test_distributions import _fundamental_a1
    alpha, beta = _fundamental_a1(e)
    mu0 = np.zeros((3, 3))
    mu0[np.arange(3), e.f] = 1.0
    mu = invariant(e, mu0, alpha, beta)
    agg = (e.nu[:, None] * mu).sum(axis=0)
    Q = reciprocation(e, mu, alpha)
    for i in range(3):
        g, V, W = rvi_solve(e, i, agg, Q)
        a_i, b_i = greedy_policy(e, i, Q, V, W)
        g_pol, _ = policy_eval(e, i, a_i, b_i, agg, Q)
        assert np.isclose(g, g_pol, atol=1e-8)
        assert b_i[e.consume_idx[i]] == 1.0

def test_gain_dominates_any_policy():
    e = get_economy("B")
    rng = np.random.default_rng(1)
    mu = rng.dirichlet(np.ones(3), size=3)
    agg = (e.nu[:, None] * mu).sum(axis=0)
    alpha_rand = rng.uniform(0, 1, (3, 3, 3)) * e.lambda_feasible
    Q = reciprocation(e, mu, alpha_rand)
    g_star, V, W = rvi_solve(e, 0, agg, Q)
    for _ in range(5):
        a = rng.integers(0, 2, (3, 3)) * e.lambda_feasible[0]
        b = rng.integers(0, 2, 3).astype(float)
        g_pol, _ = policy_eval(e, 0, a.astype(float), b, agg, Q)
        assert g_star >= g_pol - 1e-8

def test_subjective_no_reciprocation_means_no_trade_value():
    e = get_economy("A1")
    m = np.full(3, 1 / 3)
    q = np.zeros((3, 3))
    g, V, W = rvi_solve_subjective(e, 0, m, q)
    assert np.isclose(g, -e.s[e.f[0]], atol=1e-8)     # skeptical beliefs → autarky value
