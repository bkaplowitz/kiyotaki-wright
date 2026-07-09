import numpy as np

from kw.economies import get_economy
from kw.prompts import nash_prompt, sce_prompt
from kw.nash_solver import solve_nash
from kw.sce_solver import solve_sce
from kw.diagnostics import br_gain, classify_run


def test_nash_classifies_nash():
    e = get_economy("A1")
    r = solve_nash(e, *nash_prompt(e, "default"))
    assert classify_run(e, r.alpha, r.beta, r.mu, None, None) == "NASH"


def test_autarky_sce_classifies_trap():
    e = get_economy("A1")
    r = solve_sce(e, *sce_prompt(e, "skeptical-autarky"))
    assert classify_run(e, r.alpha, r.beta, r.mu, r.m, r.q) == "SCE_TRAP"


def test_br_gain_zero_at_nash():
    e = get_economy("B")
    r = solve_nash(e, *nash_prompt(e, "default"))
    b, per = br_gain(e, r.alpha, r.beta, r.mu)
    assert b <= 1e-8 and per.shape == (3,)


def test_random_policy_at_nonstationary_mu_unsettled():
    e = get_economy("A1")
    alpha = 0.5 * e.lambda_feasible.astype(np.float64)
    beta = 0.5 * e.gamma_feasible.astype(np.float64)
    mu = np.zeros((e.n_types, e.n_goods), dtype=np.float64)
    mu[np.arange(e.n_types), e.f] = 1.0
    assert classify_run(e, alpha, beta, mu, None, None) == "UNSETTLED"
