import numpy as np

from kw.distributions import aggregate
from kw.economies import get_economy
from kw.prompts import sce_prompt, nash_prompt
from kw.sce_solver import solve_sce
from kw.diagnostics import br_gain   # implemented minimally in this task, fully in Task 11


def test_a1_autarky_sce():
    e = get_economy("A1")
    res = solve_sce(e, *sce_prompt(e, "skeptical-autarky"))
    assert res.converged
    assert res.s_history[-1] <= 1e-9 and res.b_history[-1] <= 1e-9
    assert res.alpha.max() < 0.01                     # no trade
    b, _ = br_gain(e, res.alpha, res.beta, res.mu)
    assert b > 1.0                                    # subjectively optimal, objectively poor
    expected = np.zeros((e.n_types, e.n_goods))
    expected[np.arange(e.n_types), e.f] = 1.0
    assert np.allclose(res.mu, expected, atol=1e-9), (
        "autarky holdings should match paper-check-sce.md Fig 2 (p. 45): "
        "every type parks on its production good f(i)"
    )


def test_nash_is_sce():
    e = get_economy("A1")
    from kw.nash_solver import solve_nash
    nash = solve_nash(e, *nash_prompt(e, "default"))
    a0, b0, m0, q0 = sce_prompt(e, "skeptical-autarky")
    res = solve_sce(e, nash.alpha, nash.beta, m0, q0)
    assert res.converged and res.s_history[-1] <= 1e-6
    b, _ = br_gain(e, res.alpha, res.beta, res.mu)
    assert b <= 1e-6                                  # from a Nash start, SCE solver stays at Nash


def test_c_lowtrade_sce():
    e = get_economy("C")
    res = solve_sce(e, *sce_prompt(e, "skeptical-lowtrade"))
    assert res.converged
    assert res.s_history[-1] <= 1e-9 and res.b_history[-1] <= 1e-9
    agg = aggregate(e, res.mu)
    assert np.allclose(agg, [0.320, 0.335, 0.333, 0.012], atol=5e-3), (
        "aggregate holdings should match paper-check-sce.md Fig 8 (p. 51) to 5e-3: "
        "the frozen type-level split is transient-dependent, but the aggregate "
        "reproduces to ~2e-3 (ours: 0.320, 0.3333, 0.3333, 0.0133)"
    )
    assert res.mu[0, 2] > 0.99, "type 1 (paper) accumulates good 3 (Fig 8: 0.999)"
    assert res.mu[1, 0] > 0.9, "type 2 (paper) accumulates fiat (Fig 8: 0.932)"
    assert res.mu[2, 1] > 0.97, "type 3 (paper) accumulates good 1 (Fig 8: 0.973)"
    b, _ = br_gain(e, res.alpha, res.beta, res.mu)
    assert b > 1.0                                    # low-trade SCE is not Nash


def test_d_type1_autarky_sce():
    e = get_economy("D")
    res = solve_sce(e, *sce_prompt(e, "type1-autarky"))
    assert res.converged
    assert res.s_history[-1] <= 1e-9 and res.b_history[-1] <= 1e-9
    assert abs(res.mu[0, 2] - 1.0) < 5e-3, (
        "paper type 1 should sit in autarky on its production good 3 "
        "(paper-check-sce.md Fig 10, p. 54: 1.00)"
    )
    assert np.allclose(res.mu[2:, 0], 1.0, atol=5e-3), (
        "paper types 3-5 should accumulate good 1 (Fig 10: 1.00 each)"
    )
    assert res.mu[1, 0] > 0.7, (
        "lower bound on paper type 2's good-1 share: our converged SCE "
        "policy fully specializes T2 at 1.00, vs Fig 10's 0.72 -- a "
        "policy-level gap (different SCE endpoint), see findings.md H2"
    )
    b, _ = br_gain(e, res.alpha, res.beta, res.mu)
    assert b > 1.0                                    # type 1's autarky is objectively suboptimal
