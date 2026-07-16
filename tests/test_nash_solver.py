import numpy as np
from kw.economies import get_economy
from kw.prompts import nash_prompt
from kw.nash_solver import solve_nash


def test_a1_default_prompt_reaches_fundamental():
    e = get_economy("A1")
    res = solve_nash(e, *nash_prompt(e, "default"))
    assert res.converged and res.gain_gap_history[-1] <= 1e-9
    agg = (e.nu[:, None] * res.mu).sum(axis=0)
    assert np.allclose(agg, [0.5, 1 / 3, 1 / 6], atol=1e-6)
    assert res.alpha[1, 2, 0] > 0.99          # type 2 accepts good 1 (money) for good 3


def test_a2_speculative_prompt_stays_speculative():
    e = get_economy("A2")
    res = solve_nash(e, *nash_prompt(e, "speculative"))
    assert res.converged
    assert res.alpha[0, 1, 2] > 0.99          # type 1 offers good 2 for good 3


def test_a1_speculation_not_equilibrium():
    """Speculation fails as a store of value in A1 (ū=100), asserted on holdings.

    The paper's default Γ (p. 9) allows disposal, which makes the speculative arc
    an exact payoff-null tie, and policy "is not pinned down at off-path or
    indifferent trade states" (p. 57) — so we assert type 1 never carries good 3
    and holdings are fundamental, not one representative α selection at the tie.
    """
    e = get_economy("A1")
    res = solve_nash(e, *nash_prompt(e, "speculative"))
    assert res.converged
    assert res.mu[0, 2] < 1e-9                # type 1 never carries good 3
    agg = (e.nu[:, None] * res.mu).sum(axis=0)
    assert np.allclose(agg, [0.5, 1 / 3, 1 / 6], atol=1e-6)


def test_b_two_branches():
    e = get_economy("B")
    r1 = solve_nash(e, *nash_prompt(e, "default"))
    r2 = solve_nash(e, *nash_prompt(e, "speculative"))
    assert r1.converged and r2.converged
    assert not np.allclose(r1.alpha, r2.alpha, atol=0.1)


def test_c_money_prompt_fiat_valued():
    e = get_economy("C")
    res = solve_nash(e, *nash_prompt(e, "money"))
    assert res.converged
    assert res.alpha[1, 1, 0] > 0.99          # type 2 accepts fiat for its production good... trades


def test_d_converges():
    e = get_economy("D")
    res = solve_nash(e, *nash_prompt(e, "default"))
    assert res.converged and res.gain_gap_history[-1] <= 1e-9


def test_c_matches_fig8_fiat_holdings():
    e = get_economy("C")
    res = solve_nash(e, *nash_prompt(e, "money"))
    assert res.converged
    agg = (e.nu[:, None] * res.mu).sum(axis=0)
    assert abs(agg[0] - 0.32) < 1e-6, "paper-check-nash.md Fig 8: aggregate fiat share = 0.32"
    assert np.allclose(
        res.mu[:, 0], [0.26, 0.31, 0.38], atol=5e-3
    ), "paper-check-nash.md Fig 8: per-type fiat holdings 0.26/0.31/0.38 (printed to 2 dp)"


def test_d_matches_fig10_good1_share():
    e = get_economy("D")
    res = solve_nash(e, *nash_prompt(e, "default"))
    assert res.converged
    agg = (e.nu[:, None] * res.mu).sum(axis=0)
    assert abs(agg[0] - 0.612) < 5e-4, (
        "paper-check-nash.md Fig 10: good-1 aggregate share = 0.612 (printed to 3 dp)"
    )
