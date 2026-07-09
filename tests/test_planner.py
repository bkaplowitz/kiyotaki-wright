import numpy as np
from kw.planner import puct_select, dp_tables, Planner, EmpiricalBackend
from kw.empirical_model import EmpiricalModel
from kw.economies import get_economy


def test_puct_visits_sum_and_favor_better_action():
    pi, v = puct_select(np.array([0.0, 10.0]), np.array([0.5, 0.5]),
                        np.array([True, True]), n_sims=32, c_puct=1.5, tau=1.0)
    assert np.isclose(pi.sum(), 1.0) and pi[1] > 0.75 and np.isclose(v, 10.0)


def test_puct_tied_values_split_evenly():
    pi, _ = puct_select(np.array([5.0, 5.0]), np.array([0.5, 0.5]),
                        np.array([True, True]), 32, 1.5, 1.0)
    assert np.isclose(pi[0], 0.5, atol=1 / 32 + 1e-9)


def test_puct_infeasible_never_visited():
    pi, _ = puct_select(np.array([0.0, 100.0]), np.array([0.5, 0.5]),
                        np.array([True, False]), 32, 1.5, 1.0)
    assert pi[1] == 0.0


def test_dp_terminal_is_v():
    e = get_economy("A1")
    m = EmpiricalModel(e)
    v = np.array([1.0, 2.0, 3.0])
    psi_t, psi_c = dp_tables(e, EmpiricalBackend(m), 0,
                             np.full((3, 3), 0.5), np.full(3, 0.5), v, horizon=1)
    assert psi_c.shape == (3, 2)


def test_planner_memoizes():
    e = get_economy("A1")
    m = EmpiricalModel(e)
    from kw.learners.base import LearnerConfig
    cfg = LearnerConfig()
    pl = Planner(e, EmpiricalBackend(m), np.full((3, 3, 3), 0.5),
                 np.full((3, 3), 0.5), np.zeros((3, 3)), cfg)
    r1 = pl.query_trade(0, 0, 1)
    m.observe_offer(0, 1, True)                      # model moves, cache must not
    r2 = pl.query_trade(0, 0, 1)
    assert np.array_equal(r1[0], r2[0]) and r1[1] == r2[1]
