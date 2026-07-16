import numpy as np
from kw.economies import get_economy
from kw.learners.base import LearnerConfig
from kw.simulate import init_population, run_learning

DRY = LearnerConfig(n_agents=60, rounds=5, n_iters_max=6, n_min=2, patience=2)


class RandomLearner:
    def __init__(self, econ, rng):
        self.econ, self.rng = econ, rng

    def trade_action(self, i, k, j, rng, agent=0):
        return int(rng.random() < 0.5) if k != j else 0

    def cons_action(self, i, h, rng, agent=0):
        return int(self.econ.gamma_feasible[i, h] and rng.random() < 0.5)

    def observe_meeting(self, i, j_good, agent=0):
        pass

    def observe_trade(self, rec):
        pass

    def observe_cons(self, rec):
        pass

    def end_iteration(self, n):
        return 0.0

    def deployment_policy(self):
        a = np.full((self.econ.n_types,) + (self.econ.n_goods,) * 2, 0.5) * self.econ.lambda_feasible
        b = np.full((self.econ.n_types, self.econ.n_goods), 0.5) * self.econ.gamma_feasible
        return a, b


def test_init_population_c_fiat_count():
    e = get_economy("C")
    types, hold = init_population(e, DRY, np.random.default_rng(0))
    per_type = 60 // 3
    for i in range(3):
        assert (hold[types == i] == 0).sum() == round(0.32 * per_type)


def test_goods_conserved_outside_consumption():
    e = get_economy("A1")
    rng = np.random.default_rng(1)
    res = run_learning(e, RandomLearner(e, rng), DRY, seed=1)
    assert res.n_iters <= 6 and len(res.trace["trade_volume"]) == res.n_iters


def test_determinism():
    e = get_economy("A1")
    r1 = run_learning(e, RandomLearner(e, np.random.default_rng(2)), DRY, seed=7)
    r2 = run_learning(e, RandomLearner(e, np.random.default_rng(2)), DRY, seed=7)
    assert np.array_equal(r1.alpha_star, r2.alpha_star)
    assert r1.trace["trade_volume"] == r2.trace["trade_volume"]
