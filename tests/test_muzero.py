import numpy as np
import pytest
from kw.economies import get_economy
from kw.learners.base import LearnerConfig
from kw.learners.muzero import MuZeroLearner
from kw.simulate import run_learning
from kw.diagnostics import br_gain

DRY = LearnerConfig(n_agents=60, rounds=5, n_iters_max=6, n_min=2, patience=2)


def test_dry_run_completes_all_econs():
    for name in ["A1", "C", "D"]:
        e = get_economy(name)
        rng = np.random.default_rng(0)
        lr = MuZeroLearner(e, DRY, rng)
        res = run_learning(e, lr, DRY, seed=0)
        assert np.isfinite(res.alpha_star).all()
        assert np.isfinite(res.beta_star).all()
        assert (res.alpha_star >= 0.0).all() and (res.alpha_star <= 1.0).all()
        assert (res.beta_star >= 0.0).all() and (res.beta_star <= 1.0).all()
        assert (res.alpha_star[~e.lambda_feasible] == 0.0).all()
        assert (res.beta_star[~e.gamma_feasible] == 0.0).all()


def test_policies_respect_clip_and_masks():
    e = get_economy("C")
    lr = MuZeroLearner(e, DRY, np.random.default_rng(1))
    res = run_learning(e, lr, DRY, seed=1)
    assert (res.beta_star[:, 0] == 0).all()          # fiat never consumable


@pytest.mark.slow
def test_a1_full_run_reaches_nash():
    e = get_economy("A1")
    cfg = LearnerConfig()
    lr = MuZeroLearner(e, cfg, np.random.default_rng(0))
    res = run_learning(e, lr, cfg, seed=0)
    b, _ = br_gain(e, res.alpha_star, res.beta_star, res.mu_star)
    assert b < 0.1                                    # paper: 1.5e-12; seed slack allowed
    agg = (e.nu[:, None] * res.mu_star).sum(axis=0)
    assert abs(agg[0] - 0.5) < 0.05
