from pathlib import Path

import numpy as np
import pytest
import torch
from pytest_check import check
import torch as th
from train import (
    KWEnvironment,
    KWMultiAgentDQN,
    KWParams,
    analyze_equilibrium,
    plot_training_results,
    train,
)


# Fixtures
@pytest.fixture
def device() -> th.device:
    return th.device(
        "cuda"
        if th.cuda.is_available()
        else "mps"
        if th.backends.mps.is_available()
        else "cpu"
    )


@pytest.fixture
def env(device: th.device) -> KWEnvironment:
    params = KWParams()
    env = KWEnvironment(params)
    env.to(device)
    return env


@pytest.fixture
def agent(device: th.device, env: KWEnvironment) -> KWMultiAgentDQN:
    return KWMultiAgentDQN(
        num_agent_types=env.params.n_types,
        device=device,
        learning_rate=1e-4,
        gamma=env.params.gamma,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
    )


# Lightweight tests marked with 'quick'
@pytest.mark.quick
def test_environment_initialization(env: KWEnvironment) -> None:
    """Test basic environment initialization"""
    td = env.reset()
    check.is_instance(td, dict)
    check.is_true("agent_type" in td)
    check.is_true("inventory" in td)


@pytest.mark.quick
def test_agent_initialization(agent: KWMultiAgentDQN, env: KWEnvironment) -> None:
    """Test basic agent initialization"""
    check.equal(agent.num_agent_types, env.params.n_types)
    check.is_instance(agent.epsilon, float)
    check.is_true(0 <= agent.epsilon <= 1)


@pytest.mark.quick
def test_plot_training_results() -> None:
    """Test plot generation"""
    rewards = np.random.randn(10, 3)  # 10 episodes, 3 agent types
    fig = plot_training_results(rewards)
    check.is_not_none(fig)

    # Test save functionality
    temp_path = Path("test_plot.html")
    fig = plot_training_results(rewards, str(temp_path))
    check.is_true(temp_path.exists())
    temp_path.unlink()  # Cleanup


# Comprehensive tests
@pytest.mark.timeout(300)  # 5 minute timeout
def test_training_loop(env: KWEnvironment, agent: KWMultiAgentDQN) -> None:
    """Test full training loop"""
    n_episodes = 5
    episode_length = 10
    rewards = train(env, agent, n_episodes, episode_length)

    check.is_instance(rewards, list)
    check.equal(len(rewards), n_episodes)
    check.equal(len(rewards[0]), env.params.n_types)

    # Check rewards are within reasonable bounds
    rewards_array = np.array(rewards)
    check.is_true(np.all(np.isfinite(rewards_array)))


def test_analyze_equilibrium(env: KWEnvironment, agent: KWMultiAgentDQN) -> None:
    """Test equilibrium analysis"""
    results = analyze_equilibrium(env, agent, n_episodes=5)

    check.is_true("inventory_distribution" in results)
    check.is_true("trade_matrix" in results)

    inv_dist = results["inventory_distribution"]
    trade_mat = results["trade_matrix"]

    # Check shapes
    check.equal(inv_dist.shape[0], env.params.n_types)
    check.equal(trade_mat.shape, (env.params.n_types, env.params.n_types))

    # Check probability distributions
    check.is_true(np.allclose(np.sum(inv_dist), 1.0, atol=1e-5))
    check.is_true(np.all(inv_dist >= 0))


@pytest.mark.parametrize("epsilon", [0.0, 0.5, 1.0])
def test_agent_action_selection(
    agent: KWMultiAgentDQN, env: KWEnvironment, epsilon: float
) -> None:
    """Test agent action selection at different exploration rates"""
    agent.epsilon = epsilon
    td = env.reset()
    actions = agent.select_actions(td)

    check.is_instance(actions, th.Tensor)
    check.equal(len(actions), len(td["agent_type"]))
    check.is_true(th.all((actions == 0) | (actions == 1)))


# Integration tests
@pytest.mark.slow
def test_end_to_end() -> None:
    """Test complete training pipeline"""
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    params = KWParams()
    env = KWEnvironment(params)
    env.to(device)

    agent = KWMultiAgentDQN(
        num_agent_types=params.n_types,
        device=device,
        learning_rate=1e-4,
        gamma=params.discount_factor,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
    )

    # Quick training
    n_episodes = 10
    train(env, agent, n_episodes)

    # Analysis
    analyze_equilibrium(env, agent)

    # Save and load model
    temp_path = Path("test_agent.pt")
    th.save(agent.state_dict(), temp_path)

    # Load model
    new_agent = KWMultiAgentDQN(
        num_agent_types=params.n_types,
        device=device,
        learning_rate=1e-4,
        gamma=params.discount_factor,
    )
    new_agent.load_state_dict(th.load(temp_path))

    temp_path.unlink()  # Cleanup
