from typing import cast

import pytest
import pytest_check as check
import torch

from src.environments.kiyotaki_wright import KiyotakiAgent, KiyotakiWrightScenario


def approx_equal(
    a: float, b: float, abs_tol: float = 1e-5, rel_tol: float = 1e-5
) -> bool:
    """Check if two numbers are approximately equal.
    Args:
        a: The first number.
        b: The second number.
        abs_tol: The absolute tolerance.
        rel_tol: The relative tolerance.
    Returns:
        True if the two numbers are approximately equal, False otherwise.
    """
    satisfy_abs_tol = abs(a - b) < abs_tol
    satisfy_rel_tol = abs(a - b) / max(abs(a), abs(b)) < rel_tol
    return satisfy_abs_tol and satisfy_rel_tol


def test_welfare_metrics() -> None:
    """Test welfare metrics calculation and Pareto optimality"""
    device = torch.device("cpu")
    batch_dim = 1
    n_agents = 9

    scenario = KiyotakiWrightScenario()
    world = scenario.make_world(batch_dim=batch_dim, device=device, n_agents=n_agents)

    # Test initial state metrics
    for i, agent in enumerate(world.agents):
        agent = cast(KiyotakiAgent, agent)
        metrics = scenario.compute_welfare_metrics(agent, world)

        # Check metric keys exist
        for key in ["welfare_loss_pct", "exploitability_pct"]:
            check.is_true(
                key in metrics,
                f"Missing required metric '{key}' for agent {i}. "
                f"Available metrics: {list(metrics.keys())}",
            )

        # Initial state should not be Pareto optimal
        check.is_true(
            metrics["welfare_loss_pct"] > 0,
            f"Agent {i} welfare loss should be positive in non-optimal state. "
            f"Got: {metrics['welfare_loss_pct']}. "
            f"Agent type: {agent.type}, holding: {agent.holding}",
        )

        # Verify exploitability is non-negative
        check.is_true(
            metrics["exploitability_pct"] >= 0,
            f"Agent {i} exploitability should be non-negative. "
            f"Got: {metrics['exploitability_pct']}. "
            f"Agent type: {agent.type}, holding: {agent.holding}",
        )

    # Test Pareto optimal state
    print("\nTesting Pareto optimal state...")
    initial_holdings = [(a.type, a.holding) for a in world.agents]
    print(f"Initial agent holdings: {initial_holdings}")

    for i, agent in enumerate(world.agents):
        agent = cast(KiyotakiAgent, agent)
        agent.holding = agent.type  # Give each agent their preferred good

    final_holdings = [(a.type, a.holding) for a in world.agents]
    print(f"Final agent holdings: {final_holdings}")

    metrics = scenario.compute_welfare_metrics(
        cast(KiyotakiAgent, world.agents[0]), world
    )

    welfare_loss = abs(metrics["welfare_loss_pct"])
    check.is_true(
        welfare_loss < 1e-5,
        f"Expected welfare loss close to 0 in Pareto optimal state. "
        f"Got: {welfare_loss}. \n"
        f"Current metrics: {metrics}\n"
        f"Storage costs: {scenario.storage_costs}\n"
        f"Consumption values: {scenario.consumption_values}",
    )


@pytest.mark.parametrize("n_agents", [3, 6, 9])
def test_world_creation(n_agents: int) -> None:
    """Test world creation with different numbers of agents"""
    device = torch.device("cpu")
    batch_dim = 1

    scenario = KiyotakiWrightScenario()
    world = scenario.make_world(batch_dim=batch_dim, device=device, n_agents=n_agents)

    # Check number of agents
    check.equal(
        len(world.agents),
        n_agents,
        f"Expected {n_agents} agents, got {len(world.agents)}",
    )

    # Check agent distribution
    agent_types = [cast(KiyotakiAgent, agent).type for agent in world.agents]
    for type_id in range(3):
        type_count = sum(1 for t in agent_types if t == type_id)
        expected_count = n_agents // 3
        check.equal(
            type_count,
            expected_count,
            f"Expected {expected_count} agents of type {type_id}, got {type_count}. "
            f"Full type distribution: {agent_types}",
        )


def test_invalid_world_creation() -> None:
    """Test that world creation fails with invalid number of agents"""
    device = torch.device("cpu")
    batch_dim = 1
    invalid_n_agents = 4  # Not divisible by 3

    scenario = KiyotakiWrightScenario()
    with pytest.raises(AssertionError, match="Number of agents must be divisible by 3"):
        scenario.make_world(
            batch_dim=batch_dim, device=device, n_agents=invalid_n_agents
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
