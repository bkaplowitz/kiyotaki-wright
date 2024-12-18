from typing import cast

import torch

from src.environments.kiyotaki_wright import KiyotakiAgent, KiyotakiWrightScenario


def test_welfare_metrics() -> None:
    device = torch.device("cpu")
    batch_dim = 1
    n_agents = 9

    scenario = KiyotakiWrightScenario()
    world = scenario.make_world(batch_dim=batch_dim, device=device, n_agents=n_agents)

    # Test initial state metrics
    for agent in world.agents:
        agent = cast(KiyotakiAgent, agent)
        metrics = scenario.compute_welfare_metrics(agent, world)

        assert "welfare_loss_pct" in metrics
        assert "exploitability_pct" in metrics

        # Initial state should not be Pareto optimal
        assert metrics["welfare_loss_pct"] > 0

        # Verify exploitability is non-negative
        assert metrics["exploitability_pct"] >= 0

    # Test Pareto optimal state
    for agent in world.agents:
        agent = cast(KiyotakiAgent, agent)
        agent.holding = agent.type  # Give each agent their preferred good

    metrics = scenario.compute_welfare_metrics(
        cast(KiyotakiAgent, world.agents[0]), world
    )
    assert abs(metrics["welfare_loss_pct"]) < 1e-6  # Should be close to 0
