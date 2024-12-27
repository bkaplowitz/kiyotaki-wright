from typing import cast

import pytest
import pytest_check as check
import torch

from src.environments.kiyotaki_wright import (
    KiyotakiAgent,
    KiyotakiWrightScenario,
    TradingNetwork,
)


@pytest.fixture(scope="class")
def scenario() -> KiyotakiWrightScenario:
    return KiyotakiWrightScenario()


@pytest.fixture(scope="class")
def device() -> torch.device:
    return torch.device("cpu")


class TestKiyotakiWrightBase:
    """Base test utilities for Kiyotaki Wright tests"""

    @staticmethod
    def approx_equal(
        a: float, b: float, abs_tol: float = 1e-5, rel_tol: float = 1e-5
    ) -> bool:
        """Check if two numbers are approximately equal."""
        satisfy_abs_tol = abs(a - b) < abs_tol
        satisfy_rel_tol = abs(a - b) / max(abs(a), abs(b)) < rel_tol
        return satisfy_abs_tol and satisfy_rel_tol


class TestWelfareMetrics(TestKiyotakiWrightBase):
    """Tests for welfare metrics calculations"""

    @pytest.mark.fast
    def test_welfare_metrics(
        self, scenario: KiyotakiWrightScenario, device: torch.device
    ) -> None:
        """Test welfare metrics calculation"""
        batch_dim = 1
        n_agents = 9

        world = cast(
            TradingNetwork,
            scenario.env_make_world(
                batch_dim=batch_dim,
                device=device,
                n_agents=n_agents,
                types=(0, 1, 2),  # Add types parameter
                initial_holdings=None,
            ),
        )

        # Test initial state metrics
        for i, agent in enumerate(world.agents):
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
        for agent in world.agents:
            agent.holding = agent.type  # Give each agent their preferred good

        metrics = scenario.compute_welfare_metrics(world.agents[0], world)
        welfare_loss = abs(metrics["welfare_loss_pct"])
        check.is_true(
            welfare_loss < 1e-5,
            f"Expected welfare loss close to 0 in Pareto optimal state. "
            f"Got: {welfare_loss}. \n"
            f"Current metrics: {metrics}\n"
            f"Storage costs: {scenario.storage_costs}\n"
            f"Consumption values: {scenario.consumption_values}",
        )


class TestWorldCreation(TestKiyotakiWrightBase):
    """Tests for world creation and initialization"""

    @pytest.mark.fast
    @pytest.mark.parametrize("n_agents", [3, 6, 9])
    def test_world_creation(
        self, scenario: KiyotakiWrightScenario, device: torch.device, n_agents: int
    ) -> None:
        """Test world creation with different numbers of agents"""
        batch_dim = 1
        world = scenario.env_make_world(
            batch_dim=batch_dim,
            device=device,
            n_agents=n_agents,
            types=(0, 1, 2),  # Add types parameter
            initial_holdings=None,
        )

        check.is_instance(world, TradingNetwork)
        check.equal(len(world.agents), n_agents)

        # Check agent distribution
        agent_types = []
        for agent in world.agents:
            if hasattr(agent, "type"):
                agent_with_type = cast(KiyotakiAgent, agent)
                agent_types.append(agent_with_type.type)
            else:
                check.fail("Should have type 'type' but did not.")
                agent_types = [1] * len(world.agents)

        for type_id in range(3):
            type_count = sum(1 for t in agent_types if t == type_id)
            expected_count = n_agents // 3
            check.equal(
                type_count,
                expected_count,
                f"Expected {expected_count} agents of type {type_id}, got {type_count}. "
                f"Full type distribution: {agent_types}",
            )

    @pytest.mark.fast
    def test_invalid_world_creation(
        self, scenario: KiyotakiWrightScenario, device: torch.device
    ) -> None:
        """Test that world creation fails with invalid number of agents"""
        batch_dim = 1
        invalid_n_agents = 4  # Not divisible by 3

        with pytest.raises(
            ValueError, match="Number of agents must be divisible by number of types"
        ):
            scenario.env_make_world(
                batch_dim=batch_dim,
                device=device,
                n_agents=invalid_n_agents,
                types=(0, 1, 2),  # Add types parameter
                initial_holdings=None,
            )


class TestWorldReset(TestKiyotakiWrightBase):
    """Tests for world reset functionality"""

    @pytest.mark.fast
    def test_reset_world_at(
        self, scenario: KiyotakiWrightScenario, device: torch.device
    ) -> None:
        """Test reset_world_at functionality for both single and batch resets"""
        batch_dim = 4
        n_agents = 9
        world = scenario.env_make_world(
            batch_dim=batch_dim,
            device=device,
            n_agents=n_agents,
            types=(0, 1, 2),
            initial_holdings=None,
        )

        # Test full batch reset
        scenario.reset_world_at()
        for agent in world.agents:
            ag = cast(KiyotakiAgent, agent)
            check.equal(
                ag.holding,
                (ag.type + 1) % 3,
                f"Agent {ag.name} holding not reset correctly",
            )
            check.equal(
                ag.steps_without_trade.shape[0],
                batch_dim,
                "Steps without trade tensor has wrong batch dimension",
            )
            check.is_true(
                torch.all(ag.steps_without_trade == 0),
                "Steps without trade not reset to zero",
            )
            check.is_false(ag.just_traded)

        # Test single environment reset
        test_env_idx = 2
        # Set all environments to 10 steps without trade
        for agent in world.agents:
            ag = cast(KiyotakiAgent, agent)
            ag.steps_without_trade[:] = 10
            ag.just_traded = True

        # Reset only one environment
        scenario.reset_world_at(test_env_idx)

        for agent in world.agents:
            ag = cast(KiyotakiAgent, agent)
            # Check that the reset environment is zeroed
            check.equal(
                ag.steps_without_trade[test_env_idx].item(),
                0,
                f"Environment {test_env_idx} not reset correctly",
            )

            # Check that other environments are unchanged
            other_envs = torch.cat([
                ag.steps_without_trade[:test_env_idx],
                ag.steps_without_trade[test_env_idx + 1 :],
            ])
            check.is_true(
                torch.all(other_envs == 10),
                f"Unspecified environments were modified during reset. Expected all 10, got {other_envs}",
            )

    @pytest.mark.fast
    def test_reset_world_at_device_consistency(
        self, scenario: KiyotakiWrightScenario
    ) -> None:
        """Test device consistency during reset operations"""
        for device_str in ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]:
            device = torch.device(device_str)
            batch_dim = 2
            n_agents = 9

            world = scenario.env_make_world(
                batch_dim=batch_dim,
                device=device,
                n_agents=n_agents,
                types=(0, 1, 2),
                initial_holdings=None,
            )

            # Test full reset
            scenario.reset_world_at()
            for agent in world.agents:
                ag = cast(KiyotakiAgent, agent)
                check.equal(
                    ag.steps_without_trade.device,
                    device,
                    "Agent tensor on wrong device after full reset",
                )

            # Test single environment reset
            scenario.reset_world_at(0)
            for agent in world.agents:
                ag = cast(KiyotakiAgent, agent)
                check.equal(
                    ag.steps_without_trade.device,
                    device,
                    "Agent tensor on wrong device after single reset",
                )
