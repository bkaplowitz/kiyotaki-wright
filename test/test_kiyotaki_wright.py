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


class TestLearningDynamics(TestKiyotakiWrightBase):
    """Tests for learning dynamics and value updates"""

    @pytest.mark.fast
    def test_value_updates(
        self, scenario: KiyotakiWrightScenario, device: torch.device
    ) -> None:
        """Test that positive TD errors lead to value increases"""
        batch_dim = 1
        world = scenario.env_make_world(
            batch_dim=batch_dim,
            device=device,
            n_agents=3,
            types=(0, 1, 2),
            initial_holdings=None,
        )
        world = cast(TradingNetwork, world)

        # Setup test agent
        agent = world.agents[0]
        _initial_state = (agent.type, agent.holding)

        # Record initial value
        initial_value = scenario.get_value(agent, world)

        # Simulate positive TD error
        reward = 1.0
        next_value = 0.0  # Terminal state for simplicity
        td_error = reward + scenario.gamma * next_value - initial_value

        # Update with positive TD error
        scenario.update_value(agent, world, td_error)

        # Check value increased
        new_value = scenario.get_value(agent, world)
        check.greater(
            new_value,
            initial_value,
            f"Value should increase with positive TD error.\n"
            f"Initial value: {initial_value}\n"
            f"New value: {new_value}\n"
            f"TD error: {td_error}",
        )

    @pytest.mark.fast
    def test_negative_td_error(
        self, scenario: KiyotakiWrightScenario, device: torch.device
    ) -> None:
        """Test that negative TD errors lead to value decreases"""
        batch_dim = 1
        world = scenario.env_make_world(
            batch_dim=batch_dim,
            device=device,
            n_agents=3,
            types=(0, 1, 2),
            initial_holdings=None,
        )
        world = cast(TradingNetwork, world)

        # Setup test agent
        agent = world.agents[0]

        # Record initial value
        initial_value = scenario.get_value(agent, world)

        # Simulate negative TD error
        reward = -1.0
        next_value = 0.0
        beta = 0.9
        td_error = reward + beta * next_value - initial_value

        # Update with negative TD error
        scenario.update_value(agent, world, td_error)

        # Check value decreased
        new_value = scenario.get_value(agent, world)
        check.less(
            new_value,
            initial_value,
            f"Value should decrease with negative TD error.\n"
            f"Initial value: {initial_value}\n"
            f"New value: {new_value}\n"
            f"TD error: {td_error}",
        )

    @pytest.mark.fast
    def test_trading_value_consistency(
        self, scenario: KiyotakiWrightScenario, device: torch.device
    ) -> None:
        """Test that trading decisions are consistent with value estimates"""
        batch_dim = 1
        world = scenario.env_make_world(
            batch_dim=batch_dim,
            device=device,
            n_agents=3,
            types=(0, 1, 2),
            initial_holdings=None,
        )
        world = cast(TradingNetwork, world)

        # Setup two agents for trading
        agent1 = world.agents[0]
        agent2 = world.agents[1]

        # Get initial values
        value1_before = scenario.get_value(agent1, world)
        value2_before = scenario.get_value(agent2, world)

        # Simulate trading decision
        should_trade = scenario.should_trade(agent1, agent2, world)

        if should_trade:
            # If agents decide to trade, both agents should benefit after trading costs
            scenario.trade(agent1, agent2, world)

            value1_after = scenario.get_value(agent1, world)
            value2_after = scenario.get_value(agent2, world)

            check.greater(
                value1_after - value1_before,
                scenario.trading_cost,
                "Agent 1 should benefit more than trading cost",
            )
            check.greater(
                value2_after - value2_before,
                scenario.trading_cost,
                "Agent 2 should benefit more than trading cost",
            )

    def test_trading_value_consistency_reversed(
        self, scenario: KiyotakiWrightScenario, device: torch.device
    ) -> None:
        """Test that value estimates imply consistent trading decisions"""
        batch_dim = 1
        world = scenario.env_make_world(
            batch_dim=batch_dim,
            device=device,
            n_agents=3,
            types=(0, 1, 2),
            initial_holdings=None,
        )
        world = cast(TradingNetwork, world)

        # Setup two agents for trading
        agent1 = world.agents[0]
        agent2 = world.agents[1]
        agent1.holding = 1
        agent2.holding = 2
        # Set agents preferences to be opposite
        agent1.type = 2
        agent2.type = 1
        # Verify that agents should not trade
        check.is_false(scenario.should_trade(agent1, agent2, world))

    @pytest.mark.fast
    def test_learning_convergence(
        self, scenario: KiyotakiWrightScenario, device: torch.device
    ) -> None:
        """Test that values converge with repeated updates"""
        batch_dim = 1
        world = scenario.env_make_world(
            batch_dim=batch_dim,
            device=device,
            n_agents=3,
            types=(0, 1, 2),
            initial_holdings=None,
        )
        world = cast(TradingNetwork, world)

        agent = world.agents[0]

        # Record initial value
        initial_value = scenario.get_value(agent, world)

        # Calculate theoretical bounds
        max_possible_value = scenario.consumption_values.max() / (1 - scenario.gamma)
        min_possible_value = -scenario.storage_costs.max() / (1 - scenario.gamma)

        # Perform multiple updates with decreasing TD errors
        values = [initial_value]
        n_updates = 10

        for i in range(n_updates):
            # TD error decreases over time and stays within theoretical bounds

            td_error = (max_possible_value.item() - initial_value) * (1 - i / n_updates)
            scenario.update_value(agent, world, td_error)
            new_value = scenario.get_value(agent, world)
            values.append(new_value)

            # Check value stays within bounds
            check.greater(
                new_value,
                min_possible_value,
                f"Value {new_value} below theoretical minimum {min_possible_value}",
            )
            check.less(
                new_value,
                max_possible_value,
                f"Value {new_value} above theoretical maximum {max_possible_value}",
            )

        # Check convergence
        deltas = [abs(values[i + 1] - values[i]) for i in range(len(values) - 1)]
        check.is_true(
            deltas[-1] < deltas[0],
            f"Value updates should converge. Initial delta: {deltas[0]}, "
            f"Final delta: {deltas[-1]}",
        )

    @pytest.mark.fast
    def test_value_bounds(
        self, scenario: KiyotakiWrightScenario, device: torch.device
    ) -> None:
        """Test that values stay within reasonable bounds"""
        batch_dim = 1
        world = scenario.env_make_world(
            batch_dim=batch_dim,
            device=device,
            n_agents=3,
            types=(0, 1, 2),
            initial_holdings=None,
        )
        world = cast(TradingNetwork, world)

        for agent in world.agents:
            value = scenario.get_value(agent, world)

            # Values should be bounded by theoretical limits
            max_possible_value = scenario.consumption_values.max() / (
                1 - scenario.gamma
            )
            min_possible_value = -scenario.storage_costs.max() / (1 - scenario.gamma)

            check.less(
                value,
                max_possible_value,
                f"Value exceeds theoretical maximum: {value} > {max_possible_value}",
            )
            check.greater(
                value,
                min_possible_value,
                f"Value below theoretical minimum: {value} < {min_possible_value}",
            )
