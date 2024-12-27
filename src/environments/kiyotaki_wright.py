from typing import Required, TypedDict, TypeVar, cast, override
from typing import Callable
import plotly.graph_objects as go  # type: ignore
import torch
from pyparsing import Any
from vmas.simulator.core import (  # type: ignore
    Agent,
    Shape,
    World,
)
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color


class KiyotakiAgent(Agent):
    def __init__(
        self,
        name: str,
        type: int,
        holding: int,
        shape: Shape | None = None,
        just_traded: bool = False,
        steps_without_trade: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name,
            shape,  # type: ignore
            **kwargs,
        )
        self.type = type
        self.holding = holding
        self.just_traded = just_traded
        self.steps_without_trade = (
            torch.tensor(0.0) if steps_without_trade is None else steps_without_trade
        )


# Define a type variable for agent type
T = TypeVar("T", bound=Agent)


class TradingNetwork(World):
    """Trading network specialized for KiyotakiAgent"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.edges: list[tuple[int, int]] = []
        # Change to list instead of Sequence to match parent
        self._agents: list[KiyotakiAgent] = []

    @override
    def add_agent(self, agent: Agent) -> None:
        """Only way to add agents to the world"""
        if not isinstance(agent, KiyotakiAgent):
            raise TypeError("TradingNetwork only accepts KiyotakiAgent instances")
        super().add_agent(agent)

    def add_edge(self, agent1_id: int, agent2_id: int) -> None:
        self.edges.append((agent1_id, agent2_id))

    @property
    @override
    def agents(self) -> list[KiyotakiAgent]:
        return self._agents


class WorldArgs(TypedDict, total=False):
    types: Required[tuple[int, ...]]
    n_agents: Required[int]
    initial_holdings: Required[tuple[int, ...]]


class KiyotakiWrightScenario(BaseScenario):
    def __init__(self) -> None:
        super().__init__()
        # Storage costs for each good type
        self.preferences: Callable[[int, int], float] = lambda preferred_type, current_type: -1/2*(
            preferred_type - current_type
        )**2
        self._world: TradingNetwork | None = None
        self.storage_costs = torch.tensor([0.3, 0.2, 0.1])
        # Consumption values for each agent type
        self.consumption_values = torch.tensor([3.0, 3.0, 3.0])
        # Production costs
        self.production_costs = torch.tensor([0.5, 0.5, 0.5])
        # Trading costs
        self.trading_cost = 0.1
        # Meeting probability
        self.meeting_prob = 0.8
        # Maximum steps without trading before reset
        self.max_steps_without_trade = 50
        # Add discount factor for value calculations
        self.gamma = 0.95
        # Initialize value function table
        self._value_table: dict[tuple[int, int], float] = {}
        self._learning_rate = 0.1

    @override
    def make_world(
        self,
        batch_dim: int,
        device: torch.device,
        **kwargs: Any,
    ) -> TradingNetwork:
        """Create a new world with specified number of agents"""
        # Extract kwargs with defaults
        assert isinstance(kwargs, dict) and {
            "types",
            "n_agents",
            "initial_holdings",
        }.issubset(set(kwargs.keys()))

        # Create the world
        world = TradingNetwork(batch_dim=batch_dim, device=device)

        # Set up agents
        n_agents = kwargs["n_agents"]
        types = kwargs["types"]
        initial_holdings = kwargs["initial_holdings"]

        # Ensure number of agents is divisible by number of types
        if n_agents % len(types) != 0:
            raise ValueError("Number of agents must be divisible by number of types")

        # Generate initial holdings if not provided
        # In Kiyotaki-Wright, type i agents initially hold good i+1 (mod 3)
        if initial_holdings is None:
            agents_per_type = n_agents // len(types)
            initial_holdings_filled: list[int] = []
            for type_idx in types:
                holding = (type_idx + 1) % len(types)  # Each type holds next good
                initial_holdings_filled.extend([holding] * agents_per_type)
        else:
            initial_holdings_filled = initial_holdings

        if n_agents % len(initial_holdings_filled) != 0:
            raise ValueError(
                f"Expected {n_agents} initial holdings, got {len(initial_holdings_filled)}"
            )

        # Add agents
        for i in range(n_agents):
            agent_type = types[i % len(types)]
            agent = KiyotakiAgent(
                name=f"agent_{i}",
                collide=False,
                type=agent_type,
                holding=agent_type,
            )
            agent.holding = (agent.type + 1) % 3  # Each type produces next good
            agent.just_traded = False
            agent.steps_without_trade = torch.zeros(batch_dim, device=device)
            world.add_agent(agent)

            # Set agent colors based on type
            if agent.type == 0:
                agent.color = Color.RED.value
            elif agent.type == 1:
                agent.color = Color.GREEN.value
            else:
                agent.color = Color.BLUE.value

        # Important: Set the world before returning
        self._world = world
        return world

    @property
    @override
    def world(self) -> TradingNetwork:
        """The :class:`~vmas.simulator.core.World` associated toi this scenario."""
        assert self._world is not None, (
            "You first need to set `self._world` in the `make_world` method"
        )
        return self._world

    @override
    def reset_world_at(self, env_index: int | None = None) -> None:
        """Reset the world state for specified environment(s)

        Args:
            env_index: Index of environment to reset. If None, reset all environments.
        """
        for agent in self.world.agents:
            # Reset holding to initial state (type + 1 mod 3)
            agent.holding = (agent.type + 1) % 3
            agent.just_traded = False

            # Reset steps without trade counter
            if env_index is None:
                # Vectorized reset for all environments
                agent.steps_without_trade.zero_()
            else:
                # Reset single environment
                agent.steps_without_trade[env_index] = 0
            # Reset agent just traded
            agent.just_traded = False

    @override
    def reward(self, agent: Agent) -> torch.Tensor:
        """Calculate reward for an agent"""
        if not isinstance(agent, KiyotakiAgent):
            raise TypeError(
                "Agent must be a KiyotakiAgent, else this method will fail."
            )
        reward = torch.zeros(self.world.batch_dim, device=self.world.device)

        # Consumption reward if holding preferred good
        preferred_good = agent.type
        if agent.holding == preferred_good:
            reward += self.consumption_values[agent.type]
            agent.holding = (agent.type + 1) % 3  # Produce after consuming

        # Storage costs
        reward -= self.storage_costs[agent.holding]

        # Trading costs if trade occurred
        if agent.just_traded:
            reward -= self.trading_cost
            agent.just_traded = False
            agent.steps_without_trade.zero_()
        else:
            agent.steps_without_trade += 1

        return reward

    @override
    def observation(self, agent: Agent) -> torch.Tensor:
        """Return observation for an agent"""
        # Local observation (used during execution)
        if not isinstance(agent, KiyotakiAgent):
            raise TypeError(
                "Agent must be a KiyotakiAgent, else this method will fail."
            )
        type_onehot = torch.zeros(3, device=self.world.device)
        type_onehot[agent.type] = 1

        holding_onehot = torch.zeros(3, device=self.world.device)
        holding_onehot[agent.holding] = 1

        steps_without_trade_norm = (
            agent.steps_without_trade / self.max_steps_without_trade
        )

        # Global state information (used during training)
        all_agents_types = torch.zeros(
            len(self.world.agents) * 3, device=self.world.device
        )
        all_agents_holdings = torch.zeros(
            len(self.world.agents) * 3, device=self.world.device
        )

        for i, other in enumerate(self.world.agents):
            all_agents_types[i * 3 + other.type] = 1
            all_agents_holdings[i * 3 + other.holding] = 1

        # Concatenate local and global observations
        obs = torch.cat([
            type_onehot,  # Local: agent's type
            holding_onehot,  # Local: agent's holding
            steps_without_trade_norm.unsqueeze(-1),  # Local: trading history
            all_agents_types,  # Global: all agents' types
            all_agents_holdings,  # Global: all agents' holdings
        ])

        return obs

    @override
    def done(self) -> torch.Tensor:
        """Return done signal"""
        # Episode ends if all agents have not traded for too long
        return torch.tensor(
            [self.agent_done(agent) for agent in self.world.agents],
            device=self.world.device,
        )

    def agent_done(self, agent: KiyotakiAgent) -> torch.Tensor:
        """Return done signal for an agent"""
        return agent.steps_without_trade >= self.max_steps_without_trade

    @override
    def process_action(self, agent: Agent) -> None:
        """Process agent's action"""
        if not isinstance(agent, KiyotakiAgent):
            raise TypeError(
                "Agent must be a KiyotakiAgent, else this method will fail."
            )
        # Find potential trading partners
        for other in self.world.agents:
            if other is agent:
                continue

            # Randomly determine if meeting occurs
            if torch.rand(1) < self.meeting_prob:
                # Check if trade would be beneficial
                agent_wants = other.holding == agent.type
                other_wants = agent.holding == other.type

                if agent_wants and other_wants:
                    # Execute trade
                    agent.holding, other.holding = other.holding, agent.holding
                    agent.just_traded = True
                    other.just_traded = True

    def compute_welfare_metrics(
        self, agent: KiyotakiAgent, world: TradingNetwork
    ) -> dict[str, float]:
        """
        Compute welfare metrics including:
        1. Social welfare loss (deviation from Pareto optimal)
        2. Exploitability (how much an agent could gain by deviating)
        """
        # First compute current utility for all agents
        current_utilities: list[float] = []
        for a in world.agents:
            # Base utility calculation
            utility = -self.storage_costs[a.holding]  # Storage cost
            if a.holding == a.type:  # If holding preferred good
                utility += self.consumption_values[a.type]
            current_utilities.append(cast(float, utility.item()))

        current_social_welfare = sum(current_utilities)

        # Compute Pareto optimal welfare
        # In the optimal case, each agent type holds their preferred good
        optimal_utilities: list[float] = []
        for a in world.agents:
            optimal_utility = (
                self.consumption_values[a.type] - self.storage_costs[a.type]
            ).item()
            optimal_utilities.append(optimal_utility)

        optimal_social_welfare: float = sum(optimal_utilities)

        # Compute welfare loss as percentage
        welfare_loss_pct = (
            (optimal_social_welfare - current_social_welfare)
            / abs(optimal_social_welfare)
            * 100
        )

        # Compute exploitability
        # Consider all possible deviations for the agent
        current_strategy_value = current_utilities[world.agents.index(agent)]
        best_deviation_value = current_strategy_value

        # Consider each possible trading strategy
        possible_holdings = [0, 1, 2]
        for alternative_holding in possible_holdings:
            if alternative_holding == agent.holding:
                continue

            # Compute utility under this deviation
            deviation_utility = -self.storage_costs[alternative_holding]
            if alternative_holding == agent.type:
                deviation_utility += self.consumption_values[agent.type]

            # Include expected future trading opportunities
            for other in world.agents:
                if other is agent:
                    continue

                # Probability of beneficial trade
                trade_probability = self.meeting_prob * (
                    other.holding == agent.type and alternative_holding == other.type
                )

                expected_trade_value = trade_probability * (
                    self.consumption_values[agent.type] - self.trading_cost
                )

                deviation_utility += expected_trade_value

            best_deviation_value = max(best_deviation_value, deviation_utility.item())

        # Exploitability as percentage improvement possible
        exploitability_pct: float = (
            (best_deviation_value - current_strategy_value)
            / abs(current_strategy_value)
            * 100
            if current_strategy_value != 0
            else 0.0
        )

        return {
            "welfare_loss_pct": welfare_loss_pct,
            "exploitability_pct": exploitability_pct,
        }

    def log_metrics(self, world: TradingNetwork) -> dict[str, float]:
        """
        Compute and log aggregate welfare metrics for the entire economy
        """
        total_welfare_loss = 0.0
        total_exploitability = 0.0

        for agent in world.agents:
            metrics = self.compute_welfare_metrics(agent, world)
            total_welfare_loss += metrics["welfare_loss_pct"]
            total_exploitability += metrics["exploitability_pct"]

        n_agents = len(world.agents)
        return {
            "avg_welfare_loss_pct": total_welfare_loss / n_agents,
            "avg_exploitability_pct": total_exploitability / n_agents,
        }

    def visualize_metrics(
        self, agents: list[KiyotakiAgent], world: TradingNetwork
    ) -> go.Figure:
        """Visualize the current state of the economy"""
        import plotly.graph_objects as go  # type: ignore[import-untyped]
        from plotly.subplots import make_subplots  # type: ignore[import-untyped]

        # Get current metrics
        metrics = self.log_metrics(world)

        # Count holdings by type
        holdings_by_type = torch.zeros(
            (3, 3), device=world.device
        )  # [agent_type, holding]
        for agent in world.agents:
            holdings_by_type[agent.type, agent.holding] += 1

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Holdings Distribution by Agent Type",
                "Trading Network",
                "Welfare Metrics",
                "Storage Costs vs Consumption Values",
            ),
        )

        # 1. Holdings Distribution
        agent_types = ["Type 0", "Type 1", "Type 2"]
        goods = ["Good 0", "Good 1", "Good 2"]

        fig.add_trace(
            go.Heatmap(
                z=holdings_by_type.cpu().numpy(),
                x=goods,
                y=agent_types,
                colorscale="Viridis",
                showscale=True,
                text=holdings_by_type.cpu().numpy().astype(int),
                texttemplate="%{text}",
                textfont={"size": 20},
            ),
            row=1,
            col=1,
        )

        # 2. Trading Network
        edge_x = []
        edge_y = []
        node_x = []
        node_y = []
        import numpy as np

        # Create circular layout
        for i in range(len(agents)):
            angle = (2 * np.pi * i) / len(agents)
            x = np.cos(angle)
            y = np.sin(angle)
            node_x.append(x)
            node_y.append(y)

        # Add edges for recent trades
        for i, agent in enumerate(world.agents):
            if agent.just_traded:
                for j, other in enumerate(world.agents):
                    if other.just_traded and i < j:
                        edge_x.extend([node_x[i], node_x[j], None])
                        edge_y.extend([node_y[i], node_y[j], None])

        # Add edges
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                line=dict(width=0.5, color="#888"),
                hoverinfo="none",
                mode="lines",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        # Add nodes
        node_colors = ["red", "green", "blue"]
        fig.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers",
                hoverinfo="text",
                marker=dict(
                    size=15,
                    color=[node_colors[agent.type] for agent in agents],
                    line_width=2,
                ),
                text=[
                    f"Agent {i}<br>Type {agent.type}<br>Holding {agent.holding}"
                    for i, agent in enumerate(agents)
                ],
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        # 3. Welfare Metrics
        fig.add_trace(
            go.Bar(
                x=["Welfare Loss", "Exploitability"],
                y=[metrics["avg_welfare_loss_pct"], metrics["avg_exploitability_pct"]],
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        # 4. Storage Costs vs Consumption Values
        fig.add_trace(
            go.Bar(
                name="Storage Costs",
                x=goods,
                y=self.storage_costs.cpu().numpy(),
                showlegend=True,
            ),
            row=2,
            col=2,
        )

        fig.add_trace(
            go.Bar(
                name="Consumption Values",
                x=goods,
                y=self.consumption_values.cpu().numpy(),
                showlegend=True,
            ),
            row=2,
            col=2,
        )

        # Update layout
        fig.update_layout(
            title_text="Kiyotaki-Wright Economy State",
            height=800,
            showlegend=True,
        )

        return fig

    def visualize_state(self, world: TradingNetwork) -> None:
        """Visualize the current state using matplotlib"""
        import matplotlib.pyplot as plt
        import numpy as np

        # Create figure with subplots
        _fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

        # 1. Holdings Distribution Heatmap
        holdings_by_type = torch.zeros((3, 3), device=world.device)
        for agent in world.agents:
            holdings_by_type[agent.type, agent.holding] += 1

        _im = ax1.imshow(holdings_by_type.cpu().numpy(), cmap="viridis")
        ax1.set_title("Holdings Distribution")
        ax1.set_xlabel("Holding")
        ax1.set_ylabel("Agent Type")

        # Add text annotations to heatmap
        for i in range(3):
            for j in range(3):
                _text = ax1.text(
                    j,
                    i,
                    int(holdings_by_type[i, j].item()),
                    ha="center",
                    va="center",
                    color="w",
                )

        # 2. Agent Network (simplified)
        ax2.set_title("Agent Network")
        for agent in world.agents:
            if agent.just_traded:
                ax2.plot(agent.type, agent.holding, "ro")
            else:
                ax2.plot(agent.type, agent.holding, "bo")
        ax2.set_xlim(-0.5, 2.5)
        ax2.set_ylim(-0.5, 2.5)

        # 3. Welfare Metrics
        metrics = self.log_metrics(world)
        metrics_data = [
            metrics["avg_welfare_loss_pct"],
            metrics["avg_exploitability_pct"],
        ]
        ax3.bar(["Welfare Loss %", "Exploitability %"], metrics_data)
        ax3.set_title("Welfare Metrics")

        # 4. Storage Costs vs Consumption Values
        x = np.arange(3)
        width = 0.35

        ax4.bar(
            x - width / 2,
            self.storage_costs.cpu().numpy(),
            width,
            label="Storage Costs",
        )
        ax4.bar(
            x + width / 2,
            self.consumption_values.cpu().numpy(),
            width,
            label="Consumption Values",
        )
        ax4.set_xticks(x)
        ax4.set_xticklabels(["Good 0", "Good 1", "Good 2"])
        ax4.legend()
        ax4.set_title("Costs and Values")

        plt.tight_layout()
        plt.show()

    def get_value(self, agent: KiyotakiAgent, world: TradingNetwork) -> float:
        """Get the current value estimate for an agent's state"""
        state = (agent.type, agent.holding)
        # Assume preferences are single peaked always
        self._value_table =
        if state not in self._value_table:
            self._value_table[state] =
        return self._value_table[state]

    def update_value(
        self, agent: KiyotakiAgent, world: TradingNetwork, td_error: float
    ) -> None:
        """Update value estimate using TD error"""
        state = (agent.type, agent.holding)
        current_value = self.get_value(agent, world)
        self._value_table[state] = current_value + self._learning_rate * td_error

    def should_trade(
        self, agent1: KiyotakiAgent, agent2: KiyotakiAgent, world: TradingNetwork
    ) -> bool:
        """
        Determine if two agents should trade based on their value functions.
        Trade occurs only if both agents benefit individually (individually rational).
        """
        # Get current values
        value1_before = self.get_value(agent1, world)
        value2_before = self.get_value(agent2, world)

        # Simulate trade
        holding1, holding2 = agent1.holding, agent2.holding
        agent1.holding, agent2.holding = holding2, holding1

        # Get post-trade values
        value1_after = self.get_value(agent1, world)
        value2_after = self.get_value(agent2, world)

        # Restore original holdings
        agent1.holding, agent2.holding = holding1, holding2

        # Trade only if both agents benefit after accounting for trading costs
        agent1_benefit = value1_after - value1_before > self.trading_cost
        agent2_benefit = value2_after - value2_before > self.trading_cost

        return agent1_benefit and agent2_benefit

    def trade_individual_welfare(
        self, agent1: KiyotakiAgent, agent2: KiyotakiAgent, world: TradingNetwork
    ) -> tuple[float, float]:
        """Compute the welfare of a trade"""
        agent1_benefit = self.get_value(agent1, world) - self.get_value(agent1, world)
        agent2_benefit = self.get_value(agent2, world) - self.get_value(agent2, world)
        return agent1_benefit, agent2_benefit

    def trade(
        self, agent1: KiyotakiAgent, agent2: KiyotakiAgent, world: TradingNetwork
    ) -> None:
        """Execute trade between two agents"""
        # Swap holdings
        agent1.holding, agent2.holding = agent2.holding, agent1.holding

        # Mark agents as having just traded
        agent1.just_traded = True
        agent2.just_traded = True

        # Reset steps without trade
        agent1.steps_without_trade.zero_()
        agent2.steps_without_trade.zero_()
