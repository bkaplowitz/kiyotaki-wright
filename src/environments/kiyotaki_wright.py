from typing import cast

import plotly.graph_objects as go  # type: ignore
import torch
from vmas.simulator.core import Agent, World  # type: ignore
from vmas.simulator.scenario import BaseScenario  # type: ignore
from vmas.simulator.utils import Color  # type: ignore


class KiyotakiAgent(Agent):
    def __init__(self, name: str, collide: bool = False) -> None:
        super().__init__(name=name, collide=collide)
        self.type: int
        self.holding: int
        self.just_traded: bool
        self.steps_without_trade: torch.Tensor


class KiyotakiWrightScenario(BaseScenario):
    def __init__(self) -> None:
        super().__init__()
        # Storage costs for each good type
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

    def make_world(
        self, batch_dim: int, device: torch.device, n_agents: int = 9
    ) -> World:
        """Create a new world with specified number of agents"""
        world = World(batch_dim=batch_dim, device=device)

        # Ensure number of agents is divisible by 3
        assert n_agents % 3 == 0, "Number of agents must be divisible by 3"
        agents_per_type = n_agents // 3

        # Add agents
        for i in range(n_agents):
            agent = KiyotakiAgent(name=f"agent_{i}", collide=False)
            agent.type = i // agents_per_type  # 0, 1, or 2
            agent.holding = (agent.type + 1) % 3  # Each type produces next good
            agent.just_traded = False
            agent.steps_without_trade = torch.zeros(batch_dim, device=device)
            world.add_agent(agent)

            # Set agent colors based on type
            if agent.type == 0:
                agent.color = Color.RED
            elif agent.type == 1:
                agent.color = Color.GREEN
            else:
                agent.color = Color.BLUE

        return world

    def reset_world_at(self, world: World, env_index: int | None = None) -> None:
        """Reset the world state"""
        for agent in world.agents:
            agent = cast(KiyotakiAgent, agent)
            agent.holding = (agent.type + 1) % 3
            agent.just_traded = False
            if env_index is None:
                agent.steps_without_trade.zero_()
            else:
                agent.steps_without_trade[env_index] = 0

    def reward(self, agent: KiyotakiAgent, world: World) -> torch.Tensor:
        """Calculate reward for an agent"""
        reward = torch.zeros(world.batch_dim, device=world.device)

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

    def observation(self, agent: KiyotakiAgent, world: World) -> torch.Tensor:
        """Return observation for an agent"""
        # Local observation (used during execution)
        type_onehot = torch.zeros(3, device=world.device)
        type_onehot[agent.type] = 1

        holding_onehot = torch.zeros(3, device=world.device)
        holding_onehot[agent.holding] = 1

        steps_without_trade_norm = (
            agent.steps_without_trade / self.max_steps_without_trade
        )

        # Global state information (used during training)
        all_agents_types = torch.zeros(len(world.agents) * 3, device=world.device)
        all_agents_holdings = torch.zeros(len(world.agents) * 3, device=world.device)

        for i, other in enumerate(world.agents):
            agent = cast(KiyotakiAgent, other)
            all_agents_types[i * 3 + agent.type] = 1
            all_agents_holdings[i * 3 + agent.holding] = 1

        # Concatenate local and global observations
        obs = torch.cat([
            type_onehot,  # Local: agent's type
            holding_onehot,  # Local: agent's holding
            steps_without_trade_norm.unsqueeze(-1),  # Local: trading history
            all_agents_types,  # Global: all agents' types
            all_agents_holdings,  # Global: all agents' holdings
        ])

        return obs

    def done(self, agent: KiyotakiAgent, world: World) -> torch.Tensor:
        """Return done signal"""
        # Episode ends if agent hasn't traded for too long
        return agent.steps_without_trade >= self.max_steps_without_trade

    def process_action(self, agent: KiyotakiAgent, world: World) -> None:
        """Process agent's action"""
        # Find potential trading partners
        for other in world.agents:
            other = cast(KiyotakiAgent, other)
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
        self, agent: KiyotakiAgent, world: World
    ) -> dict[str, float]:
        """
        Compute welfare metrics including:
        1. Social welfare loss (deviation from Pareto optimal)
        2. Exploitability (how much an agent could gain by deviating)
        """
        # First compute current utility for all agents
        current_utilities = []
        for a in world.agents:
            a = cast(KiyotakiAgent, a)
            # Base utility calculation
            utility = -self.storage_costs[a.holding]  # Storage cost
            if a.holding == a.type:  # If holding preferred good
                utility += self.consumption_values[a.type]
            current_utilities.append(utility)

        current_social_welfare = sum(current_utilities)

        # Compute Pareto optimal welfare
        # In the optimal case, each agent type holds their preferred good
        optimal_utilities: list[float] = []
        for a in world.agents:
            a = cast(KiyotakiAgent, a)
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
                other = cast(KiyotakiAgent, other)
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
        exploitability_pct = (
            (best_deviation_value - current_strategy_value)
            / abs(current_strategy_value)
            * 100
            if current_strategy_value != 0
            else 0
        )

        return {
            "welfare_loss_pct": welfare_loss_pct,
            "exploitability_pct": exploitability_pct,
        }

    def log_metrics(self, world: World) -> dict[str, float]:
        """
        Compute and log aggregate welfare metrics for the entire economy
        """
        total_welfare_loss = 0.0
        total_exploitability = 0.0

        for agent in world.agents:
            agent = cast(KiyotakiAgent, agent)
            metrics = self.compute_welfare_metrics(agent, world)
            total_welfare_loss += metrics["welfare_loss_pct"]
            total_exploitability += metrics["exploitability_pct"]

        n_agents = len(world.agents)
        return {
            "avg_welfare_loss_pct": total_welfare_loss / n_agents,
            "avg_exploitability_pct": total_exploitability / n_agents,
        }

    def visualize_metrics(self, world: World) -> go.Figure:
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
            a = cast(KiyotakiAgent, agent)
            holdings_by_type[a.type, a.holding] += 1

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
        for i, agent in enumerate(world.agents):
            agent = cast(KiyotakiAgent, agent)
            angle = (2 * np.pi * i) / len(world.agents)
            x = np.cos(angle)
            y = np.sin(angle)
            node_x.append(x)
            node_y.append(y)

        # Add edges for recent trades
        for i, agent in enumerate(world.agents):
            a = cast(KiyotakiAgent, agent)
            if a.just_traded:
                for j, other in enumerate(world.agents):
                    o = cast(KiyotakiAgent, other)
                    if o.just_traded and i < j:
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
        agents = cast(list[KiyotakiAgent], world.agents)
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

    def visualize_state(self, world: World) -> None:
        """Visualize the current state using matplotlib"""
        import matplotlib.pyplot as plt
        import numpy as np

        # Create figure with subplots
        _fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

        # 1. Holdings Distribution Heatmap
        holdings_by_type = torch.zeros((3, 3), device=world.device)
        for agent in world.agents:
            agent = cast(KiyotakiAgent, agent)
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
            agent = cast(KiyotakiAgent, agent)
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
