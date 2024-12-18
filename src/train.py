import json
import logging
from pathlib import Path

import numpy as np
import plotly.graph_objects as go  # type: ignore[import-untyped]
import torch
from tensordict import TensorDict  # type: ignore[import-untyped]
from tqdm import tqdm

from src.kw_agent import KWMultiAgentDQN  # type: ignore[import-untyped]
from src.kw_environment import KWEnvironment, KWParams  # type: ignore[import-untyped]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_training_results(
    rewards: np.ndarray, save_path: str | None = None
) -> go.Figure:
    """Plot training results"""
    episodes = list(range(len(rewards)))

    fig = go.Figure()

    for i in range(rewards.shape[1]):
        fig.add_trace(
            go.Scatter(
                x=episodes, y=rewards[:, i], mode="lines", name=f"Type {i} Rewards"
            )
        )

    fig.update_layout(
        title="Training Rewards by Agent Type",
        xaxis_title="Episode",
        yaxis_title="Average Reward",
        showlegend=True,
    )

    if save_path:
        fig.write_html(save_path)
    return fig


def analyze_equilibrium(
    env: KWEnvironment, agent: KWMultiAgentDQN, n_episodes: int = 100
) -> dict:
    """Analyze the equilibrium strategies of trained agents"""
    inventory_distributions = []
    trade_matrices = torch.zeros(
        (env.params.n_types, env.params.n_types), device=agent.device
    )

    for _ in range(n_episodes):
        td = TensorDict(
            env.reset(),
            batch_size=torch.Size([env.params.n_types * env.params.n_agents]),
        )
        actions = agent.select_actions(td)
        td["action"] = actions

        # Record trades
        for i in range(0, len(actions), 2):
            if i + 1 >= len(actions):
                break

            if actions[i] and actions[i + 1]:
                type1 = td.get("agent_type")[i]
                type2 = td.get("agent_type")[i + 1]
                trade_matrices[type1, type2] += 1
                trade_matrices[type2, type1] += 1

        env.step(td)
        inventory_distributions.append(env.get_inventory_distribution())

    return {
        "inventory_distribution": torch.stack(inventory_distributions)
        .mean(0)
        .cpu()
        .numpy(),
        "trade_matrix": (trade_matrices / n_episodes).cpu().numpy(),
    }


def train(
    env: KWEnvironment,
    agent: KWMultiAgentDQN,
    n_episodes: int,
    episode_length: int = 1000,
) -> list:
    """Train the agent"""
    episode_rewards = []
    td = TensorDict(env.reset(), batch_size=env.batch_size)
    for episode in tqdm(range(n_episodes), desc="Training"):
        episode_reward = torch.zeros(env.params.n_types, device=agent.device)
        td = TensorDict(env.reset(), batch_size=env.batch_size)
        # Run episode
        for _ in range(episode_length):
            # Select actions
            actions = agent.select_actions(td)
            td.set("action", actions)

            # Step environment
            next_td = env.step(tensordict=td)

            # Store experience
            agent.add_to_buffer(td)

            # Update rewards
            rewards = next_td.get("reward")
            for agent_type in range(env.params.n_types):
                type_mask = td.get("agent_type") == agent_type
                episode_reward[agent_type] += rewards[type_mask].mean()

            td = next_td

        # Update networks
        losses = agent.update()

        # Log progress
        mean_rewards = episode_reward / episode_length
        episode_rewards.append(mean_rewards.cpu().numpy())

        if (episode + 1) % 100 == 0:
            logger.info(
                f"Episode {episode + 1}/{n_episodes} - "
                f"Rewards: {mean_rewards.cpu().numpy()} - "
                f"Loss: {np.mean(losses) if losses else 'N/A'}"
            )

    return episode_rewards


def main() -> None:
    """Main training function"""
    # Create output directory
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Initialize environment
    params = KWParams()
    env = KWEnvironment(params)
    env.to(device)

    # Initialize agent
    agent = KWMultiAgentDQN(
        num_agent_types=params.n_types,
        device=device,
        learning_rate=1e-4,
        gamma=params.discount_factor,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
    )

    # Train
    n_episodes = 1000
    logger.info("Starting training...")
    rewards = train(env, agent, n_episodes)

    # Plot results
    plot_training_results(np.array(rewards), str(output_dir / "training_results.html"))

    # Analyze equilibrium
    logger.info("\nAnalyzing equilibrium...")
    equilibrium = analyze_equilibrium(env, agent)

    logger.info("\nFinal Inventory Distribution:")
    logger.info(equilibrium["inventory_distribution"])
    logger.info("\nTrade Matrix:")
    logger.info(equilibrium["trade_matrix"])

    # Save results
    torch.save(agent.state_dict(), output_dir / "agent.pt")

    with open(output_dir / "equilibrium_analysis.json", "w") as f:
        json.dump(
            {
                "inventory_distribution": equilibrium[
                    "inventory_distribution"
                ].tolist(),
                "trade_matrix": equilibrium["trade_matrix"].tolist(),
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
