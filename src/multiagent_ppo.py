import torch
from matplotlib import pyplot as plt
from tensordict.nn import TensorDictModule  # type: ignore[import-untyped]
from tensordict.nn.distributions import (  # type: ignore[import-untyped]
    NormalParamExtractor,
)
from torch import multiprocessing  # type: ignore[import-untyped]

# Data collection
from torchrl.collectors import SyncDataCollector  # type: ignore[import-untyped]
from torchrl.data.replay_buffers import ReplayBuffer  # type: ignore[import-untyped]
from torchrl.data.replay_buffers.samplers import (  # type: ignore[import-untyped]
    SamplerWithoutReplacement,
)
from torchrl.data.replay_buffers.storages import (  # type: ignore[import-untyped]
    LazyTensorStorage,
)

# Env
from torchrl.envs import RewardSum, TransformedEnv  # type: ignore[import-untyped]
from torchrl.envs.libs.vmas import VmasEnv  # type: ignore[import-untyped]
from torchrl.envs.utils import check_env_specs  # type: ignore[import-untyped]

# Multi-agent network
from torchrl.modules import (  # type: ignore[import-untyped]
    MultiAgentMLP,
    ProbabilisticActor,
    TanhNormal,
)

# Loss
from torchrl.objectives import (  # type: ignore[import-untyped]
    ClipPPOLoss,
    ValueEstimators,
)
from tqdm import tqdm

# Utils
torch.manual_seed(0)

# Devices
is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)
vmas_device = device  # The device where the simulator is run

# Sampling hyperparameters
frames_per_batch = 6_000  # Number of frames collected per training iteration
n_iters = 10  # Number of sampling and training iterations
total_frames = frames_per_batch * n_iters

# Training hyperparameters
num_epochs = 30  # Number of optimization steps per training iteration
minibatch_size = 400  # Size of minibatches
lr = 3e-4  # Learning rate
max_grad_norm = 1.0  # Maximum norm for gradients

# PPO hyperparameters
clip_epsilon = 0.2  # Clip value for PPO loss
gamma = 0.99  # Discount factor
lmbda = 0.9  # Lambda for GAE
entropy_eps = 1e-4  # Entropy coefficient


def train_vmas() -> tuple[TransformedEnv, ProbabilisticActor]:
    # Create environment
    max_steps = 100  # Episode steps before done
    num_vmas_envs = frames_per_batch // max_steps
    scenario_name = "navigation"
    n_agents = 3

    env = VmasEnv(
        scenario=scenario_name,
        num_envs=num_vmas_envs,
        continuous_actions=True,
        max_steps=max_steps,
        device=vmas_device,
        n_agents=n_agents,
    )

    # Add reward sum transform
    env = TransformedEnv(
        env,
        RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
    )

    # Verify environment specs
    check_env_specs(env)

    # Create policy network
    share_parameters_policy = True
    policy_net = torch.nn.Sequential(
        MultiAgentMLP(
            n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
            n_agent_outputs=2 * env.action_spec.shape[-1],
            n_agents=env.n_agents,
            centralised=False,
            share_params=share_parameters_policy,
            device=device,
            depth=2,
            num_cells=256,
            activation_class=torch.nn.Tanh,
        ),
        NormalParamExtractor(),
    )

    policy_module = TensorDictModule(
        policy_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "loc"), ("agents", "scale")],
    )

    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.unbatched_action_spec,
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[env.action_key],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env.unbatched_action_spec[env.action_key].space.low,
            "high": env.unbatched_action_spec[env.action_key].space.high,
        },
        return_log_prob=True,
        log_prob_key=("agents", "sample_log_prob"),
    )

    # Create critic network
    share_parameters_critic = True
    mappo = True  # IPPO if False

    critic_net = MultiAgentMLP(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
        n_agent_outputs=1,
        n_agents=env.n_agents,
        centralized=mappo,
        share_params=share_parameters_critic,
        device=device,
        depth=2,
        num_cells=256,
        activation_class=torch.nn.Tanh,
    )

    critic = TensorDictModule(
        module=critic_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "state_value")],
    )

    # Create collector
    collector = SyncDataCollector(
        env,
        policy,
        device=vmas_device,
        storing_device=device,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
    )

    # Create replay buffer
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(frames_per_batch, device=device),
        sampler=SamplerWithoutReplacement(),
        batch_size=minibatch_size,
    )

    # Create loss module
    loss_module = ClipPPOLoss(
        actor_network=policy,
        critic_network=critic,
        clip_epsilon=clip_epsilon,
        entropy_coef=entropy_eps,
        normalize_advantage=False,
    )

    loss_module.set_keys(
        reward=env.reward_key,
        action=env.action_key,
        sample_log_prob=("agents", "sample_log_prob"),
        value=("agents", "state_value"),
        done=("agents", "done"),
        terminated=("agents", "terminated"),
    )

    loss_module.make_value_estimator(ValueEstimators.GAE, gamma=gamma, lmbda=lmbda)
    GAE = loss_module.value_estimator

    optim = torch.optim.Adam(loss_module.parameters(), lr)

    # Training loop
    episode_rewards = []
    pbar = tqdm(total=n_iters, desc="Training")

    for tensordict_data in collector:
        # Expand done and terminated to match reward shape
        tensordict_data.set(
            ("next", "agents", "done"),
            tensordict_data.get(("next", "done"))
            .unsqueeze(-1)
            .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
        )
        tensordict_data.set(
            ("next", "agents", "terminated"),
            tensordict_data.get(("next", "terminated"))
            .unsqueeze(-1)
            .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
        )

        # Compute GAE
        with torch.no_grad():
            GAE(
                tensordict_data,
                params=loss_module.critic_network_params,
                target_params=loss_module.target_critic_network_params,
            )

        # Flatten batch for shuffling
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view)

        # Training epochs
        for _ in range(num_epochs):
            for _ in range(frames_per_batch // minibatch_size):
                subdata = replay_buffer.sample()
                loss_vals = loss_module(subdata)
                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                optim.step()
                optim.zero_grad()

        # Update policy weights
        collector.update_policy_weights_()

        # Logging
        done = tensordict_data.get(("next", "agents", "done"))
        episode_reward = (
            tensordict_data.get(("next", "agents", "episode_reward"))[done]
            .mean()
            .item()
        )
        episode_rewards.append(episode_reward)
        pbar.set_description(f"episode_reward = {episode_reward:.2f}")
        pbar.update()

    pbar.close()

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.xlabel("Training iterations")
    plt.ylabel("Episode reward")
    plt.title("Training Progress")
    plt.show()

    return env, policy


def render_policy(
    env: TransformedEnv,
    policy: ProbabilisticActor,
    num_episodes: int = 1,
) -> None:
    """Render policy behavior in the environment."""
    print(f"\nEpisode {num_episodes}")

    for episode in range(num_episodes):
        print(f"Starting episode {episode + 1}")
        tensordict = env.reset()
        done = False
        steps = 0
        max_steps = 100  # Add safety limit

        # Verify shapes match before starting
        observation = tensordict[("agents", "observation")]
        if observation.shape[-2] != env.n_agents:
            raise ValueError(
                f"Observation shape mismatch. Expected {env.n_agents} agents, "
                f"got shape {observation.shape}"
            )

        while not done and steps < max_steps:
            with torch.no_grad():
                tensordict = policy(tensordict)
            tensordict = env.step(tensordict)
            done = tensordict["done"].any().item()
            env.render()
            steps += 1

        print(f"Episode {episode + 1} completed in {steps} steps")

    env.close()


def train_kiyotaki_wright(
    frames_per_batch: int = 6_000,
    n_iters: int = 50,
    visualize_every: int = 5,
) -> tuple[TransformedEnv, ProbabilisticActor]:
    """Train agents in Kiyotaki-Wright economy and visualize results"""
    import matplotlib.pyplot as plt

    from src.environments.kiyotaki_wright import KiyotakiWrightScenario

    # Create environment
    scenario = KiyotakiWrightScenario()

    # Initialize the world first
    world = scenario.make_world(
        batch_dim=frames_per_batch // 100,
        device=device,
        n_agents=9,
        types=[0, 1, 2],
    )
    scenario._world = world  # Store reference to world

    env = VmasEnv(
        scenario=scenario,
        num_envs=frames_per_batch // 100,  # Adjust based on max_steps
        continuous_actions=True,
        max_steps=100,
        device=device,
    )

    # Add reward sum transform
    env = TransformedEnv(
        env,
        RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
    )

    # Verify environment specs
    check_env_specs(env)

    # Create policy network
    share_parameters_policy = True
    policy_net = torch.nn.Sequential(
        MultiAgentMLP(
            n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
            n_agent_outputs=2 * env.action_spec.shape[-1],
            n_agents=env.n_agents,
            centralised=False,
            share_params=share_parameters_policy,
            device=device,
            depth=2,
            num_cells=256,
            activation_class=torch.nn.Tanh,
        ),
        NormalParamExtractor(),
    )

    policy_module = TensorDictModule(
        policy_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "loc"), ("agents", "scale")],
    )

    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.unbatched_action_spec,
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[env.action_key],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env.unbatched_action_spec[env.action_key].space.low,
            "high": env.unbatched_action_spec[env.action_key].space.high,
        },
        return_log_prob=True,
        log_prob_key=("agents", "sample_log_prob"),
    )

    # Create critic network
    share_parameters_critic = True
    mappo = True  # IPPO if False

    critic_net = MultiAgentMLP(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
        n_agent_outputs=1,
        n_agents=env.n_agents,
        centralized=mappo,
        share_params=share_parameters_critic,
        device=device,
        depth=2,
        num_cells=256,
        activation_class=torch.nn.Tanh,
    )

    critic = TensorDictModule(
        module=critic_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "state_value")],
    )

    # Create collector
    collector = SyncDataCollector(
        env,
        policy,
        device=device,
        storing_device=device,
        frames_per_batch=frames_per_batch,
        total_frames=frames_per_batch * n_iters,
    )

    # Create replay buffer
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(frames_per_batch, device=device),
        sampler=SamplerWithoutReplacement(),
        batch_size=minibatch_size,
    )

    # Create loss module
    loss_module = ClipPPOLoss(
        actor_network=policy,
        critic_network=critic,
        clip_epsilon=clip_epsilon,
        entropy_coef=entropy_eps,
        normalize_advantage=False,
    )

    loss_module.set_keys(
        reward=env.reward_key,
        action=env.action_key,
        sample_log_prob=("agents", "sample_log_prob"),
        value=("agents", "state_value"),
        done=("agents", "done"),
        terminated=("agents", "terminated"),
    )

    loss_module.make_value_estimator(ValueEstimators.GAE, gamma=gamma, lmbda=lmbda)
    GAE = loss_module.value_estimator

    optim = torch.optim.Adam(loss_module.parameters(), lr)

    # Training metrics
    welfare_losses = []
    exploitability_scores = []

    # Training loop
    pbar = tqdm(total=n_iters, desc="Training")

    for i, tensordict_data in enumerate(collector):
        # Expand done and terminated to match reward shape
        tensordict_data.set(
            ("next", "agents", "done"),
            tensordict_data.get(("next", "done"))
            .unsqueeze(-1)
            .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
        )
        tensordict_data.set(
            ("next", "agents", "terminated"),
            tensordict_data.get(("next", "terminated"))
            .unsqueeze(-1)
            .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
        )

        # Compute GAE
        with torch.no_grad():
            GAE(
                tensordict_data,
                params=loss_module.critic_network_params,
                target_params=loss_module.target_critic_network_params,
            )

        # Flatten batch for shuffling
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view)

        # Training epochs
        for _ in range(num_epochs):
            for _ in range(frames_per_batch // minibatch_size):
                subdata = replay_buffer.sample()
                loss_vals = loss_module(subdata)
                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                optim.step()
                optim.zero_grad()

        # Update policy weights
        collector.update_policy_weights_()

        # Collect metrics
        metrics = scenario.log_metrics(env.env.world)
        welfare_losses.append(metrics["avg_welfare_loss_pct"])
        exploitability_scores.append(metrics["avg_exploitability_pct"])

        # Visualize periodically
        if i % visualize_every == 0:
            plt.close("all")  # Close all figures

            # Show metrics visualization
            scenario.visualize_state(env.env.world)

            # Plot training progress
            plt.figure(figsize=(10, 5))
            plt.plot(welfare_losses, label="Welfare Loss %")
            plt.plot(exploitability_scores, label="Exploitability %")
            plt.xlabel("Training Iterations")
            plt.ylabel("Percentage")
            plt.title("Training Progress")
            plt.legend()
            plt.show()

        pbar.update()

    pbar.close()
    return env, policy


if __name__ == "__main__":
    env, policy = train_kiyotaki_wright()
    print("\nTraining complete!")
