from __future__ import annotations

import torch
from benchmarl import Experiment, ExperimentConfig
from benchmarl.algorithms import MappoConfig
from benchmarl.models import MlpConfig
from benchmarl.tasks import VmasTask
from matplotlib import pyplot as plt


def train_benchmarl() -> None:
    """Train using BenchMARL's implementation of MAPPO"""
    # Create experiment configuration
    experiment = Experiment(
        task=VmasTask.NAVIGATION.get_from_yaml(),
        algorithm_config=MappoConfig.get_from_yaml(),
        model_config=MlpConfig.get_from_yaml(),
        critic_model_config=MlpConfig.get_from_yaml(),
        seed=0,
        config=ExperimentConfig.get_from_yaml(),
    )

    # Override some default configurations
    experiment.config.max_n_iters = 10
    experiment.config.on_policy_collected_frames_per_batch = 6000
    experiment.config.num_epochs = 30
    experiment.config.minibatch_size = 400
    experiment.config.lr = 3e-4
    experiment.config.max_grad_norm = 1.0
    experiment.config.gamma = 0.99
    experiment.config.gae_lambda = 0.9
    experiment.config.entropy_eps = 1e-4
    experiment.config.clip_epsilon = 0.2

    # Set device
    if torch.cuda.is_available():
        experiment.config.train_device = "cuda:0"
        experiment.config.collect_device = "cuda:0"
    else:
        experiment.config.train_device = "cpu"
        experiment.config.collect_device = "cpu"

    # Run experiment
    experiment.run()

    # Plot results using experiment's logging data
    plot_results(experiment)


def plot_results(experiment: Experiment) -> None:
    """Plot training results"""
    plt.figure(figsize=(10, 5))
    # Get episode rewards from experiment logs
    rewards = experiment.get_metric_values("episode_reward")
    plt.plot(rewards)
    plt.xlabel("Training iterations")
    plt.ylabel("Episode reward")
    plt.title("Training Progress")
    plt.show()


if __name__ == "__main__":
    train_benchmarl()
