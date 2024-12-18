import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any
from collections import deque
import random

from kw_environment import KWEnvironment, KWParams
from kw_agent import DQN


class KWLightningModule(pl.LightningModule):
    def __init__(
        self,
        env: KWEnvironment,
        hidden_dim: int = 128,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_update: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["env"])
        self.env = env

        # Create policy and target networks for each agent type
        self.policy_nets = nn.ModuleList([
            DQN(state_dim=2, action_dim=2, hidden_dim=hidden_dim)
            for _ in range(env.params.n_types)
        ])
        self.target_nets = nn.ModuleList([
            DQN(state_dim=2, action_dim=2, hidden_dim=hidden_dim)
            for _ in range(env.params.n_types)
        ])

        # Initialize target networks
        for policy_net, target_net in zip(self.policy_nets, self.target_nets):
            target_net.load_state_dict(policy_net.state_dict())

        # Create replay buffers for each agent type
        self.replay_buffers = [
            deque(maxlen=buffer_size) for _ in range(env.params.n_types)
        ]

        self.epsilon = epsilon_start
        self.steps = 0
        self.episode_rewards = []
        self.current_episode_rewards = np.zeros(env.params.n_types)
        self.inventory_distributions = []

    def forward(self, x):
        # Forward is for inference only
        return [net(x) for net in self.policy_nets]

    def _get_actions(self, state: np.ndarray) -> np.ndarray:
        actions = np.zeros(len(state), dtype=int)

        for i, policy_net in enumerate(self.policy_nets):
            agent_indices = self.env.agent_types == i
            agent_states = state[agent_indices]

            for j, agent_state in enumerate(agent_states):
                if random.random() > self.epsilon:
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(agent_state).unsqueeze(0)
                        q_values = policy_net(state_tensor)
                        action = q_values.max(1)[1].item()
                else:
                    action = random.randint(0, 1)
                actions[agent_indices][j] = action

        return actions

    def training_step(self, batch, batch_idx):
        # Run one episode
        state = self.env.reset()
        episode_length = 1000
        total_loss = 0

        for _ in range(episode_length):
            actions = self._get_actions(state)
            next_state, rewards, done, info = self.env.step(actions)

            # Store transitions for each agent type
            for i in range(self.env.params.n_types):
                agent_indices = self.env.agent_types == i
                agent_states = state[agent_indices]
                agent_next_states = next_state[agent_indices]
                agent_actions = actions[agent_indices]
                agent_rewards = rewards[agent_indices]

                for s, a, r, ns in zip(
                    agent_states, agent_actions, agent_rewards, agent_next_states
                ):
                    self.replay_buffers[i].append((s, a, r, ns, done))

                self.current_episode_rewards[i] += np.mean(agent_rewards)

            # Train on replay buffer
            if all(
                len(buffer) >= self.hparams.batch_size for buffer in self.replay_buffers
            ):
                loss = self._train_on_buffers()
                total_loss += loss

            state = next_state
            if done:
                break

        # Update target networks
        self.steps += 1
        if self.steps % self.hparams.target_update == 0:
            for policy_net, target_net in zip(self.policy_nets, self.target_nets):
                target_net.load_state_dict(policy_net.state_dict())

        # Update epsilon
        self.epsilon = max(
            self.hparams.epsilon_end, self.epsilon * self.hparams.epsilon_decay
        )

        # Log metrics
        mean_rewards = self.current_episode_rewards / episode_length
        self.episode_rewards.append(mean_rewards)
        self.inventory_distributions.append(info["inventory_distribution"])

        for i, reward in enumerate(mean_rewards):
            self.log(f"reward_type_{i}", reward, prog_bar=True)
        self.log("epsilon", self.epsilon, prog_bar=True)
        self.log("loss", total_loss / episode_length)

        self.current_episode_rewards = np.zeros(self.env.params.n_types)

        return {"loss": total_loss / episode_length, "rewards": mean_rewards}

    def _train_on_buffers(self) -> float:
        total_loss = 0

        for i, (policy_net, target_net, replay_buffer) in enumerate(
            zip(self.policy_nets, self.target_nets, self.replay_buffers)
        ):
            # Sample from replay buffer
            batch = random.sample(replay_buffer, self.hparams.batch_size)
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(
                *batch
            )

            state_batch = torch.FloatTensor(np.array(state_batch))
            action_batch = torch.LongTensor(action_batch)
            reward_batch = torch.FloatTensor(reward_batch)
            next_state_batch = torch.FloatTensor(np.array(next_state_batch))
            done_batch = torch.FloatTensor(done_batch)

            # Compute current Q values
            current_q_values = policy_net(state_batch).gather(
                1, action_batch.unsqueeze(1)
            )

            # Compute next Q values
            with torch.no_grad():
                next_q_values = target_net(next_state_batch).max(1)[0]
                next_q_values[done_batch.bool()] = 0.0
                expected_q_values = reward_batch + self.hparams.gamma * next_q_values

            # Compute loss
            loss = nn.MSELoss()(current_q_values.squeeze(), expected_q_values)
            total_loss += loss.item()

            # Optimize
            self.manual_backward(loss)

        return total_loss

    def configure_optimizers(self):
        optimizers = [
            torch.optim.Adam(net.parameters(), lr=self.hparams.learning_rate)
            for net in self.policy_nets
        ]
        return optimizers

    def get_metrics(self) -> Dict[str, Any]:
        """Get training metrics for analysis"""
        return {
            "rewards": np.array(self.episode_rewards),
            "inventory_distributions": np.array(self.inventory_distributions),
        }
