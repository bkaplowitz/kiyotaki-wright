from __future__ import annotations

from typing import TypedDict, cast

import torch as th
from tensordict import TensorDict, TensorDictBase  # type: ignore[import-untyped]
from torch import Tensor, nn
from torchrl.data import (  # type: ignore[import-untyped]
    LazyTensorStorage,
    OneHot,
    TensorDictReplayBuffer,
)
from torchrl.modules import MLP, EGreedyModule  # type: ignore[import-untyped]
from torchrl.objectives import DQNLoss  # type: ignore[import-untyped]


class KWDQNNet(nn.Module):
    """DQN network for Kiyotaki-Wright agents"""

    def __init__(
        self,
        state_dim: int = 2,  # agent type and inventory
        hidden_dim: int = 128,
        num_actions: int = 2,  # accept or reject trade
    ):
        super().__init__()

        self.net = MLP(
            in_features=state_dim,
            out_features=num_actions,
            num_cells=[hidden_dim, hidden_dim],
            activation_class=nn.ReLU,
        )

    def forward(self, tensordict: TensorDict) -> TensorDict:
        # Combine agent_type and inventory into a single state tensor
        state = th.stack(
            [
                tensordict.get("agent_type").float(),
                tensordict.get("inventory").float(),
            ],
            dim=-1,
        )  # Shape: [batch_size, 2]

        # Get Q-values
        q_values = self.net(state)  # Shape: [batch_size, num_actions]

        # Store Q-values in tensordict

        return TensorDict(
            {
                "q_values": q_values,
                "agent_type": tensordict.get("agent_type"),
                "inventory": tensordict.get("inventory"),
            },
            batch_size=tensordict.batch_size,
        )


class KWAgentStateDict(TypedDict):
    policy_nets: list[dict[str, Tensor]]
    optimizers: list[dict[str, Tensor]]
    steps: int


class KWMultiAgentDQN:
    """
    Multi-agent DQN implementation for Kiyotaki-Wright model
    Each type of agent has its own network but shares parameters within type
    """

    def __init__(
        self,
        num_agent_types: int,
        device: th.device,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 10,
    ):
        self.device = device
        self.num_agent_types = num_agent_types
        self.batch_size = batch_size
        self.batch_dim = 0
        self.action_dim = -1
        self.target_update_freq = target_update_freq

        # Create networks for each agent type
        self.value_nets = nn.ModuleList([
            KWDQNNet().to(device) for _ in range(num_agent_types)
        ])

        # Create target networks
        self.target_nets = nn.ModuleList([
            KWDQNNet().to(device) for _ in range(num_agent_types)
        ])

        # Initialize target networks with value network weights
        for target_net, value_net in zip(
            self.target_nets, self.value_nets, strict=False
        ):
            target_net.load_state_dict(value_net.state_dict())

        # Create optimizers
        self.optimizers = [
            th.optim.Adam(net.parameters(), lr=learning_rate) for net in self.value_nets
        ]

        # Create replay buffers for each agent type
        self.replay_buffers = [
            TensorDictReplayBuffer(
                storage=LazyTensorStorage(buffer_size),
                batch_size=batch_size,
            )
            for _ in range(num_agent_types)
        ]

        # Create epsilon-greedy module with proper spec
        action_spec = OneHot(2)
        self.epsilon_greedy = EGreedyModule(
            spec=action_spec,
            annealing_num_steps=1000,
            eps_init=epsilon_start,
            eps_end=epsilon_end,
        )

        # Create DQN losses
        self.dqn_losses = [
            DQNLoss(
                value_network=value_net,
                loss_function="l2",
                delay_value=True,
                double_dqn=True,  # Enable double DQN
                action_space="one-hot",  # Since we have binary actions
                reduction="mean",
            )
            for value_net in self.value_nets
        ]

        self.steps = 0

    def select_actions(self, tensordict: TensorDictBase) -> th.Tensor:
        """Select actions for all agents using epsilon-greedy policy"""
        with th.no_grad():
            # Ensure tensordict has proper batch dimensions
            if tensordict.batch_dims == 0:
                tensordict = tensordict.unsqueeze(0)

            # Initialize actions with the correct shape to match tensordict
            actions = th.zeros(
                tensordict.get("agent_type").shape, dtype=th.int64, device=self.device
            )

            for agent_type in range(self.num_agent_types):
                # Get indices for this agent type
                type_mask = tensordict.get("agent_type") == agent_type
                if not type_mask.any():
                    continue

                # Create type-specific tensordict
                type_td = TensorDict(
                    {
                        "agent_type": tensordict.get("agent_type")[type_mask].squeeze(
                            0
                        ),
                        "inventory": tensordict.get("inventory")[type_mask].squeeze(0),
                    },
                    batch_size=th.Size([cast(int, type_mask.sum().item())]),
                    device=self.device,  # Add device
                )

                # Get Q-values
                type_td = self.value_nets[agent_type](type_td)

                # Convert Q-values to action probabilities
                q_values = type_td.get("q_values")
                greedy_actions = q_values.argmax(dim=-1)

                # Convert to one-hot format for epsilon-greedy
                one_hot_actions = th.zeros(
                    (*greedy_actions.shape, 2), dtype=th.int64, device=self.device
                )
                one_hot_actions.scatter_(-1, greedy_actions.unsqueeze(-1), 1)

                # Add action to tensordict for epsilon-greedy
                type_td.set("action", one_hot_actions)

                # Apply epsilon-greedy exploration
                type_td = self.epsilon_greedy(type_td)

                # Convert back from one-hot to indices and assign to the correct positions
                actions[type_mask] = (
                    type_td.get("action").argmax(dim=-1).to(self.device)
                )

            return actions.squeeze()  # Remove batch dimension if present

    def update(self) -> list[float] | None:
        """Update all agent networks using their respective replay buffers"""
        losses = []

        # Update each agent type's network
        for agent_type in range(self.num_agent_types):
            if len(self.replay_buffers[agent_type]) < self.batch_size:
                continue

            # Sample from replay buffer
            samples = self.replay_buffers[agent_type].sample()

            # Compute loss
            loss = self.dqn_losses[agent_type](samples)
            losses.append(loss.item())

            # Optimize
            self.optimizers[agent_type].zero_grad()
            loss.backward()
            self.optimizers[agent_type].step()

            # Update target network
            self.dqn_losses[agent_type].update_target_params()

        self.steps += 1
        return losses if losses else None

    def add_to_buffer(self, tensordict: TensorDictBase) -> None:
        """Add experiences to appropriate replay buffers"""
        for agent_type in range(self.num_agent_types):
            type_mask = tensordict.get("agent_type") == agent_type
            if not type_mask.any():
                continue

            self.replay_buffers[agent_type].add(tensordict[type_mask])

    def state_dict(self) -> KWAgentStateDict:
        return {
            "policy_nets": [net.state_dict() for net in self.value_nets],
            "optimizers": [opt.state_dict() for opt in self.optimizers],
            "steps": self.steps,
        }

    def load_state_dict(self, state_dict: KWAgentStateDict) -> None:
        for _i, (net, state) in enumerate(
            zip(self.value_nets, state_dict["policy_nets"], strict=False)
        ):
            net.load_state_dict(state)
        for _, (opt, state) in enumerate(
            zip(self.optimizers, state_dict["optimizers"], strict=False)
        ):
            opt.load_state_dict(state)
        self.steps = state_dict["steps"]

    def _set_batch_size(self, batch_size: int | None) -> None:
        self._batch_size = batch_size
