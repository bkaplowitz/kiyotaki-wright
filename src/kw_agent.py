import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer, TensorSpec
from torchrl.modules import MLP, EGreedyModule
from torchrl.objectives import DQNLoss


# TODO: Add Dreamer
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
        """Forward pass of the network"""
        state = torch.stack(
            [tensordict["agent_type"].float(), tensordict["inventory"].float()],
            dim=-1,
        )
        q_values = self.net(state)
        tensordict.set("q_values", q_values)
        return tensordict


class KWMultiAgentDQN:
    """
    Multi-agent DQN implementation for Kiyotaki-Wright model
    Each type of agent has its own network but shares parameters within type
    """

    def __init__(
        self,
        num_agent_types: int,
        device: torch.device,
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
        self.target_update_freq = target_update_freq

        # Create networks for each agent type
        self.policy_nets = nn.ModuleList([
            KWDQNNet().to(device) for _ in range(num_agent_types)
        ])

        self.target_nets = nn.ModuleList([
            KWDQNNet().to(device) for _ in range(num_agent_types)
        ])

        # Initialize target networks
        for policy_net, target_net in zip(
            self.policy_nets, self.target_nets, strict=False
        ):
            target_net.load_state_dict(policy_net.state_dict())

        # Create optimizers
        self.optimizers = [
            torch.optim.Adam(net.parameters(), lr=learning_rate)
            for net in self.policy_nets
        ]

        # Create replay buffers for each agent type
        self.replay_buffers = [
            TensorDictReplayBuffer(
                storage=LazyTensorStorage(buffer_size),
                batch_size=batch_size,
            )
            for _ in range(num_agent_types)
        ]

        # Create epsilon-greedy modules with proper parameters
        action_spec = TensorSpec(
            shape=(),  # scalar actions
            dtype=torch.int64,
            device=device,
        )

        # Calculate annealing steps from decay rate
        annealing_steps = int(-1 / np.log(epsilon_decay))  # Convert decay rate to steps

        self.epsilon_greedy = EGreedyModule(
            spec=action_spec,
            eps_init=epsilon_start,
            eps_end=epsilon_end,
            annealing_num_steps=annealing_steps,
        )

        # Create DQN losses with correct parameter names
        self.dqn_losses = [
            DQNLoss(
                value_network=policy_net,
                target_network=target_net,
                gamma=gamma,
            )
            for policy_net, target_net in zip(
                self.policy_nets, self.target_nets, strict=False
            )
        ]

        self.steps = 0

    def select_actions(self, tensordict: TensorDict) -> torch.Tensor:
        """Select actions for all agents using epsilon-greedy policy"""
        with torch.no_grad():
            actions = torch.zeros(
                tensordict.batch_size[0], dtype=torch.int64, device=self.device
            )

            for agent_type in range(self.num_agent_types):
                type_mask = tensordict["agent_type"] == agent_type
                if not type_mask.any():
                    continue

                type_td = tensordict[type_mask]
                type_td = self.policy_nets[agent_type](type_td)
                type_td = self.epsilon_greedy(type_td)
                actions[type_mask] = type_td["action"]

            return actions

    def update(self) -> list[float] | None:
        """Update all agent networks using their respective replay buffers"""
        losses = []

        for agent_type in range(self.num_agent_types):
            if len(self.replay_buffers[agent_type]) < self.batch_size:
                continue

            samples = self.replay_buffers[agent_type].sample()
            loss = self.dqn_losses[agent_type](samples)
            losses.append(loss.item())

            self.optimizers[agent_type].zero_grad()
            loss.backward()
            self.optimizers[agent_type].step()

        return losses if losses else None

    def add_to_buffer(self, tensordict: TensorDict) -> None:
        """Add experiences to appropriate replay buffers"""
        for agent_type in range(self.num_agent_types):
            type_mask = tensordict["agent_type"] == agent_type
            if not type_mask.any():
                continue
            self.replay_buffers[agent_type].add(tensordict[type_mask])

    def state_dict(self):
        return {
            "policy_nets": [net.state_dict() for net in self.policy_nets],
            "target_nets": [net.state_dict() for net in self.target_nets],
            "optimizers": [opt.state_dict() for opt in self.optimizers],
            "steps": self.steps,
        }

    def load_state_dict(self, state_dict):
        for _i, (net, state) in enumerate(
            zip(self.policy_nets, state_dict["policy_nets"], strict=False)
        ):
            net.load_state_dict(state)
        for _i, (net, state) in enumerate(
            zip(self.target_nets, state_dict["target_nets"], strict=False)
        ):
            net.load_state_dict(state)
        for _i, (opt, state) in enumerate(
            zip(self.optimizers, state_dict["optimizers"], strict=False)
        ):
            opt.load_state_dict(state)
        self.steps = state_dict["steps"]
