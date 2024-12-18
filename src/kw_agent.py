import tensordict
import torch
import torch.nn as nn
from torchrl.data import (
    LazyTensorStorage,
    OneHotDiscreteTensorSpec,
    TensorDictReplayBuffer,
)
from torchrl.modules import MLP, EGreedyModule
from torchrl.objectives import DQNLoss


# TODO: Dreamer later
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

    def forward(self, tensordict: tensordict.TensorDict):
        # Combine agent_type and inventory into a single state tensor
        state = torch.stack(
            [tensordict.get("agent_type").float(), tensordict.get("inventory").float()],
            dim=-1,
        )

        # Get Q-values
        q_values = self.net(state)

        # Store Q-values in tensordict
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

        # Create epsilon-greedy module with proper spec
        action_spec = OneHotDiscreteTensorSpec(2)  # 2 actions: accept or reject
        self.epsilon_greedy = EGreedyModule(
            spec=action_spec,
            annealing_num_steps=1000,  # Adjust this based on your needs
            eps_init=epsilon_start,
            eps_end=epsilon_end,
        )

        # Create DQN losses
        self.dqn_losses = [
            DQNLoss(
                value_network=policy_net,
                gamma=gamma,
                double_dqn=True,  # Enable double DQN
            )
            for policy_net in self.policy_nets
        ]

        self.steps = 0

    def select_actions(self, tensordict: tensordict.TensorDict) -> torch.Tensor:
        """Select actions for all agents using epsilon-greedy policy"""
        with torch.no_grad():
            actions = torch.zeros(
                tensordict.batch_size, dtype=torch.int64, device=self.device
            )

            for agent_type in range(self.num_agent_types):
                # Get indices for this agent type
                type_mask = tensordict.get("agent_type") == agent_type
                if not type_mask.any():
                    continue

                # Get Q-values for this agent type
                type_td = tensordict[type_mask]
                type_td = self.policy_nets[agent_type](type_td)

                # Select actions using epsilon-greedy
                type_td = self.epsilon_greedy(type_td)
                actions[type_mask] = type_td.get("action")

            return actions

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

        # Update target networks
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            for policy_net in self.policy_nets:
                policy_net.load_state_dict(policy_net.state_dict())

        return losses if losses else None

    def add_to_buffer(self, tensordict: tensordict.TensorDict):
        """Add experiences to appropriate replay buffers"""
        for agent_type in range(self.num_agent_types):
            type_mask = tensordict.get("agent_type") == agent_type
            if not type_mask.any():
                continue

            self.replay_buffers[agent_type].add(tensordict[type_mask])

    def state_dict(self):
        return {
            "policy_nets": [net.state_dict() for net in self.policy_nets],
            "optimizers": [opt.state_dict() for opt in self.optimizers],
            "steps": self.steps,
        }

    def load_state_dict(self, state_dict):
        for _, (net, state) in enumerate(
            zip(self.policy_nets, state_dict["policy_nets"], strict=False)
        ):
            net.load_state_dict(state)
        for _, (opt, state) in enumerate(
            zip(self.optimizers, state_dict["optimizers"], strict=False)
        ):
            opt.load_state_dict(state)
        self.steps = state_dict["steps"]
